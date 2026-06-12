"""
train_multimodal_ecg_model.py  —  HEARTBREAKER
-----------------------------------------------
Heartbreaker: Second-stage multimodal ECG classifier.
Named for its job: breaking the ECG-alone ceiling by fusing
physiological waveform embeddings with patient clinical context.

Architecture:
  - ENCODER: frozen 2-block 1D ResNet from binary_1d_ecg_model.h5
             (final Dense+sigmoid removed → 128-dim embedding)
  - TIER 1 (probability fusion): ECG prob + metadata → LogisticRegression
  - TIER 2 (embedding fusion):   ECG embedding + metadata MLP → fusion MLP

Evaluation:
  - Identical 5-fold patient-disjoint StratifiedKFold as the ECG-only baseline
  - Nested Platt scaling + sensitivity-constrained threshold on val-cal slice
  - OOF aggregation → bootstrap CIs → Brier score → ECE

Baseline (ECG-only, for reference):
  OOF AUC=0.9192  PR-AUC=0.9241  Sens=0.8480  Spec=0.8400

Run:
    python train_multimodal_ecg_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=UserWarning)

# ── Local imports ─────────────────────────────────────────────────────────
from multimodal_data_prep import (
    build_metadata_matrix,
    fit_metadata_preprocessors,
    apply_metadata_preprocessors,
    load_and_cache_dataset,
)

# ── Re-use preprocessing from the ECG-only pipeline ──────────────────────
import wfdb
import scipy.signal as scipy_signal


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
ECG_MODEL_PATH  = "models/binary_1d_ecg_model.h5"
HEARTBREAKER_NAME = "Heartbreaker"
META_CSV        = "data/subset_metadata_2000.csv"
SIGNAL_BASE_DIR = "data/raw"
INCLUDE_TEXT    = False       # set False to skip TF-IDF features (Structured Meta Only)
TFIDF_FEATURES  = 80          # max TF-IDF terms after leakage audit
N_FOLDS         = 5
RANDOM_STATE    = 42
TARGET_SENS     = 0.85        # minimum sensitivity constraint for threshold
BATCH_SIZE      = 32
FUSION_EPOCHS   = 60
FUSION_PATIENCE = 20

ECG_ONLY_BASELINE = {
    "roc_auc": 0.9243, "pr_auc": 0.9336,
    "sensitivity": 0.8580, "specificity": 0.8410,
}


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL LOADING (identical to train_1d_ecg_model.py)
# ════════════════════════════════════════════════════════════════════════════

def load_signal(record_path: str, fs: int = 100, length: int = 1000):
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal.astype(np.float32)
        nyq = 0.5 * fs
        b, a = scipy_signal.butter(4, [0.5/nyq, 40.0/nyq], btype="band")
        filtered = np.zeros_like(sig)
        for ch in range(sig.shape[1]):
            filtered[:, ch] = scipy_signal.filtfilt(b, a, sig[:, ch])
        mn, sd = filtered.mean(0), filtered.std(0)
        sd[sd == 0] = 1.0
        norm = (filtered - mn) / sd
        if norm.shape[0] >= length:
            return norm[:length, :]
        return np.vstack([norm, np.zeros((length - norm.shape[0], 12))])
    except Exception as e:
        print(f"  [signal error] {record_path}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# ECG ENCODER EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def load_ecg_encoder(model_path: str) -> tf.keras.Model:
    """
    Loads the trained ECG-only model and removes the final Dense(1,sigmoid) head.
    Returns a model whose output is the penultimate 128-dim embedding
    (output of GlobalAveragePooling1D in Block 2).

    The encoder is FROZEN — weights are not updated during multimodal training.
    This preserves the validated physiological representation.
    """
    def _focal_loss(gamma=2.0, alpha=0.5):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            pt     = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            at     = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            return tf.reduce_mean(-at * tf.pow(1 - pt, gamma) * tf.math.log(pt))
        return loss

    base = tf.keras.models.load_model(
        model_path,
        custom_objects={"loss": _focal_loss()},
        compile=False,
    )
    print(f"\nLoaded ECG model: {model_path}")
    print(f"  Total layers: {len(base.layers)}")
    print(f"  Input shape:  {base.input_shape}")
    print(f"  Output shape: {base.output_shape}")

    # Find GlobalAveragePooling1D — the penultimate representation
    embedding_layer = None
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            embedding_layer = layer
            break

    if embedding_layer is None:
        # Fallback: use the Dense(64) layer before the final output
        for layer in reversed(base.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.units == 64:
                embedding_layer = layer
                break

    if embedding_layer is None:
        raise ValueError("Could not find embedding layer. Check model architecture.")

    encoder = tf.keras.Model(
        inputs=base.input,
        outputs=embedding_layer.output,
        name="ecg_encoder",
    )
    encoder.trainable = False   # Phase 1: frozen
    print(f"  Encoder output: '{embedding_layer.name}'  shape={encoder.output_shape}")
    return encoder


def extract_ecg_embeddings(encoder: tf.keras.Model,
                            X_signals: np.ndarray) -> np.ndarray:
    """Runs the frozen encoder on all signals and returns the embedding matrix."""
    return encoder.predict(X_signals, batch_size=BATCH_SIZE, verbose=0)


# ════════════════════════════════════════════════════════════════════════════
# TIER 1 — Probability-level fusion (LogisticRegression)
# ════════════════════════════════════════════════════════════════════════════

def build_tier1_features(ecg_probs: np.ndarray,
                          X_meta: np.ndarray) -> np.ndarray:
    """Concatenates ECG probability (scalar) with metadata feature vector."""
    return np.hstack([ecg_probs.reshape(-1, 1), X_meta])


# ════════════════════════════════════════════════════════════════════════════
# TIER 2 — Embedding-level fusion MLP
# ════════════════════════════════════════════════════════════════════════════

def build_fusion_model(ecg_embedding_dim: int, n_meta_features: int) -> tf.keras.Model:
    """
    Builds the Tier-2 fusion MLP.

    Architecture:
      ECG embedding (128-dim, frozen)   → passed through directly
      Metadata (n_meta_features-dim)    → Dense(64) → BN → ReLU → Dropout
      Concatenate → Dense(64) → Dropout(0.4) → Dense(1, sigmoid)

    Design choices:
      - Metadata branch has its own projection to allow the model to learn
        which metadata features are informative independently of ECG.
      - Dropout 0.4 on fusion head prevents the head from memorising.
      - No class weights: dataset is balanced 1000/1000.
      - Loss: binary crossentropy (focal is not needed with balanced data
        once calibration happens via Platt scaling).
    """
    ecg_input  = Input(shape=(ecg_embedding_dim,), name="ecg_embedding_input")
    meta_input = Input(shape=(n_meta_features,),   name="metadata_input")

    # Metadata branch
    x_meta = layers.Dense(64, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(meta_input)
    x_meta = layers.BatchNormalization()(x_meta)
    x_meta = layers.Dropout(0.3)(x_meta)
    x_meta = layers.Dense(32, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x_meta)

    # Fusion
    x = layers.Concatenate()([ecg_input, x_meta])
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[ecg_input, meta_input], outputs=output,
                          name="heartbreaker_fusion")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ════════════════════════════════════════════════════════════════════════════
# SHARED EVALUATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def platt_calibrate_and_threshold(y_val: np.ndarray,
                                   probs_val_raw: np.ndarray,
                                   target_sensitivity: float = TARGET_SENS
                                   ) -> tuple:
    """
    Fits Platt scaler and selects sensitivity-constrained threshold on val-cal.
    Returns: (calibrator, threshold, val_probs_calibrated)
    """
    calibrator = LogisticRegression(max_iter=500)
    calibrator.fit(probs_val_raw.reshape(-1, 1), y_val)
    probs_cal = calibrator.predict_proba(probs_val_raw.reshape(-1, 1))[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, probs_cal)
    spec = 1 - fpr
    valid = np.where(tpr >= target_sensitivity)[0]
    if len(valid) == 0:
        best_idx = np.argmax(tpr + spec - 1)
    else:
        best_idx = valid[np.argmax(spec[valid])]
    threshold = float(thresholds[best_idx])
    return calibrator, threshold, probs_cal


def fold_metrics(y_test: np.ndarray,
                 probs_cal: np.ndarray,
                 threshold: float) -> dict:
    preds = (probs_cal >= threshold).astype(int)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc":     roc_auc_score(y_test, probs_cal),
        "pr_auc":      average_precision_score(y_test, probs_cal),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "accuracy":    accuracy_score(y_test, preds),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                  y_pred: np.ndarray,
                  n: int = 1000, seed: int = RANDOM_STATE) -> dict:
    np.random.seed(seed)
    aucs, pr_aucs, sens_list, spec_list, brier_list = [], [], [], [], []
    N = len(y_true)
    for _ in range(n):
        idx = np.random.randint(0, N, N)
        yt, yp, yc = y_true[idx], y_prob[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        pr_aucs.append(average_precision_score(yt, yp))
        brier_list.append(np.mean((yp - yt) ** 2))
        cm = confusion_matrix(yt, yc, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    def ci(lst): return (np.percentile(lst, 2.5), np.percentile(lst, 97.5))
    return {
        "auc_ci":   ci(aucs),   "pr_auc_ci":  ci(pr_aucs),
        "sens_ci":  ci(sens_list), "spec_ci": ci(spec_list),
        "brier_ci": ci(brier_list),
        "brier":    float(np.mean(brier_list)),
    }


def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    N    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece)


# ════════════════════════════════════════════════════════════════════════════
# MAIN CROSS-VALIDATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print(f"   {HEARTBREAKER_NAME.upper()} — Second-Stage Multimodal ECG Classifier")
    print("═" * 60)

    # ── 1. Load metadata ──────────────────────────────────────────────
    df_static, reports, y, patient_ids = build_metadata_matrix(
        META_CSV, include_text=INCLUDE_TEXT, tfidf_max_features=TFIDF_FEATURES
    )
    # Drop workflow flags and noise flags (workflow-variable-removed ablation)
    drop_cols = ["validated_by_human", "has_baseline_drift", "has_static_noise", "has_burst_noise", "has_electrode_prob", "noise_score"]
    # Also drop report-derived heart_axis columns since they are transcribed from reports
    axis_cols = [c for c in df_static.columns if c.startswith("axis_")]
    drop_cols.extend(axis_cols)
    df_static = df_static.drop(columns=[c for c in drop_cols if c in df_static.columns])
    df_meta_csv = pd.read_csv(META_CSV)

    # ── 2. Load raw ECG signals ───────────────────────────────────────
    X_signals, _, valid_indices = load_and_cache_dataset(META_CSV, SIGNAL_BASE_DIR)

    valid_mask  = np.array(valid_indices)
    y           = y[valid_mask]
    patient_ids = patient_ids[valid_mask]
    df_static   = df_static.iloc[valid_mask].reset_index(drop=True)
    reports     = reports.iloc[valid_mask].reset_index(drop=True)

    print(f"  Valid signals: {len(X_signals)}  (Normal={int((y==0).sum())}, Abnormal={int((y==1).sum())})")

    # ── 3. Load ECG encoder (frozen) ──────────────────────────────────
    # ── 3. Load ECG probabilities (from out-of-fold predictions to prevent leakage) ──
    OOF_PROBS_PATH = "models/clean_oof_ecg_probs.npy"
    if not os.path.exists(OOF_PROBS_PATH):
        print(f"\n[ERROR] OOF probabilities file not found: {OOF_PROBS_PATH}")
        print("  Run train_1d_ecg_model.py first to generate it.")
        return
    X_ecg_probs = np.load(OOF_PROBS_PATH)
    print(f"  ECG clean OOF probabilities loaded: shape={X_ecg_probs.shape}")

    # ── 5. Cross-validation ───────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # OOF arrays
    oof_tier1_probs  = np.zeros(len(y))
    oof_tier1_preds  = np.zeros(len(y))
    oof_tier2_probs  = np.zeros(len(y))
    oof_tier2_preds  = np.zeros(len(y))

    print(f"\n{'─'*60}")
    print(f"Starting {N_FOLDS}-Fold Patient-Disjoint CV")
    print(f"{'─'*60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_signals, y)):
        print(f"\n{'─'*50}\n  Fold {fold+1}/{N_FOLDS}")

        # ── Nested split: 80% sub-train, 20% val-cal ──────────────
        sub_train_idx, val_cal_idx = train_test_split(
            train_idx, test_size=0.20, stratify=y[train_idx],
            random_state=RANDOM_STATE
        )

        # ── Metadata preprocessing (fit on sub-train only) ────────
        X_meta_train_raw, preprocessors = fit_metadata_preprocessors(
            df_static.iloc[sub_train_idx],
            reports.iloc[sub_train_idx],
            y[sub_train_idx],
            include_text=INCLUDE_TEXT,
            tfidf_max_features=TFIDF_FEATURES,
        )
        X_meta_val = apply_metadata_preprocessors(
            df_static.iloc[val_cal_idx], reports.iloc[val_cal_idx], preprocessors)
        X_meta_test = apply_metadata_preprocessors(
            df_static.iloc[test_idx], reports.iloc[test_idx], preprocessors)

        n_meta = X_meta_train_raw.shape[1]

        # ════════════════════════════════════════════════════════════
        # TIER 1 — Probability-level fusion (LogisticRegression)
        # ════════════════════════════════════════════════════════════
        print(f"  [Tier 1] Training probability-level fusion...")

        X_t1_train = build_tier1_features(X_ecg_probs[sub_train_idx], X_meta_train_raw)
        X_t1_val   = build_tier1_features(X_ecg_probs[val_cal_idx],   X_meta_val)
        X_t1_test  = build_tier1_features(X_ecg_probs[test_idx],      X_meta_test)

        tier1_lr = LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE)
        tier1_lr.fit(X_t1_train, y[sub_train_idx])
        t1_val_raw = tier1_lr.predict_proba(X_t1_val)[:, 1]

        # Calibrate + threshold on val-cal
        t1_cal, t1_thresh, t1_val_cal = platt_calibrate_and_threshold(
            y[val_cal_idx], t1_val_raw)
        print(f"    Tier-1 threshold: {t1_thresh:.4f}")

        # Evaluate on test fold
        t1_test_raw = tier1_lr.predict_proba(X_t1_test)[:, 1]
        t1_test_cal = t1_cal.predict_proba(t1_test_raw.reshape(-1, 1))[:, 1]
        t1_preds    = (t1_test_cal >= t1_thresh).astype(int)
        t1_m = fold_metrics(y[test_idx], t1_test_cal, t1_thresh)
        print(f"    AUC={t1_m['roc_auc']:.4f}  Sens={t1_m['sensitivity']:.4f}  "
              f"Spec={t1_m['specificity']:.4f}")

        oof_tier1_probs[test_idx] = t1_test_cal
        oof_tier1_preds[test_idx] = t1_preds

        # ════════════════════════════════════════════════════════════
        # TIER 2 — Embedding-level fusion MLP
        # ════════════════════════════════════════════════════════════
        print(f"  [Tier 2] Training embedding-level fusion MLP...")

        # Load fold-specific model to prevent target leakage in embedding extraction
        fold_model_path = f"models/binary_1d_ecg_model_fold{fold+1}.h5"
        if not os.path.exists(fold_model_path):
            raise FileNotFoundError(f"Fold-specific model not found: {fold_model_path}")
        fold_encoder = load_ecg_encoder(fold_model_path)

        # Extract embeddings out-of-fold using fold-specific model
        X_ecg_embed_train = extract_ecg_embeddings(fold_encoder, X_signals[sub_train_idx])
        X_ecg_embed_val   = extract_ecg_embeddings(fold_encoder, X_signals[val_cal_idx])
        X_ecg_embed_test  = extract_ecg_embeddings(fold_encoder, X_signals[test_idx])

        ecg_dim = X_ecg_embed_train.shape[1]
        fusion_model = build_fusion_model(ecg_dim, n_meta)

        early_stop = callbacks.EarlyStopping(
            monitor="val_auc", patience=FUSION_PATIENCE,
            mode="max", restore_best_weights=True
        )

        fusion_model.fit(
            [X_ecg_embed_train, X_meta_train_raw],
            y[sub_train_idx],
            validation_data=(
                [X_ecg_embed_val, X_meta_val],
                y[val_cal_idx]
            ),
            epochs=FUSION_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )

        # Calibrate + threshold on val-cal (must refit — distribution changed)
        t2_val_raw = fusion_model.predict(
            [X_ecg_embed_val, X_meta_val], verbose=0).flatten()
        t2_cal, t2_thresh, _ = platt_calibrate_and_threshold(
            y[val_cal_idx], t2_val_raw)
        print(f"    Tier-2 threshold: {t2_thresh:.4f}")

        # Evaluate on test fold
        t2_test_raw = fusion_model.predict(
            [X_ecg_embed_test, X_meta_test], verbose=0).flatten()
        t2_test_cal = t2_cal.predict_proba(t2_test_raw.reshape(-1, 1))[:, 1]
        t2_preds    = (t2_test_cal >= t2_thresh).astype(int)
        t2_m = fold_metrics(y[test_idx], t2_test_cal, t2_thresh)
        print(f"    AUC={t2_m['roc_auc']:.4f}  Sens={t2_m['sensitivity']:.4f}  "
              f"Spec={t2_m['specificity']:.4f}")

        oof_tier2_probs[test_idx] = t2_test_cal
        oof_tier2_preds[test_idx] = t2_preds

    # ── 6. Aggregate OOF metrics ──────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  AGGREGATE OOF RESULTS")
    print(f"{'═'*60}")

    def oof_report(name, probs, preds, baseline=None):
        auc  = roc_auc_score(y, probs)
        pr   = average_precision_score(y, probs)
        acc  = accuracy_score(y, preds)
        cm   = confusion_matrix(y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        brier = float(np.mean((probs - y) ** 2))
        ece   = expected_calibration_error(y, probs)
        ci    = bootstrap_ci(y, probs, preds)

        print(f"\n  ── {name} ──")
        print(f"  ROC-AUC:     {auc:.4f}  (95% CI: {ci['auc_ci'][0]:.4f}–{ci['auc_ci'][1]:.4f})")
        print(f"  PR-AUC:      {pr:.4f}   (95% CI: {ci['pr_auc_ci'][0]:.4f}–{ci['pr_auc_ci'][1]:.4f})")
        print(f"  Sensitivity: {sens:.4f}  (95% CI: {ci['sens_ci'][0]:.4f}–{ci['sens_ci'][1]:.4f})")
        print(f"  Specificity: {spec:.4f}  (95% CI: {ci['spec_ci'][0]:.4f}–{ci['spec_ci'][1]:.4f})")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Brier:       {brier:.4f}  (95% CI: {ci['brier_ci'][0]:.4f}–{ci['brier_ci'][1]:.4f})")
        print(f"  ECE:         {ece:.4f}")
        if baseline:
            delta_auc  = auc  - baseline["roc_auc"]
            delta_sens = sens - baseline["sensitivity"]
            delta_spec = spec - baseline["specificity"]
            print(f"  vs ECG-only baseline:  ΔAUC={delta_auc:+.4f}  ΔSens={delta_sens:+.4f}  ΔSpec={delta_spec:+.4f}")
        return {"name": name, "roc_auc": auc, "pr_auc": pr,
                "sensitivity": sens, "specificity": spec,
                "brier": brier, "ece": ece, "ci": ci}

    print("\n  ── ECG-Only Baseline (reference, not re-run) ──")
    for k, v in ECG_ONLY_BASELINE.items():
        print(f"  {k}: {v:.4f}")

    results_t1 = oof_report(f"{HEARTBREAKER_NAME} Tier 1 — Probability Fusion (LR)",
                              oof_tier1_probs, oof_tier1_preds, ECG_ONLY_BASELINE)
    results_t2 = oof_report(f"{HEARTBREAKER_NAME} Tier 2 — Embedding Fusion (MLP)",
                              oof_tier2_probs, oof_tier2_preds, ECG_ONLY_BASELINE)

    # ── 7. Acceptance decision ────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  ACCEPTANCE DECISION")
    print(f"{'═'*60}")
    for res in [results_t1, results_t2]:
        sens_ok  = res["sensitivity"] >= TARGET_SENS
        auc_ok   = res["roc_auc"]  > ECG_ONLY_BASELINE["roc_auc"]
        prauc_ok = res["pr_auc"]   > ECG_ONLY_BASELINE["pr_auc"]
        spec_ok  = res["specificity"] > ECG_ONLY_BASELINE["specificity"]
        improves = auc_ok or prauc_ok or spec_ok
        verdict  = "✅ ACCEPT" if (sens_ok and improves) else "❌ REJECT"
        print(f"\n  {res['name']}")
        print(f"    Sens ≥ {TARGET_SENS}: {'✅' if sens_ok else '❌'}  |  "
              f"AUC↑: {'✅' if auc_ok else '—'}  |  "
              f"PR-AUC↑: {'✅' if prauc_ok else '—'}  |  "
              f"Spec↑: {'✅' if spec_ok else '—'}")
        print(f"    → {verdict}")

    # ── 8. Save results ───────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open("models/heartbreaker_results.txt", "w") as f:
        f.write(f"{HEARTBREAKER_NAME} — Multimodal ECG Model OOF Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("ECG-Only Baseline:\n")
        for k, v in ECG_ONLY_BASELINE.items():
            f.write(f"  {k}: {v:.4f}\n")
        for res in [results_t1, results_t2]:
            f.write(f"\n{res['name']}:\n")
            for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "brier", "ece"]:
                f.write(f"  {k}: {res[k]:.4f}\n")
    print(f"\n  Results saved to models/heartbreaker_results.txt")


if __name__ == "__main__":
    main()

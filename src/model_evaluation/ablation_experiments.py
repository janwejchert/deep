"""
ablation_experiments.py
-----------------------
Runs 6 experiments varying class weight and validation threshold target.
Goal: find the best sensitivity-specificity balance.

Experiment grid:
  A: weight 1:4, sens >= 0.90  (baseline)
  B: weight 1:3, sens >= 0.85
  C: weight 1:2, sens >= 0.85
  D: weight 1:2, sens >= 0.80
  E: weight 1:1, Youden-J
  F: weight 1:1, sens >= 0.80
"""

import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             average_precision_score, roc_curve)

# ─── Signal loading ────────────────────────────────────────────────────────────

def load_and_preprocess_signal(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        fs = 100
        nyq = 0.5 * fs
        b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
        filtered = np.zeros_like(sig)
        for i in range(sig.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, sig[:, i])
        mean = np.mean(filtered, axis=0)
        std  = np.std(filtered, axis=0)
        std[std == 0] = 1.0
        norm = (filtered - mean) / std
        if norm.shape[0] >= 1000:
            norm = norm[:1000, :]
        else:
            norm = np.vstack([norm, np.zeros((1000 - norm.shape[0], 12))])
        return norm
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        return None

# ─── Model ────────────────────────────────────────────────────────────────────

def build_1d_cnn():
    inp = layers.Input(shape=(1000, 12))
    x = layers.Conv1D(32,  15, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64,  11, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128,  7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(256,  5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return m

# ─── Threshold selection ──────────────────────────────────────────────────────

def select_threshold(y_val, y_prob, min_sensitivity=None):
    """
    If min_sensitivity is given: among thresholds achieving >= min_sensitivity
    on the validation set, pick the one maximising specificity.
    If no threshold meets the target (or min_sensitivity is None): Youden-J.
    """
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    spec = 1 - fpr
    if min_sensitivity is not None:
        valid = np.where(tpr >= min_sensitivity)[0]
        if len(valid) > 0:
            best = valid[np.argmax(spec[valid])]
            return float(thresholds[best])
    # Youden-J fallback
    j = tpr + spec - 1
    return float(thresholds[np.argmax(j)])

# ─── Single experiment ────────────────────────────────────────────────────────

def run_experiment(X, y, class_weight_val, min_sensitivity, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, pr_aucs, accs, sens_l, spec_l, thresh_l = [], [], [], [], [], []

    print(f"\n{'='*60}")
    print(f"Experiment {label}  |  weight 1:{int(class_weight_val)}  |  "
          f"sens_target={min_sensitivity if min_sensitivity else 'Youden-J'}")
    print(f"{'='*60}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_sub, X_val, y_sub, y_val = train_test_split(
            X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=42)

        cw = {0: 1.0, 1: float(class_weight_val)}
        mdl = build_1d_cnn()
        es  = callbacks.EarlyStopping(monitor='val_auc', patience=15,
                                      mode='max', restore_best_weights=True)
        mdl.fit(X_sub, y_sub, epochs=40, batch_size=32,
                validation_data=(X_val, y_val),
                class_weight=cw, callbacks=[es], verbose=0)

        y_val_prob = mdl.predict(X_val, verbose=0).flatten()
        thr = select_threshold(y_val, y_val_prob, min_sensitivity)
        thresh_l.append(thr)

        y_te_prob = mdl.predict(X_te, verbose=0).flatten()
        y_te_cls  = (y_te_prob >= thr).astype(int)

        auc  = roc_auc_score(y_te, y_te_prob)
        prauc = average_precision_score(y_te, y_te_prob)
        acc  = accuracy_score(y_te, y_te_cls)
        tn, fp, fn, tp = confusion_matrix(y_te, y_te_cls).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        aucs.append(auc); pr_aucs.append(prauc); accs.append(acc)
        sens_l.append(sens); spec_l.append(spec)
        print(f"  Fold {fold+1}: thr={thr:.4f}  AUC={auc:.4f}  PR={prauc:.4f}"
              f"  Sens={sens:.4f}  Spec={spec:.4f}")

    result = dict(
        experiment=label,
        weight=class_weight_val,
        sens_target=min_sensitivity,
        roc_auc=f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        pr_auc=f"{np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}",
        accuracy=f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
        sensitivity=f"{np.mean(sens_l):.4f} ± {np.std(sens_l):.4f}",
        specificity=f"{np.mean(spec_l):.4f} ± {np.std(spec_l):.4f}",
        mean_roc=np.mean(aucs),
        mean_sens=np.mean(sens_l),
        mean_spec=np.mean(spec_l),
    )
    print(f"\n  SUMMARY → ROC-AUC {result['roc_auc']}  "
          f"Sens {result['sensitivity']}  Spec {result['specificity']}")
    return result

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    metadata_path = 'data/subset_metadata_2000.csv'
    base_dir      = 'data/raw'

    df = pd.read_csv(metadata_path)
    X, y = [], []
    print("Loading signals...")
    for _, row in df.iterrows():
        p = os.path.join(base_dir, row['filename_lr'])
        if os.path.exists(p + '.dat'):
            sig = load_and_preprocess_signal(p)
            if sig is not None:
                X.append(sig)
                y.append(0 if row['class'] == 'Normal' else 1)
    X, y = np.array(X), np.array(y)
    print(f"Loaded {len(X)} records  |  Normal={np.sum(y==0)}  Abnormal={np.sum(y==1)}\n")

    # ── Experiment grid ──────────────────────────────────────────────────────
    experiments = [
        ('A', 4.0, 0.90),   # baseline
        ('B', 3.0, 0.85),
        ('C', 2.0, 0.85),
        ('D', 2.0, 0.80),
        ('E', 1.0, None),   # Youden-J
        ('F', 1.0, 0.80),
    ]

    results = []
    for label, weight, sens_tgt in experiments:
        r = run_experiment(X, y, weight, sens_tgt, label)
        results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("ABLATION SUMMARY TABLE")
    print("="*80)
    header = f"{'Exp':>4} {'Weight':>8} {'SensTgt':>9} {'ROC-AUC':>18} {'PR-AUC':>18} {'Sensitivity':>18} {'Specificity':>18}"
    print(header)
    print("-"*80)
    for r in results:
        tgt = f"{r['sens_target']:.2f}" if r['sens_target'] else "Youden"
        print(f"{r['experiment']:>4} {r['weight']:>8.1f} {tgt:>9} "
              f"{r['roc_auc']:>18} {r['pr_auc']:>18} "
              f"{r['sensitivity']:>18} {r['specificity']:>18}")

    # Best by balanced (sens + spec) / 2
    best = max(results, key=lambda r: (r['mean_sens'] + r['mean_spec']) / 2)
    print(f"\n★  Best balanced experiment: {best['experiment']}  "
          f"(weight 1:{int(best['weight'])}, "
          f"target={best['sens_target'] if best['sens_target'] else 'Youden-J'})")
    print(f"   ROC-AUC {best['roc_auc']}  "
          f"Sens {best['sensitivity']}  Spec {best['specificity']}")

    # Save results
    os.makedirs('reports', exist_ok=True)
    with open('reports/ablation_results.txt', 'w') as f:
        f.write("ECG 1D-CNN Ablation Experiments\n")
        f.write("="*80 + "\n")
        f.write(header + "\n" + "-"*80 + "\n")
        for r in results:
            tgt = f"{r['sens_target']:.2f}" if r['sens_target'] else "Youden"
            f.write(f"{r['experiment']:>4} {r['weight']:>8.1f} {tgt:>9} "
                    f"{r['roc_auc']:>18} {r['pr_auc']:>18} "
                    f"{r['sensitivity']:>18} {r['specificity']:>18}\n")
        f.write(f"\nBest balanced: Experiment {best['experiment']}\n")
        f.write(f"ROC-AUC {best['roc_auc']} | "
                f"Sens {best['sensitivity']} | Spec {best['specificity']}\n")
    print("\nResults saved to reports/ablation_results.txt")

if __name__ == '__main__':
    main()

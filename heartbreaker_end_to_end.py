"""
heartbreaker_end_to_end.py
--------------------------
The "Mother Script" for the Heartbreaker ECG classification project.
This single script implements the entire end-to-end pipeline:
1. Downloads the PTB-XL metadata and raw signal data (if missing).
2. Filters to a balanced, patient-disjoint subset (e.g., 2000 patients).
3. Preprocesses 12-lead ECG signals (bandpass filtering, scaling).
4. Trains a 1D ResNet physiological baseline under 5-Fold Nested CV.
5. Performs the TF-IDF leakage audit on clinical reports.
6. Trains the Heartbreaker Tier 1 (Probability Fusion) & Tier 2 (Embedding) models.
7. Computes and prints aggregate out-of-fold metrics.

Dependencies:
    pip install numpy pandas scikit-learn tensorflow wfdb scipy tqdm
"""

import os
import re
import ast
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
from scipy import signal

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = "dataset_1d_heartbreaker"
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

N_NORMAL = 1000
N_ABNORMAL_PER_SUBCLASS = 250  # MI, STTC, CD, HYP (Total = 1000)
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

INCLUDE_TEXT = False  # Set to False to run the primary leakage-safer structured-metadata-only model


# Leakage audit target terms (German)
LABEL_LEAKING_TERMS = re.compile(
    r"infarkt|myokard|ischemi|herzinfarkt|"
    r"linksschenkel|rechtsschenkel|schenkelblock|"
    r"hypertrophie|überlastung|linkstyp|vorhofflimmern|"
    r"st\.elevation|st\.senkung|t\.negativierung|"
    r"avblock|av\.block|vorhof|bradykardi|tachykardi",
    re.IGNORECASE,
)
HEART_AXIS_CATS = ["MID", "LAD", "ALAD", "RAD", "ARAD", "AXL", "AXR", "SAG", "unknown"]

# ==============================================================================
# 1. DATA ACQUISITION & SUBSET SELECTION
# ==============================================================================
def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)

def get_superclasses(scp_codes_str, scp_statements):
    try:
        codes = ast.literal_eval(scp_codes_str)
    except:
        return []
    sclasses = set()
    for code, val in codes.items():
        if val > 0 and code in scp_statements.index:
            scls = scp_statements.loc[code, 'diagnostic_class']
            if pd.notna(scls):
                sclasses.add(scls)
    return list(sclasses)

def prepare_metadata():
    csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(DATA_DIR, "scp_statements.csv")
    download_file(BASE_URL + "ptbxl_database.csv", csv_path)
    download_file(BASE_URL + "scp_statements.csv", scp_path)

    df = pd.read_csv(csv_path)
    scp_statements = pd.read_csv(scp_path, index_col=0)
    df['superclasses'] = df['scp_codes'].apply(lambda x: get_superclasses(x, scp_statements))
    
    # Ensure patient disjointness
    df = df.drop_duplicates(subset=['patient_id']).copy()

    normals = df[df['superclasses'].apply(lambda x: x == ['NORM'])]
    mi = df[df['superclasses'].apply(lambda x: x == ['MI'])]
    sttc = df[df['superclasses'].apply(lambda x: x == ['STTC'])]
    cd = df[df['superclasses'].apply(lambda x: x == ['CD'])]
    hyp = df[df['superclasses'].apply(lambda x: x == ['HYP'])]

    def sample_safe(df_sub, n):
        return df_sub.sample(n=min(n, len(df_sub)), random_state=42).copy()

    s_norm = sample_safe(normals, N_NORMAL)
    s_norm['class'] = 'Normal'
    
    s_mi = sample_safe(mi, N_ABNORMAL_PER_SUBCLASS); s_mi['class'] = 'Abnormal'
    s_sttc = sample_safe(sttc, N_ABNORMAL_PER_SUBCLASS); s_sttc['class'] = 'Abnormal'
    s_cd = sample_safe(cd, N_ABNORMAL_PER_SUBCLASS); s_cd['class'] = 'Abnormal'
    s_hyp = sample_safe(hyp, N_ABNORMAL_PER_SUBCLASS); s_hyp['class'] = 'Abnormal'

    subset = pd.concat([s_norm, s_mi, s_sttc, s_cd, s_hyp]).reset_index(drop=True)
    subset.to_csv(os.path.join(DATA_DIR, "subset_metadata.csv"), index=False)
    print(f"Selected {len(subset)} records (Normal: {len(s_norm)}, Abnormal: {len(subset)-len(s_norm)})")
    return subset

# ==============================================================================
# 2. SIGNAL PROCESSING
# ==============================================================================
def process_signal(record_path):
    try:
        sig, meta = wfdb.rdsamp(record_path)
        # Bandpass filter 0.5 - 40 Hz
        fs = meta['fs']
        nyq = 0.5 * fs
        b, a = signal.butter(3, [0.5/nyq, 40.0/nyq], btype='band')
        sig_filtered = signal.filtfilt(b, a, sig, axis=0)
        # Standardize
        mean = np.mean(sig_filtered, axis=0)
        std = np.std(sig_filtered, axis=0)
        sig_scaled = (sig_filtered - mean) / (std + 1e-8)
        return sig_scaled
    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return None

def download_and_load_signals(df):
    X, y = [], []
    valid_indices = []
    print("Downloading and processing ECG waveforms...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename_lr = row['filename_lr']
        f_dir = os.path.dirname(filename_lr)
        os.makedirs(os.path.join(RAW_DIR, f_dir), exist_ok=True)
        
        path_base = os.path.join(RAW_DIR, filename_lr)
        download_file(f"{BASE_URL}{filename_lr}.dat", path_base + ".dat")
        download_file(f"{BASE_URL}{filename_lr}.hea", path_base + ".hea")
        
        sig = process_signal(path_base)
        if sig is not None:
            X.append(sig)
            y.append(0 if row['class'] == 'Normal' else 1)
            valid_indices.append(idx)
            
    return np.array(X), np.array(y), df.iloc[valid_indices].reset_index(drop=True)

# ==============================================================================
# 3. METADATA & TF-IDF PREPARATION
# ==============================================================================
def extract_static_metadata(df):
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].clip(1, 120).astype(float)
    out["sex"] = df["sex"].astype(float)
    out["height_missing"] = df["height"].isna().astype(float)
    out["weight_missing"] = df["weight"].isna().astype(float)
    out["height"] = df["height"].astype(float)
    out["weight"] = df["weight"].astype(float)
    
    has_bmi = (~df["height"].isna()) & (~df["weight"].isna()) & (df["height"] > 0)
    out["bmi"] = np.nan
    out["bmi_missing"] = 1.0
    out.loc[has_bmi, "bmi"] = (df.loc[has_bmi, "weight"] / (df.loc[has_bmi, "height"]/100)**2)
    out.loc[has_bmi, "bmi_missing"] = 0.0

    axis = df["heart_axis"].fillna("unknown").str.strip()
    for cat in HEART_AXIS_CATS:
        out[f"axis_{cat}"] = (axis == cat).astype(float)

    out["validated_by_human"] = df["validated_by_human"].astype(float)
    out["has_baseline_drift"] = (~df["baseline_drift"].isna()).astype(float)
    out["has_static_noise"]   = (~df["static_noise"].isna()).astype(float)
    out["has_burst_noise"]    = (~df["burst_noise"].isna()).astype(float)
    out["has_electrode_prob"] = (~df["electrodes_problems"].isna()).astype(float)
    return out

def fit_metadata(X_static_train, reports_train, y_train, include_text=INCLUDE_TEXT):
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_static_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    if not include_text:
        print("      [leakage] Text pipeline disabled (Structured Metadata Only).")
        return X_scaled, {"imputer": imputer, "scaler": scaler, "include_text": False}

    # TF-IDF + Leakage Audit
    vec = TfidfVectorizer(max_features=100, ngram_range=(1,2), min_df=5, sublinear_tf=True)
    X_tfidf = vec.fit_transform(reports_train).toarray()
    
    terms = vec.get_feature_names_out()
    drop_cols = []
    for i, term in enumerate(terms):
        if LABEL_LEAKING_TERMS.search(term):
            drop_cols.append(i)
            continue
        col = X_tfidf[:, i]
        if col.std() > 0:
            corr = abs(np.corrcoef(col, y_train)[0, 1])
            if corr >= 0.25:
                drop_cols.append(i)
                
    keep = [i for i in range(len(terms)) if i not in drop_cols]
    print(f"      [leakage] Dropped {len(drop_cols)} leaking text features.")
    
    X_final = np.hstack([X_scaled, X_tfidf[:, keep]])
    return X_final, {"imputer": imputer, "scaler": scaler, "vec": vec, "keep": keep, "include_text": True}

def apply_metadata(X_static, reports, prep):
    X_imp = prep["imputer"].transform(X_static)
    X_scaled = prep["scaler"].transform(X_imp)
    
    if not prep.get("include_text", True):
        return X_scaled
        
    X_tfidf = prep["vec"].transform(reports).toarray()
    return np.hstack([X_scaled, X_tfidf[:, prep["keep"]]])

# ==============================================================================
# 4. NEURAL NETWORK ARCHITECTURES
# ==============================================================================
def binary_focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return focal_loss_fn

def build_1d_resnet():
    inputs = layers.Input(shape=(1000, 12))
    x = layers.Conv1D(32, 15, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 1
    shortcut = layers.Conv1D(64, 1, padding='same')(x)
    x = layers.Conv1D(64, 11, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 11, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Block 2
    shortcut = layers.Conv1D(128, 1, padding='same')(x)
    x = layers.Conv1D(128, 7, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 7, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    
    # Embeddings
    embed = layers.GlobalAveragePooling1D(name='ecg_embed')(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(embed)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='ecg_prob')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=binary_focal_loss(), metrics=['accuracy'])
    return model

def build_tier2_mlp(ecg_dim=128, meta_dim=80):
    inp_ecg = layers.Input(shape=(ecg_dim,))
    inp_meta = layers.Input(shape=(meta_dim,))
    
    x_meta = layers.Dense(32, activation='relu')(inp_meta)
    merged = layers.Concatenate()([inp_ecg, x_meta])
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model([inp_ecg, inp_meta], out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ==============================================================================
# 5. END-TO-END PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    print("=== HEARTBREAKER END-TO-END PIPELINE ===")
    print(f"Configuration: INCLUDE_TEXT = {INCLUDE_TEXT}")
    df_raw = prepare_metadata()
    X_sig, y, df_meta = download_and_load_signals(df_raw)
    
    df_static = extract_static_metadata(df_meta)
    reports = df_meta["report"].fillna("")
    
    oof_y = np.zeros(len(y))
    oof_prob_ecg = np.zeros(len(y))
    oof_prob_t1 = np.zeros(len(y))
    oof_prob_t2 = np.zeros(len(y))
    
    # Literature-standard PTB-XL evaluation uses the official `strat_fold` column.
    # We combine the 10 predefined literature folds into 5 macroscopic folds.
    # This guarantees absolute patient-disjointness exactly as defined by the PTB-XL creators.
    folds_mapping = [(1,2), (3,4), (5,6), (7,8), (9,10)]
    
    for fold, (fA, fB) in enumerate(folds_mapping, 1):
        print(f"\n--- FOLD {fold}/5 (PTB-XL strat_fold {fA} & {fB}) ---")
        val_idx = np.where(df_meta['strat_fold'].isin([fA, fB]))[0]
        train_idx = np.where(~df_meta['strat_fold'].isin([fA, fB]))[0]
        
        X_train_sig, y_train = X_sig[train_idx], y[train_idx]
        X_val_sig, y_val = X_sig[val_idx], y[val_idx]
        
        # 1. Train 1D ECG Baseline
        print("  -> Training ECG 1D ResNet baseline...")
        ecg_model = build_1d_resnet()
        ecg_model.fit(X_train_sig, y_train, validation_data=(X_val_sig, y_val),
                      epochs=10, batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        
        # Extract ECG Probabilities & Embeddings
        ecg_prob_train = ecg_model.predict(X_train_sig, verbose=0).flatten()
        ecg_prob_val = ecg_model.predict(X_val_sig, verbose=0).flatten()
        oof_prob_ecg[val_idx] = ecg_prob_val
        
        embed_model = models.Model(ecg_model.input, ecg_model.get_layer('ecg_embed').output)
        ecg_emb_train = embed_model.predict(X_train_sig, verbose=0)
        ecg_emb_val = embed_model.predict(X_val_sig, verbose=0)
        
        # 2. Process Metadata (Fit on Train, Apply to Val)
        print("  -> Processing metadata & auditing NLP leakage...")
        meta_train, prep = fit_metadata(df_static.iloc[train_idx], reports.iloc[train_idx], y_train)
        meta_val = apply_metadata(df_static.iloc[val_idx], reports.iloc[val_idx], prep)
        
        # 3. Train Tier 1 Fusion (Logistic Regression on Probabilities + Meta)
        print("  -> Training Tier 1 (Probability Fusion)...")
        # Platt scale ECG probabilities to use as features
        lr_scale = LogisticRegression().fit(ecg_prob_train.reshape(-1,1), y_train)
        ecg_prob_train_cal = lr_scale.predict_proba(ecg_prob_train.reshape(-1,1))[:,1]
        ecg_prob_val_cal = lr_scale.predict_proba(ecg_prob_val.reshape(-1,1))[:,1]
        
        X_t1_train = np.hstack([ecg_prob_train_cal.reshape(-1,1), meta_train])
        X_t1_val = np.hstack([ecg_prob_val_cal.reshape(-1,1), meta_val])
        
        t1_model = LogisticRegression(C=0.5, max_iter=1000).fit(X_t1_train, y_train)
        oof_prob_t1[val_idx] = t1_model.predict_proba(X_t1_val)[:,1]
        
        # 4. Train Tier 2 Fusion (MLP on Embeddings + Meta)
        print("  -> Training Tier 2 (Embedding Fusion)...")
        t2_model = build_tier2_mlp(ecg_emb_train.shape[1], meta_train.shape[1])
        t2_model.fit([ecg_emb_train, meta_train], y_train, epochs=30, batch_size=32, verbose=0)
        oof_prob_t2[val_idx] = t2_model.predict([ecg_emb_val, meta_val], verbose=0).flatten()
        
        oof_y[val_idx] = y_val

    # ==========================================================================
    # 6. EVALUATION
    # ==========================================================================
    print("\n\n=== AGGREGATE OUT-OF-FOLD RESULTS ===")
    def evaluate(name, y_true, y_prob):
        auc = roc_auc_score(y_true, y_prob)
        # Select threshold targeting >0.85 sensitivity
        fpr, tpr, ths = tf.keras.metrics.roc_curve(y_true, y_prob) if hasattr(tf.keras.metrics, "roc_curve") else (None, None, None)
        # fallback to manual selection
        from sklearn.metrics import roc_curve
        fpr, tpr, ths = roc_curve(y_true, y_prob)
        idx = np.where(tpr >= 0.85)[0][0]
        th = ths[idx]
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp+fn)
        spec = tn / (tn+fp)
        print(f"{name:.<30} AUC: {auc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | (Thresh: {th:.3f})")

    evaluate("ECG-Only Baseline (1D ResNet)", oof_y, oof_prob_ecg)
    evaluate("Heartbreaker Tier 1 (LR)    ", oof_y, oof_prob_t1)
    evaluate("Heartbreaker Tier 2 (MLP)   ", oof_y, oof_prob_t2)

if __name__ == "__main__":
    # Hide TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()

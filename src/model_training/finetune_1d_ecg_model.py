import os
import numpy as np
import pandas as pd
import ast
import wfdb
import scipy.signal as signal
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_signal(path, length=1000):
    sig, fields = wfdb.rdsamp(path)
    
    fs = 100
    nyq = 0.5 * fs
    b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    
    filtered_sig = np.zeros_like(sig)
    for i in range(sig.shape[1]):
        filtered_sig[:, i] = signal.filtfilt(b, a, sig[:, i])
        
    sig = filtered_sig[:length]
    
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0)
    std[std == 0] = 1.0
    norm_sig = (sig - mean) / std
    
    if norm_sig.shape[0] < length:
        pad = np.zeros((length - norm_sig.shape[0], 12))
        norm_sig = np.vstack([norm_sig, pad])
        
    return norm_sig.astype("float32")

class ECGDataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.augment:
            batch_x = tf.convert_to_tensor(batch_x)
            noise = tf.random.normal(tf.shape(batch_x), stddev=0.01)
            batch_x = batch_x + noise
            
            scale = tf.random.uniform([tf.shape(batch_x)[0], 1, 12], 0.9, 1.1)
            batch_x = batch_x * scale
            
            shift = tf.random.uniform([], -20, 20, dtype=tf.int32)
            batch_x = tf.roll(batch_x, shift, axis=1)
            batch_x = batch_x.numpy()
            
        return batch_x, batch_y

def build_encoder(input_shape=(1000, 12), dim=128):
    inp = keras.Input(input_shape)
    
    x = layers.Conv1D(32, kernel_size=15, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(x)
    
    x = layers.Conv1D(64, kernel_size=11, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, kernel_size=11, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    shortcut = layers.Conv1D(128, kernel_size=1, padding='same')(x)
    
    x = layers.Conv1D(128, kernel_size=7, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=7, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    emb = layers.Dense(dim)(x)
    return keras.Model(inp, emb, name="encoder")

def build_classifier(encoder_weights="models/ecg_ssl_encoder.h5", trainable_backbone=False):
    encoder = build_encoder()
    encoder.load_weights(encoder_weights)
    encoder.trainable = trainable_backbone
    
    inp = keras.Input((1000, 12))
    # Crucial: training=False keeps BatchNorm statistics frozen
    x = encoder(inp, training=False)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inp, out)
    return model, encoder

def boot_ci(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        s.append(fn(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])

def main():
    print("Loading labeled signals...")
    df = pd.read_csv("data/subset_metadata.csv")
    
    X, y, pids = [], [], []
    for i, row in df.iterrows():
        sig = load_signal(f"data/raw/{row['filename_lr']}")
        X.append(sig)
        y.append(1 if row['class'] == 'Abnormal' else 0)
        pids.append(row['patient_id'])
        
    X = np.array(X)
    y = np.array(y)
    pids = np.array(pids)
    
    print(f"Total samples: {len(X)}")
    print(f"Normal: {np.sum(y == 0)}, Abnormal: {np.sum(y == 1)}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_prob = np.zeros(len(y))
    oof_true = y.copy()
    oof_thresholds = []
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=pids)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Nested split for thresholding/calibration
        n_val = len(X_train_full) // 4
        X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
        X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]
        
        # Phase 1: Linear Probe
        model, encoder = build_classifier("models/ecg_ssl_encoder.h5", trainable_backbone=False)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                      metrics=[keras.metrics.AUC(name="auc")])
                      
        train_gen = ECGDataGenerator(X_train, y_train, 32, augment=True)
        val_gen = ECGDataGenerator(X_val, y_val, 32, augment=False)
        
        print("Phase 1: Linear Probing (frozen backbone)")
        model.fit(train_gen, validation_data=val_gen, epochs=50,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)],
                  verbose=0)
                  
        # Phase 2: Conservative fine-tuning
        # Unfreeze the last conv block
        for layer in encoder.layers:
            layer.trainable = isinstance(layer, layers.Conv1D) and layer is encoder.layers[-3]
            
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                      metrics=[keras.metrics.AUC(name="auc")])
                      
        print("Phase 2: Conservative fine-tuning (last conv block unfrozen)")
        model.fit(train_gen, validation_data=val_gen, epochs=30,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)],
                  verbose=0)
                  
        # Platt Scaling Calibration
        val_raw_preds = model.predict(X_val, verbose=0).flatten()
        calibrator = LogisticRegression(random_state=42)
        calibrator.fit(val_raw_preds.reshape(-1, 1), y_val)
        
        val_cal_preds = calibrator.predict_proba(val_raw_preds.reshape(-1, 1))[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_val, val_cal_preds)
        valid_idx = np.where(recall >= 0.85)[0]
        if len(valid_idx) > 0:
            target_idx = valid_idx[np.argmax(precision[valid_idx])]
            thresh = thresholds[min(target_idx, len(thresholds)-1)]
        else:
            thresh = 0.5
            
        print(f"Selected Validation Threshold (Nested): {thresh:.4f}")
        oof_thresholds.append(thresh)
        
        test_raw_preds = model.predict(X_test, verbose=0).flatten()
        test_cal_preds = calibrator.predict_proba(test_raw_preds.reshape(-1, 1))[:, 1]
        oof_prob[test_idx] = test_cal_preds

    avg_thresh = np.mean(oof_thresholds)
    oof_pred = (oof_prob >= avg_thresh).astype(int)
    
    auc = roc_auc_score(oof_true, oof_prob)
    pr_auc = average_precision_score(oof_true, oof_prob)
    acc = np.mean(oof_pred == oof_true)
    tn, fp, fn, tp = confusion_matrix(oof_true, oof_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    auc_ci = boot_ci(oof_true, oof_prob, roc_auc_score)
    pr_auc_ci = boot_ci(oof_true, oof_prob, average_precision_score)
    
    def boot_acc(y, p): return np.mean((p >= avg_thresh).astype(int) == y)
    def boot_sens(y, p):
        cm = confusion_matrix(y, (p >= avg_thresh).astype(int), labels=[0, 1])
        if cm.sum(axis=1)[1] == 0: return np.nan
        return cm[1,1] / cm.sum(axis=1)[1]
    def boot_spec(y, p):
        cm = confusion_matrix(y, (p >= avg_thresh).astype(int), labels=[0, 1])
        if cm.sum(axis=1)[0] == 0: return np.nan
        return cm[0,0] / cm.sum(axis=1)[0]
        
    acc_ci = boot_ci(oof_true, oof_prob, boot_acc)
    sens_ci = boot_ci(oof_true, oof_prob, boot_sens)
    spec_ci = boot_ci(oof_true, oof_prob, boot_spec)
    
    print("\n=====================================")
    print("Aggregate Out-Of-Fold (OOF) Results (N=200):")
    print(f"OOF ROC-AUC: {auc:.4f} (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")
    print(f"OOF PR-AUC: {pr_auc:.4f} (95% CI: {pr_auc_ci[0]:.4f} - {pr_auc_ci[1]:.4f})")
    print(f"OOF Accuracy: {acc:.4f} (95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f})")
    print(f"OOF Sensitivity: {sens:.4f} (95% CI: {sens_ci[0]:.4f} - {sens_ci[1]:.4f})")
    print(f"OOF Specificity: {spec:.4f} (95% CI: {spec_ci[0]:.4f} - {spec_ci[1]:.4f})")
    print("=====================================\n")

if __name__ == '__main__':
    main()

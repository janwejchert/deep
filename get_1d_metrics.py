import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, accuracy_score, f1_score

def load_and_preprocess_signal(record_path):
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal
    fs = 100
    nyq = 0.5 * fs
    b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    filtered_sig = np.zeros_like(sig)
    for i in range(sig.shape[1]):
        filtered_sig[:, i] = signal.filtfilt(b, a, sig[:, i])
    mean = np.mean(filtered_sig, axis=0)
    std = np.std(filtered_sig, axis=0)
    std[std == 0] = 1.0
    norm_sig = (filtered_sig - mean) / std
    if norm_sig.shape[0] >= 1000:
        norm_sig = norm_sig[:1000, :]
    else:
        pad = np.zeros((1000 - norm_sig.shape[0], 12))
        norm_sig = np.vstack([norm_sig, pad])
    return norm_sig

def main():
    metadata_path = 'dataset_1d/subset_metadata.csv'
    df = pd.read_csv(metadata_path)
    base_dir = 'dataset_1d/raw'
    
    X = []
    y = []
    
    for idx, row in df.iterrows():
        record_name = row['filename_lr']
        record_path = os.path.join(base_dir, record_name)
        if os.path.exists(record_path + ".dat"):
            sig = load_and_preprocess_signal(record_path)
            if sig is not None:
                X.append(sig)
                y.append(1 if row['class'] == 'Abnormal' else 0)
                
    X = np.array(X)
    y = np.array(y)
    
    model = tf.keras.models.load_model('binary_1d_ecg_model.h5')
    y_prob = model.predict(X).flatten()
    
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    spec = 1 - fpr
    
    # Youden
    J = tpr + spec - 1
    idx_j = np.argmax(J)
    t_youden = thresholds[idx_j]
    
    # Sens >= 0.8
    idx_80 = np.where(tpr >= 0.80)[0][0]
    t_80 = thresholds[idx_80]
    
    # Sens >= 0.9
    idx_90 = np.where(tpr >= 0.90)[0][0]
    t_90 = thresholds[idx_90]
    
    print("\nOperating Point Table:")
    print("Threshold | Sensitivity | Specificity | Accuracy | F1")
    for t in [0.5, t_youden, t_80, t_90]:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print(f"{t:.4f}    | {se:.4f}      | {sp:.4f}      | {acc:.4f}   | {f1:.4f}")

if __name__ == '__main__':
    main()

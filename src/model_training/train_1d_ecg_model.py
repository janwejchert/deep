import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from multimodal_data_prep import load_and_cache_dataset

def load_and_preprocess_signal(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        
        fs = 100
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        
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
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        return None

def binary_focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss_val = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)
    return loss

class ECGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
        
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = np.copy(self.X[batch_indices])
        y_batch = self.y[batch_indices]
        
        if self.augment:
            for i in range(len(X_batch)):
                if np.random.rand() > 0.5:
                    X_batch[i] += np.random.normal(0, 0.02, X_batch[i].shape)
                    X_batch[i] *= np.random.uniform(0.85, 1.15)
                    shift = np.random.randint(-40, 40)
                    X_batch[i] = np.roll(X_batch[i], shift, axis=0)
                    
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_smaller_resnet_1d():
    inputs = layers.Input(shape=(1000, 12))
    
    # Initial Convolution
    x = layers.Conv1D(32, kernel_size=15, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Residual Block 1
    shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(x)
    
    x = layers.Conv1D(64, kernel_size=11, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, kernel_size=11, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Residual Block 2
    shortcut = layers.Conv1D(128, kernel_size=1, padding='same')(x)
    
    x = layers.Conv1D(128, kernel_size=7, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=7, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classifier
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    # Balanced Focal Loss: alpha=0.5
    model.compile(optimizer='adam', loss=binary_focal_loss(gamma=2.0, alpha=0.5), metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def compute_bootstrap_ci(y_true, y_pred_probs, y_pred_classes, n_bootstraps=1000):
    np.random.seed(42)
    aucs, pr_aucs, accs, sens, specs = [], [], [], [], []
    
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        y_true_b = y_true[indices]
        y_prob_b = y_pred_probs[indices]
        y_class_b = y_pred_classes[indices]
        
        # Only compute if both classes are present in the bootstrap sample
        if len(np.unique(y_true_b)) < 2:
            continue
            
        aucs.append(roc_auc_score(y_true_b, y_prob_b))
        pr_aucs.append(average_precision_score(y_true_b, y_prob_b))
        accs.append(accuracy_score(y_true_b, y_class_b))
        
        cm = confusion_matrix(y_true_b, y_class_b)
        tn, fp, fn, tp = cm.ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
    def get_ci(metric_list):
        return np.percentile(metric_list, 2.5), np.percentile(metric_list, 97.5)
        
    return {
        'auc_ci': get_ci(aucs),
        'pr_auc_ci': get_ci(pr_aucs),
        'acc_ci': get_ci(accs),
        'sens_ci': get_ci(sens),
        'spec_ci': get_ci(specs)
    }

def main():
    metadata_path = 'data/subset_metadata_2000.csv'
    base_dir = 'data/raw'
    
    X, y, _ = load_and_cache_dataset(metadata_path, base_dir)
    
    if len(X) == 0:
        print("No valid signals loaded.")
        return
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store out-of-fold predictions
    oof_probs_cal = np.zeros(len(y))
    oof_preds_class = np.zeros(len(y))
    
    print("\nStarting 5-Fold Cross Validation with Nested Threshold Calibration...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Nested split: 80% sub-train, 20% validation-calibration (strictly out of test fold)
        X_sub_train, X_val_cal, y_sub_train, y_val_cal = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=42
        )
        
        # Setup Data Generators (No class weights: natively 1:1)
        train_gen = ECGDataGenerator(X_sub_train, y_sub_train, batch_size=32, shuffle=True, augment=True)
        val_gen = ECGDataGenerator(X_val_cal, y_val_cal, batch_size=32, shuffle=False, augment=False)
        
        model = build_smaller_resnet_1d()
        early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)
        
        model.fit(
            train_gen,
            epochs=100,
            validation_data=val_gen,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Save fold-specific model to prevent target leakage in downstream multimodal models
        fold_model_path = f"models/binary_1d_ecg_model_fold{fold+1}.h5"
        model.save(fold_model_path)
        print(f"  Saved Fold {fold+1} model to {fold_model_path}")
        
        # Fit Platt Scaler exclusively on nested validation data
        y_val_cal_probs_raw = model.predict(X_val_cal, verbose=0).flatten()
        calibrator = LogisticRegression()
        calibrator.fit(y_val_cal_probs_raw.reshape(-1, 1), y_val_cal)
        y_val_cal_probs = calibrator.predict_proba(y_val_cal_probs_raw.reshape(-1, 1))[:, 1]
        
        # Fit Threshold exclusively on nested validation data
        fpr_v, tpr_v, thresholds_v = roc_curve(y_val_cal, y_val_cal_probs)
        specificity_v = 1 - fpr_v
        
        target_sensitivity = 0.85
        valid_indices = np.where(tpr_v >= target_sensitivity)[0]
        if len(valid_indices) == 0:
            J = tpr_v + specificity_v - 1
            best_idx = np.argmax(J)
            selected_threshold = thresholds_v[best_idx]
        else:
            best_idx = valid_indices[np.argmax(specificity_v[valid_indices])]
            selected_threshold = thresholds_v[best_idx]
            
        print(f"Selected Validation Threshold (Nested): {selected_threshold:.4f}")
        
        # Evaluate on the completely held-out test fold
        y_test_probs_raw = model.predict(X_test, verbose=0).flatten()
        y_test_probs_cal = calibrator.predict_proba(y_test_probs_raw.reshape(-1, 1))[:, 1]
        y_test_preds = (y_test_probs_cal >= selected_threshold).astype(int)
        
        cm_fold = confusion_matrix(y_test, y_test_preds, labels=[0, 1])
        tn, fp, fn, tp = cm_fold.ravel()
        spec_fold = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens_fold = tp / (tp + fn) if (tp + fn) > 0 else 0
        auc_fold = roc_auc_score(y_test, y_test_probs_cal)
        
        print(f"  Fold {fold+1} Test ROC-AUC: {auc_fold:.4f}")
        print(f"  Fold {fold+1} Test Sensitivity: {sens_fold:.4f}")
        print(f"  Fold {fold+1} Test Specificity: {spec_fold:.4f}")
        
        oof_probs_cal[test_idx] = y_test_probs_cal
        oof_preds_class[test_idx] = y_test_preds
        
    print("\n=====================================")
    print("Aggregate Out-Of-Fold (OOF) Results (N=2000):")
    
    auc = roc_auc_score(y, oof_probs_cal)
    pr_auc = average_precision_score(y, oof_probs_cal)
    acc = accuracy_score(y, oof_preds_class)
    
    cm = confusion_matrix(y, oof_preds_class)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Bootstrap CIs
    ci = compute_bootstrap_ci(y, oof_probs_cal, oof_preds_class)
    
    print(f"OOF ROC-AUC: {auc:.4f} (95% CI: {ci['auc_ci'][0]:.4f} - {ci['auc_ci'][1]:.4f})")
    print(f"OOF PR-AUC: {pr_auc:.4f} (95% CI: {ci['pr_auc_ci'][0]:.4f} - {ci['pr_auc_ci'][1]:.4f})")
    print(f"OOF Accuracy: {acc:.4f} (95% CI: {ci['acc_ci'][0]:.4f} - {ci['acc_ci'][1]:.4f})")
    print(f"OOF Sensitivity: {sens:.4f} (95% CI: {ci['sens_ci'][0]:.4f} - {ci['sens_ci'][1]:.4f})")
    print(f"OOF Specificity: {spec:.4f} (95% CI: {ci['spec_ci'][0]:.4f} - {ci['spec_ci'][1]:.4f})")
    print("=====================================")
    
    # Save OOF calibrated probabilities for downstream multimodal models and stress tests
    np.save("models/clean_oof_ecg_probs.npy", oof_probs_cal)
    
    # Save results to models
    os.makedirs('models', exist_ok=True)
    with open('models/1d_model_results.txt', 'w') as f:
        f.write("Option 1A: 1D-CNN on PTB-XL (Single-Source, Confound-Free, OOF Aggregated)\n")
        f.write("=========================================================================\n")
        f.write(f"OOF ROC-AUC: {auc:.4f} (95% CI: {ci['auc_ci'][0]:.4f} - {ci['auc_ci'][1]:.4f})\n")
        f.write(f"OOF PR-AUC: {pr_auc:.4f} (95% CI: {ci['pr_auc_ci'][0]:.4f} - {ci['pr_auc_ci'][1]:.4f})\n")
        f.write(f"OOF Accuracy: {acc:.4f} (95% CI: {ci['acc_ci'][0]:.4f} - {ci['acc_ci'][1]:.4f})\n")
        f.write(f"OOF Sensitivity (Recall of Abnormal): {sens:.4f} (95% CI: {ci['sens_ci'][0]:.4f} - {ci['sens_ci'][1]:.4f})\n")
        f.write(f"OOF Specificity: {spec:.4f} (95% CI: {ci['spec_ci'][0]:.4f} - {ci['spec_ci'][1]:.4f})\n")
        
    print("\nTraining final model on full 2000 records...")
    final_model = build_smaller_resnet_1d()
    final_train_gen = ECGDataGenerator(X, y, batch_size=32, shuffle=True, augment=True)
    
    final_model.fit(
        final_train_gen,
        epochs=40,
        verbose=0
    )
    final_model.save('models/binary_1d_ecg_model.h5')
    print("Final model saved successfully.")
    
if __name__ == '__main__':
    main()

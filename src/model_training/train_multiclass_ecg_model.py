import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def load_and_preprocess_signal(record_path):
    try:
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
            return norm_sig[:1000, :]
        return np.vstack([norm_sig, np.zeros((1000 - norm_sig.shape[0], 12))])
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        return None

def load_multiclass_dataset(metadata_path, base_dir):
    df = pd.read_csv(metadata_path)
    X, y, valid_indices = [], [], []
    print(f"Loading {len(df)} signals...")
    for i, row in df.iterrows():
        file_path = os.path.join(base_dir, row['filename_lr'] + '.dat')
        record_path = file_path.replace('.dat', '')
        sig = load_and_preprocess_signal(record_path)
        if sig is not None:
            X.append(sig)
            y.append([row[f'label_{sc}'] for sc in SUPERCLASSES])
            valid_indices.append(i)
    return np.array(X), np.array(y), valid_indices

class MultiClassECGDataGenerator(tf.keras.utils.Sequence):
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

def build_multiclass_resnet_1d():
    inputs = layers.Input(shape=(1000, 12))
    
    x = layers.Conv1D(32, kernel_size=15, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 1
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
    
    # Block 2
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
    
    # Multi-label classifier
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=5)])
    return model

def main():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    base_dir = 'data/raw'
    
    X, y, valid_indices = load_multiclass_dataset(metadata_path, base_dir)

    if len(X) == 0:
        print("No valid signals loaded.")
        return

    print(f"Loaded {len(X)} records. Shape: {X.shape}")
    print(f"Class distribution: {np.sum(y, axis=0)}")

    # Enforce patient-disjoint folds via groups=patient_id (records are 1-per-patient),
    # stratified on the dominant label to balance prevalence across folds.
    patient_ids = pd.read_csv(metadata_path).iloc[valid_indices]['patient_id'].values
    primary_label = np.argmax(y, axis=1)
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros_like(y, dtype=float)
    
    print("\nStarting 5-Fold Cross Validation (Multi-label)...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, primary_label, groups=patient_ids)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        train_gen = MultiClassECGDataGenerator(X_train, y_train, batch_size=32, shuffle=True, augment=True)
        val_gen = MultiClassECGDataGenerator(X_test, y_test, batch_size=32, shuffle=False, augment=False)
        
        model = build_multiclass_resnet_1d()
        early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)
        
        model.fit(
            train_gen,
            epochs=40,
            validation_data=val_gen,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_test_probs = model.predict(X_test, verbose=0)
        oof_probs[test_idx] = y_test_probs
        
        # Quick AUC printout for the fold
        for c_idx, class_name in enumerate(SUPERCLASSES):
            try:
                auc = roc_auc_score(y_test[:, c_idx], y_test_probs[:, c_idx])
                print(f"  {class_name} Fold AUC: {auc:.4f}")
            except ValueError:
                print(f"  {class_name} Fold AUC: N/A (only one class in test split)")
                
    print("\n=====================================")
    print("Aggregate Out-Of-Fold (OOF) Multi-label Results:")
    os.makedirs('models', exist_ok=True)
    
    with open('models/multiclass_results.txt', 'w') as f:
        f.write("Multi-Label OOF Results (5 Diagnostic Superclasses)\n")
        f.write("===================================================\n")
        for c_idx, class_name in enumerate(SUPERCLASSES):
            try:
                auc = roc_auc_score(y[:, c_idx], oof_probs[:, c_idx])
                pr = average_precision_score(y[:, c_idx], oof_probs[:, c_idx])
                print(f"[{class_name}] OOF ROC-AUC: {auc:.4f} | PR-AUC: {pr:.4f}")
                f.write(f"[{class_name}] OOF ROC-AUC: {auc:.4f} | PR-AUC: {pr:.4f}\n")
            except ValueError:
                pass
                
    np.save("models/clean_oof_multiclass_probs.npy", oof_probs)
    
    print("\nTraining final multiclass model on full dataset...")
    final_model = build_multiclass_resnet_1d()
    final_train_gen = MultiClassECGDataGenerator(X, y, batch_size=32, shuffle=True, augment=True)
    final_model.fit(final_train_gen, epochs=40, verbose=1)
    final_model.save('models/multiclass_1d_ecg_model.h5')
    print("Final Multiclass model saved to models/multiclass_1d_ecg_model.h5")

if __name__ == '__main__':
    main()

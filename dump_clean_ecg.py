import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from multimodal_data_prep import build_metadata_matrix
from train_multimodal_ecg_model import load_signal
from heartbreaker_end_to_end import build_1d_resnet
import warnings
warnings.filterwarnings("ignore")

META_CSV = "dataset_1d/subset_metadata_2000.csv"
SIGNAL_BASE_DIR = "dataset_1d/raw"
N_FOLDS = 5
RANDOM_STATE = 42

df_static, reports, y, pids = build_metadata_matrix(META_CSV, include_text=False)
import pandas as pd
df_meta_csv = pd.read_csv(META_CSV)
X_signals = []
valid_indices = []
for i, row in df_meta_csv.iterrows():
    sig = load_signal(os.path.join(SIGNAL_BASE_DIR, row["filename_lr"]))
    if sig is not None:
        X_signals.append(sig)
        valid_indices.append(i)

X_signals = np.array(X_signals, dtype=np.float32)
y = y[valid_indices]
oof_ecg_probs = np.zeros(len(y))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_signals, y)):
    print(f"Training Fold {fold+1}")
    model = build_1d_resnet()
    model.fit(X_signals[train_idx], y[train_idx], epochs=10, batch_size=32, verbose=0,
              validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    oof_ecg_probs[test_idx] = model.predict(X_signals[test_idx], verbose=0).flatten()

np.save("clean_oof_ecg_probs.npy", oof_ecg_probs)
print("Saved clean_oof_ecg_probs.npy")

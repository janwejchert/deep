import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_signal(path, length=1000):
    try:
        sig, fields = wfdb.rdsamp(path)
        
        # bandpass 0.5-40Hz
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
    except Exception as e:
        return None

def augment(x):
    # x: (1000, 12). NO flips/reversal — they corrupt ECG semantics.
    x = x + tf.random.normal(tf.shape(x), stddev=0.05)            # noise
    x = x * tf.random.uniform([1, 12], 0.85, 1.15)               # amplitude scale
    shift = tf.random.uniform([], -50, 50, tf.int32)            # temporal shift
    x = tf.roll(x, shift, axis=0)
    # random lead dropout
    if tf.random.uniform([]) < 0.3:
        lead = tf.random.uniform([], 0, 12, tf.int32)
        mask = tf.one_hot(lead, 12) == 0
        x = x * tf.cast(mask, x.dtype)
    return x

def build_encoder(input_shape=(1000, 12), dim=128):
    """1D-CNN encoder. Modest 2-block ResNet structure."""
    inp = keras.Input(input_shape)
    
    # Initial Convolution
    x = layers.Conv1D(32, kernel_size=15, padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
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
    
    emb = layers.Dense(dim)(x)
    return keras.Model(inp, emb, name="encoder")

def build_projector(dim=128, proj=64):
    return keras.Sequential([
        layers.Input((dim,)),
        layers.Dense(dim), layers.ReLU(),
        layers.Dense(proj)
    ], name="projector")

def nt_xent(z1, z2, temperature=0.1):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    z = tf.concat([z1, z2], 0)               # (2B, proj)
    sim = tf.matmul(z, z, transpose_b=True) / temperature   # (2B, 2B)
    B = tf.shape(z1)[0]
    # mask self-similarity
    sim = sim - tf.eye(2 * B) * 1e9
    # positives: i <-> i+B
    targets = tf.concat([tf.range(B, 2 * B), tf.range(0, B)], 0)
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(targets, sim, from_logits=True))

def data_generator(pretrain_meta, base_dir, batch_size=64):
    while True:
        # Shuffle for each epoch
        pretrain_meta = pretrain_meta.sample(frac=1).reset_index(drop=True)
        batch_x = []
        for i, row in pretrain_meta.iterrows():
            record_path = os.path.join(base_dir, row['filename_lr'])
            if os.path.exists(record_path + ".dat"):
                sig = load_signal(record_path)
                if sig is not None:
                    batch_x.append(sig)
                    
            if len(batch_x) == batch_size:
                yield tf.convert_to_tensor(np.array(batch_x))
                batch_x = []

def main():
    pretrain_meta_path = 'data/pretrain_metadata.csv'
    if not os.path.exists(pretrain_meta_path):
        print("Run download_ptbxl_pretrain.py first!")
        return
        
    pretrain_meta = pd.read_csv(pretrain_meta_path)
    
    eval_meta = pd.read_csv('data/subset_metadata.csv')
    eval_patients = set(eval_meta['patient_id'].unique())
    
    assert len(set(pretrain_meta.patient_id) & eval_patients) == 0, "LEAKAGE: eval patients present in pretraining pool"
    print("Hard disjointness assertion passed.")
    
    encoder = build_encoder()
    projector = build_projector()
    opt = keras.optimizers.Adam(1e-3)
    
    @tf.function
    def pretrain_step(batch):
        v1 = tf.map_fn(augment, batch)
        v2 = tf.map_fn(augment, batch)
        with tf.GradientTape() as tape:
            z1 = projector(encoder(v1, training=True), training=True)
            z2 = projector(encoder(v2, training=True), training=True)
            loss = nt_xent(z1, z2)
        vars_ = encoder.trainable_variables + projector.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        return loss

    base_dir = 'data/raw'
    batch_size = 64
    epochs = 40
    
    # We may not have all 21000 files downloaded depending on limits, count available files
    available_files = sum(1 for f in pretrain_meta['filename_lr'] if os.path.exists(os.path.join(base_dir, f + ".dat")))
    steps_per_epoch = available_files // batch_size
    print(f"Starting Contrastive SSL Pretraining on {available_files} records ({steps_per_epoch} steps per epoch).")
    
    gen = data_generator(pretrain_meta, base_dir, batch_size=batch_size)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            batch = next(gen)
            loss = pretrain_step(batch)
            epoch_loss += float(loss)
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}/{steps_per_epoch}, Loss: {float(loss):.4f}")
        print(f"--- Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/steps_per_epoch:.4f}")
        
    os.makedirs('models', exist_ok=True)
    encoder.save_weights("models/ecg_ssl_encoder.h5")
    print("Pretraining complete. Saved to models/ecg_ssl_encoder.h5.")

if __name__ == '__main__':
    main()

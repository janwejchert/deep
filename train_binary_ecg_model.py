import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl, axis=-1)
    return focal_loss_fixed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.25)
    args = parser.parse_args()

    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/val'
    
    if not os.path.exists(train_dir):
        print(f"Error: Dataset directory {train_dir} not found.")
        return

    # ============================================================================
    # IMPROVEMENT 1: Resolution upgrade 128 -> 224 (EfficientNetB0 native)
    # ============================================================================
    batch_size = 16  # Halved for memory safety at higher resolution
    img_height = 384
    img_width = 384

    print("Loading training dataset...")
    train_ds = image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',
        class_names=['Normal', 'Abnormal'],
        crop_to_aspect_ratio=True
    )

    print("Loading validation dataset...")
    val_ds = image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',
        class_names=['Normal', 'Abnormal'],
        crop_to_aspect_ratio=True
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    # ============================================================================
    # IMPROVEMENT 2: Domain-safe augmentation via tf.data (not in-model)
    # ============================================================================
    # ECG-safe transforms only: no flips, no rotation beyond trivial
    def augment(image, label):
        # Small random translation via padding + crop
        image = tf.image.resize_with_crop_or_pad(image, img_height + 20, img_width + 20)
        image = tf.image.random_crop(image, size=[tf.shape(image)[0], img_height, img_width, 3])
        # Mild brightness jitter
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds_aug = train_ds_aug.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # We also need the un-augmented train_ds for label counting
    # ============================================================================
    # Dynamic Class Weights (unchanged — proven correct)
    # ============================================================================
    print("\nComputing dynamic class weights...")
    labels = np.concatenate([y for x, y in train_ds.unbatch()], axis=0).flatten()
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    total = counts.sum()
    class_weights = {idx: total / (len(class_names) * counts[idx]) for idx in range(len(class_names))}
    print(f"Class Indices: {dict(enumerate(class_names))}")
    print(f"Counts by index: {dict(enumerate(counts))}")
    print(f"Dynamic Class Weights: {class_weights}\n")

    # ============================================================================
    # Build Model — Phase 1 (frozen base, NO augmentation layers in graph)
    # ============================================================================
    base_model = EfficientNetB0(input_shape=(img_height, img_width, 3),
                                include_top=False,
                                weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    # training=False ensures BN stays in inference mode
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # ============================================================================
    # IMPROVEMENT 3: Focal loss for class imbalance
    # ============================================================================
    METRICS = [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="acc")
    ]

    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=binary_focal_loss(gamma=args.gamma, alpha=args.alpha),
                  metrics=METRICS)

    print("\n=======================================================")
    print("PHASE 1: Training Classification Head (Frozen Base)")
    print("=======================================================")
    
    phase1_epochs = 5
    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weights
    )

    # ============================================================================
    # IMPROVEMENT 3 (cont): Deeper fine-tuning — top 80 layers, BN still frozen
    # ============================================================================
    print("\n=======================================================")
    print("PHASE 2: Fine-Tuning Top 80 Layers (BN Frozen, Micro LR)")
    print("=======================================================")
    
    base_model.trainable = True

    # Freeze everything except the top 80 layers
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    # Explicitly freeze ALL BatchNormalization layers to prevent drift
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    n_train = sum(l.trainable for l in base_model.layers)
    print(f"Phase 2: {n_train} trainable layers in base_model (BatchNorm excluded)\n")

    # Recompile with micro LR and focal loss
    model.compile(optimizer=optimizers.Adam(1e-5),
                  loss=binary_focal_loss(gamma=args.gamma, alpha=args.alpha),
                  metrics=METRICS)

    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/binary_ecg_model.h5'
    
    # Robust Callbacks monitoring val_auc
    checkpoint_cb = callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_auc', mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-7
    )
    
    csv_logger = callbacks.CSVLogger("/Users/felipedeleon/Desktop/Deep Ler,Project/docs/training_log.csv")

    phase2_epochs = 30
    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, csv_logger],
        class_weight=class_weights
    )

    print(f"\nTraining complete. Best model saved to {model_path}")

    with open('/Users/felipedeleon/Desktop/Deep Ler,Project/class_names_binary.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')

if __name__ == '__main__':
    main()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0

def main():
    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel/val'
    
    if not os.path.exists(train_dir):
        print(f"Error: Dataset directory {train_dir} not found.")
        return

    batch_size = 16
    img_height = 224
    img_width = 224

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
        crop_to_aspect_ratio=True,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    # Domain-safe augmentation via tf.data
    def augment(image, label):
        image = tf.image.resize_with_crop_or_pad(image, img_height + 20, img_width + 20)
        image = tf.image.random_crop(image, size=[tf.shape(image)[0], img_height, img_width, 3])
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds_aug = train_ds_aug.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Dynamic Class Weights
    print("\nComputing dynamic class weights...")
    labels = np.concatenate([y for x, y in train_ds.unbatch()], axis=0).flatten()
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    total = counts.sum()
    class_weights = {idx: total / (len(class_names) * counts[idx]) for idx in range(len(class_names))}
    print(f"Class Indices: {dict(enumerate(class_names))}")
    print(f"Counts by index: {dict(enumerate(counts))}")
    print(f"Dynamic Class Weights: {class_weights}\n")

    # Build Model — Phase 1 (frozen base, NO augmentation layers in graph)
    base_model = EfficientNetB0(input_shape=(img_height, img_width, 3),
                                include_top=False,
                                weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)

    METRICS = [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="acc")
    ]

    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
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

    # PHASE 2: Capacity Reduced Fine-Tuning
    print("\n=======================================================")
    print("PHASE 2: Fine-Tuning Top 15 Layers (BN Frozen, Micro LR)")
    print("=======================================================")
    
    base_model.trainable = True

    # Freeze everything except the top 15 layers
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    # Explicitly freeze ALL BatchNormalization layers to prevent drift
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    n_train = sum(l.trainable for l in base_model.layers)
    print(f"Phase 2: {n_train} trainable layers in base_model (BatchNorm excluded)\n")

    model.compile(optimizer=optimizers.Adam(1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  metrics=METRICS)

    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/binary_ecg_model_reduced.h5'
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_auc', mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-7
    )
    
    csv_logger = callbacks.CSVLogger("/Users/felipedeleon/Desktop/Deep Ler,Project/docs/training_log_reduced.csv")

    phase2_epochs = 30
    model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, csv_logger],
        class_weight=class_weights
    )

    print(f"\nTraining complete. Best capacity-reduced model saved to {model_path}")

if __name__ == '__main__':
    main()

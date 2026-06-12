import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import DenseNet121

# Literature-driven custom Focal Loss for class-imbalanced medical image datasets
@tf.keras.utils.register_keras_serializable(package="Custom")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        
        # Alpha balancing factor
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        
        # Modulating factor
        modulating_factor = tf.math.pow(1.0 - p_t, self.gamma)
        
        # Combined loss
        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

def main():
    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split/val'
    
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
        crop_to_aspect_ratio=True
    )

    class_names = train_ds.class_names

    # Advanced Augmentation: Domain-safe + Gaussian Noise + Cutout (Random Erasing)
    def random_erasing(image, probability=0.4, sl=0.02, sh=0.1, r1=0.3):
        if tf.random.uniform([]) > probability:
            return image
        
        img_shape = tf.shape(image)
        img_h = img_shape[0]
        img_w = img_shape[1]
        img_c = img_shape[2]
        area = tf.cast(img_h * img_w, tf.float32)
        
        target_area = tf.random.uniform([], sl, sh) * area
        aspect_ratio = tf.random.uniform([], r1, 1.0/r1)
        
        h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)
        
        h = tf.minimum(h, img_h)
        w = tf.minimum(w, img_w)
        
        x = tf.random.uniform([], 0, img_h - h, dtype=tf.int32)
        y = tf.random.uniform([], 0, img_w - w, dtype=tf.int32)
        
        paddings = [[x, img_h - x - h], [y, img_w - y - w], [0, 0]]
        mask = tf.pad(tf.ones([h, w, img_c]), paddings, constant_values=0.0)
        mask = 1.0 - mask
        return image * mask

    def augment_and_preprocess(image, label):
        # 1. Translation via padding & cropping
        image = tf.image.resize_with_crop_or_pad(image, img_height + 20, img_width + 20)
        image = tf.image.random_crop(image, size=[tf.shape(image)[0], img_height, img_width, 3])
        
        # 2. Brightness jitter
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # 3. Add mild Gaussian Noise (stddev=3.0 in [0,255] range)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=3.0, dtype=tf.float32)
        image = tf.clip_by_value(image + noise, 0.0, 255.0)
        
        # 4. Cutout (Random Erasing)
        image = tf.map_fn(random_erasing, image)
        
        # 5. Preprocess for DenseNet
        image = tf.keras.applications.densenet.preprocess_input(image)
        
        return image, label

    def preprocess_only(image, label):
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_aug = train_ds.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds_aug = train_ds_aug.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    val_ds_preprocessed = val_ds.map(preprocess_only, num_parallel_calls=AUTOTUNE)
    val_ds_preprocessed = val_ds_preprocessed.cache().prefetch(buffer_size=AUTOTUNE)

    # Class Weights (retained for Phase 1 stability)
    labels = np.concatenate([y for x, y in train_ds.unbatch()], axis=0).flatten()
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    total = counts.sum()
    class_weights = {idx: total / (len(class_names) * counts[idx]) for idx in range(len(class_names))}
    print("Dynamic Class Weights:", class_weights)

    # Build DenseNet121 model
    base_model = DenseNet121(input_shape=(img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)

    METRICS = [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="acc")
    ]

    # Use custom Focal Loss instead of BCE
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=FocalLoss(alpha=0.25, gamma=2.0),
                  metrics=METRICS)

    print("\n=======================================================")
    print("PHASE 1: Training Classification Head (Focal Loss)")
    print("=======================================================")
    model.fit(
        train_ds_aug,
        validation_data=val_ds_preprocessed,
        epochs=5,
        class_weight=class_weights
    )

    # Phase 2: Denser Fine-Tuning with Cosine Decay LR Scheduler
    print("\n=======================================================")
    print("PHASE 2: Fine-Tuning DenseNet (BN Frozen, Cosine Decay LR)")
    print("=======================================================")
    base_model.trainable = True

    # Freeze early layers, keep the top layers trainable
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    # Freeze BatchNormalization
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    n_train = sum(l.trainable for l in base_model.layers)
    print(f"Phase 2: {n_train} trainable layers in base_model (BatchNorm excluded)\n")

    # Cosine Decay scheduler step-by-step
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    phase2_epochs = 30
    total_decay_steps = phase2_epochs * steps_per_epoch
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_decay_steps,
        alpha=0.01  # down to 1e-7
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  loss=FocalLoss(alpha=0.25, gamma=2.0),
                  metrics=METRICS)

    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/densenet_ecg_model.h5'
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_auc', mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )
    
    csv_logger = callbacks.CSVLogger("/Users/felipedeleon/Desktop/Deep Ler,Project/docs/training_log_densenet.csv")

    model.fit(
        train_ds_aug,
        validation_data=val_ds_preprocessed,
        epochs=phase2_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, csv_logger],
        class_weight=class_weights
    )

    print(f"\nTraining complete. Best DenseNet model with Focal Loss saved to {model_path}")

if __name__ == '__main__':
    main()

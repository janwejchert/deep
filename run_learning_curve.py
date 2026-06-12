import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import roc_auc_score

def build_model(img_height, img_width):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def evaluate_test_auc(model, test_ds):
    y_true = np.concatenate([y for x, y in test_ds], axis=0).flatten()
    y_pred_probs = model.predict(test_ds, verbose=0).flatten()
    return roc_auc_score(y_true, y_pred_probs)

def main():
    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split/val'
    test_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split/test'
    docs_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/docs'
    
    if not os.path.exists(train_dir):
        print("Dataset directory not found.")
        return

    batch_size = 16
    img_height = 224
    img_width = 224

    # Load base datasets
    train_ds = image_dataset_from_directory(
        train_dir, seed=123, image_size=(img_height, img_width),
        batch_size=batch_size, label_mode='binary', crop_to_aspect_ratio=True
    )
    val_ds = image_dataset_from_directory(
        val_dir, seed=123, image_size=(img_height, img_width),
        batch_size=batch_size, label_mode='binary', crop_to_aspect_ratio=True
    )
    # Test dataset must have shuffle=False
    test_ds = image_dataset_from_directory(
        test_dir, seed=123, image_size=(img_height, img_width),
        batch_size=batch_size, label_mode='binary', crop_to_aspect_ratio=True, shuffle=False
    )

    class_names = train_ds.class_names
    total_batches = tf.data.experimental.cardinality(train_ds).numpy()
    print(f"Total training batches: {total_batches}")

    # Domain-safe augmentation helper
    def augment(image, label):
        image = tf.image.resize_with_crop_or_pad(image, img_height + 20, img_width + 20)
        image = tf.image.random_crop(image, size=[tf.shape(image)[0], img_height, img_width, 3])
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    # Fractions to evaluate
    fractions = [0.25, 0.50, 0.75, 1.0]
    results = {}

    for frac in fractions:
        print(f"\n=======================================================")
        print(f"TRAINING ON {frac*100:.0f}% OF DATA")
        print(f"=======================================================")
        
        # Take fraction of batches
        take_batches = max(1, int(total_batches * frac))
        train_ds_subset = train_ds.take(take_batches)
        
        # Label counting for class weights on this subset
        labels = np.concatenate([y for x, y in train_ds_subset.unbatch()], axis=0).flatten()
        counts = np.bincount(labels.astype(int), minlength=len(class_names))
        total_samples = counts.sum()
        class_weights = {idx: total_samples / (len(class_names) * counts[idx]) for idx in range(len(class_names))}
        print(f"Training subset samples: {total_samples}, class weights: {class_weights}")

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds_subset_aug = train_ds_subset.map(augment, num_parallel_calls=AUTOTUNE)
        train_ds_subset_aug = train_ds_subset_aug.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        
        val_ds_cached = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model, base_model = build_model(img_height, img_width)
        METRICS = [tf.keras.metrics.AUC(name="auc")]

        # Phase 1: Train Head Only (5 epochs)
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=METRICS
        )
        model.fit(
            train_ds_subset_aug,
            validation_data=val_ds_cached,
            epochs=5,
            class_weight=class_weights,
            verbose=0
        )

        # Phase 2: Fine-Tuning Top 15 Layers (BN frozen, 20 epochs for speed/learning curve)
        base_model.trainable = True
        for layer in base_model.layers[:-15]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=METRICS
        )

        early_stopping_cb = callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        )

        model.fit(
            train_ds_subset_aug,
            validation_data=val_ds_cached,
            epochs=20,
            callbacks=[early_stopping_cb],
            class_weight=class_weights,
            verbose=0
        )

        # Evaluate final test AUC
        test_auc = evaluate_test_auc(model, test_ds)
        print(f"Result for {frac*100:.0f}% data -> Test AUC: {test_auc:.4f}")
        results[frac] = test_auc

    # Save results as text
    lc_path = os.path.join(docs_dir, 'learning_curve.txt')
    with open(lc_path, 'w') as f:
        f.write("Data Fraction,Samples,Test AUC\n")
        for frac, auc_val in results.items():
            samples = int(total_batches * 16 * frac)
            f.write(f"{frac:.2f},{samples},{auc_val:.4f}\n")
    print(f"\nSaved learning curve results to {lc_path}")

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    x_axis = [frac * 100 for frac in results.keys()]
    y_axis = list(results.values())
    plt.plot(x_axis, y_axis, marker='o', color='darkblue', linewidth=2)
    plt.title('Data Scaling Diagnostic: Test AUC vs. Training Data %')
    plt.xlabel('Percentage of Training Data used (%)')
    plt.ylabel('Clean Test AUC')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'learning_curve.png'), dpi=150)
    plt.close()
    print("Saved learning curve plot to docs/learning_curve.png")

if __name__ == '__main__':
    main()

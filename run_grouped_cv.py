import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import imagehash
from collections import defaultdict

DATA_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 15

def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model


def load_image_and_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img, label

def augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT + 20, IMG_WIDTH + 20)
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def main():
    # 1. Gather all images and compute clusters
    records = []
    print("Gathering images and computing hashes...")
    for split in ['train', 'val', 'test']:
        for cls in ['Normal', 'Abnormal']:
            d = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                p = os.path.join(d, fn)
                try:
                    with Image.open(p) as img:
                        h = imagehash.phash(img.convert("L"), hash_size=16)
                    label = 0 if cls == 'Normal' else 1
                    records.append({'path': p, 'label': label, 'phash': h})
                except Exception as e:
                    pass
    
    print(f"Total images loaded: {len(records)}")
    
    # Clustering
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["phash"] - records[j]["phash"] <= 6:
                union(i, j)

    groups = np.array([find(i) for i in range(n)])
    paths = np.array([r['path'] for r in records])
    labels = np.array([r['label'] for r in records])
    
    unique_groups = np.unique(groups)
    print(f"Total unique clusters (groups): {len(unique_groups)}")
    
    # 2. 5-Fold Grouped CV
    gkf = GroupKFold(n_splits=5)
    fold_metrics = {'auc': [], 'acc': [], 'sens': [], 'spec': [], 'f1': [], 'prec': []}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(paths, labels, groups)):
        print(f"\n================ FOLD {fold+1} ================")
        
        X_train, y_train = paths[train_idx], labels[train_idx]
        X_val, y_val = paths[val_idx], labels[val_idx]
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        train_ds = train_ds.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Calculate class weights for this fold
        num_normals = np.sum(y_train == 0)
        num_abnormals = np.sum(y_train == 1)
        total = len(y_train)
        weight_for_0 = (1 / num_normals) * (total / 2.0)
        weight_for_1 = (1 / num_abnormals) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Class weights: {class_weight}")
        
        model, base_model = build_model()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        
        print("Phase 1: Training top layers...")
        model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=0
        )
        
        print("Phase 2: Fine-tuning with frozen BN...")
        base_model.trainable = True
        # Keep BN layers frozen
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        early_stopping_ft = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        
        model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds,
            class_weight=class_weight,
            callbacks=[early_stopping_ft],
            verbose=0
        )
        
        # Predict on validation to get deterministic AUC
        print("Evaluating...")
        y_val_pred_probs = []
        for x_b, _ in val_ds:
            y_val_pred_probs.append(model.predict(x_b, verbose=0).flatten())
        y_val_pred_probs = np.concatenate(y_val_pred_probs)
        
        auc_val = roc_auc_score(y_val, y_val_pred_probs)
        
        # Find optimal threshold on TRAIN set (no threshold leakage)
        y_train_pred_probs = []
        # Predict on train without shuffle
        train_eval_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        for x_b, _ in train_eval_ds:
            y_train_pred_probs.append(model.predict(x_b, verbose=0).flatten())
        y_train_pred_probs = np.concatenate(y_train_pred_probs)
        
        fpr_t, tpr_t, thresholds_t = roc_curve(y_train, y_train_pred_probs)
        optimal_idx = np.argmax(tpr_t - fpr_t)
        optimal_threshold = thresholds_t[optimal_idx]
        print(f"Optimal threshold found on train: {optimal_threshold:.4f}")
        
        y_val_pred_classes = (y_val_pred_probs >= optimal_threshold).astype(int)
        
        acc_val = accuracy_score(y_val, y_val_pred_classes)
        f1_val = f1_score(y_val, y_val_pred_classes)
        prec_val = precision_score(y_val, y_val_pred_classes)
        sens_val = recall_score(y_val, y_val_pred_classes) # Sensitivity is recall for class 1
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_classes).ravel()
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Fold {fold+1} Metrics -> AUC: {auc_val:.4f}, Acc: {acc_val:.4f}, Sens: {sens_val:.4f}, Spec: {spec_val:.4f}, F1: {f1_val:.4f}")
        
        fold_metrics['auc'].append(auc_val)
        fold_metrics['acc'].append(acc_val)
        fold_metrics['sens'].append(sens_val)
        fold_metrics['spec'].append(spec_val)
        fold_metrics['f1'].append(f1_val)
        fold_metrics['prec'].append(prec_val)

    print("\n================ FINAL CV RESULTS ================")
    mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}
    
    print(f"Mean AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"Mean Acc: {mean_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
    print(f"Mean Sens: {mean_metrics['sens']:.4f} ± {std_metrics['sens']:.4f}")
    print(f"Mean Spec: {mean_metrics['spec']:.4f} ± {std_metrics['spec']:.4f}")
    print(f"Mean F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

    with open('docs/grouped_cv_results.txt', 'w') as f:
        f.write("5-Fold Grouped CV Results on dataset_skel\n")
        f.write(f"Mean AUC ± Std: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}\n")
        f.write(f"Mean Accuracy ± Std: {mean_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}\n")
        f.write(f"Mean Sensitivity ± Std: {mean_metrics['sens']:.4f} ± {std_metrics['sens']:.4f}\n")
        f.write(f"Mean Specificity ± Std: {mean_metrics['spec']:.4f} ± {std_metrics['spec']:.4f}\n")
        f.write(f"Mean F1-score ± Std: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n")

if __name__ == '__main__':
    main()

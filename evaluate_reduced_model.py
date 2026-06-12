import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
from scipy import stats

def evaluate_split(model, ds, split_name):
    print(f"\nEvaluating on {split_name} dataset...")
    y_true = np.concatenate([y for x, y in ds], axis=0).flatten()
    y_pred_probs = model.predict(ds).flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"{split_name} ROC AUC Score: {roc_auc:.4f}")
    return y_true, y_pred_probs, fpr, tpr, thresholds, roc_auc

def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lo, hi

def find_threshold_for_sensitivity(y_true, y_prob, target_sens):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = tpr >= target_sens
    if not np.any(valid):
        return thresholds[-1], tpr[-1], 1 - fpr[-1]
    idx = np.where(valid)[0]
    best_idx = idx[0]
    thresh = thresholds[best_idx]
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]
    return thresh, sens, spec

def mcnemar_test(y_true, y_pred_model, y_pred_baseline):
    model_correct = (y_pred_model == y_true)
    baseline_correct = (y_pred_baseline == y_true)
    b = np.sum(model_correct & ~baseline_correct)
    c = np.sum(~model_correct & baseline_correct)
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return p_value

def tta_predict(model, ds, img_height, img_width):
    augmentations = [
        lambda x: x,
        lambda x: tf.image.adjust_brightness(x, 0.05),
        lambda x: tf.image.adjust_brightness(x, -0.05),
        lambda x: tf.roll(x, shift=int(img_width * 0.02), axis=2),
        lambda x: tf.roll(x, shift=-int(img_width * 0.02), axis=2),
    ]
    all_preds = []
    for aug_fn in augmentations:
        preds = []
        for images, _ in ds:
            aug_images = aug_fn(images)
            pred = model.predict(aug_images, verbose=0)
            preds.append(pred.flatten())
        all_preds.append(np.concatenate(preds))
    return np.mean(all_preds, axis=0)

def main():
    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel/val'
    test_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_skel/test'
    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/binary_ecg_model_reduced.h5'
    docs_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/docs'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    batch_size = 16
    img_height = 224
    img_width = 224

    kwargs = dict(
        seed=123, image_size=(img_height, img_width), batch_size=batch_size,
        label_mode='binary', class_names=['Normal', 'Abnormal'],
        crop_to_aspect_ratio=True, shuffle=False
    )

    train_ds = image_dataset_from_directory(train_dir, **kwargs)
    val_ds = image_dataset_from_directory(val_dir, **kwargs)
    test_ds = image_dataset_from_directory(test_dir, **kwargs)
    class_names = test_ds.class_names

    print("Loading reduced model...")
    model = tf.keras.models.load_model(model_path)

    train_y, train_prob, _, _, _, train_auc = evaluate_split(model, train_ds, "Train")
    val_y, val_prob, val_fpr, val_tpr, val_thresholds, val_auc = evaluate_split(model, val_ds, "Validation")
    
    J = val_tpr - val_fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = val_thresholds[optimal_idx]
    print(f"\nYouden's J Optimal Threshold: {optimal_threshold:.4f}")

    thresh_90, sens_90, spec_90 = find_threshold_for_sensitivity(val_y, val_prob, 0.90)
    thresh_95, sens_95, spec_95 = find_threshold_for_sensitivity(val_y, val_prob, 0.95)

    test_y, test_prob, test_fpr, test_tpr, test_thresholds, test_auc = evaluate_split(model, test_ds, "Test")
    auc_lo, auc_hi = bootstrap_auc_ci(test_y, test_prob)
    print(f"Test AUC 95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]")

    print("\nRunning TTA...")
    tta_prob = tta_predict(model, test_ds, img_height, img_width)
    tta_auc = roc_auc_score(test_y, tta_prob)
    tta_lo, tta_hi = bootstrap_auc_ci(test_y, tta_prob)
    print(f"TTA Test AUC: {tta_auc:.4f} [95% CI: {tta_lo:.4f}, {tta_hi:.4f}]")

    y_pred_classes = (test_prob >= optimal_threshold).astype(int)
    report = classification_report(test_y, y_pred_classes, target_names=class_names, zero_division=0)
    print(report)

    cm = confusion_matrix(test_y, y_pred_classes)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    baseline_acc = np.sum(test_y) / len(test_y)
    test_acc = (tn + tp) / len(test_y)

    trivial_baseline = np.ones_like(test_y).astype(int)
    mcnemar_p = mcnemar_test(test_y.astype(int), y_pred_classes, trivial_baseline)

    summary = {
        "train_auc": round(train_auc, 4),
        "val_auc": round(val_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_auc_ci_lo": round(auc_lo, 4),
        "test_auc_ci_hi": round(auc_hi, 4),
        "tta_test_auc": round(tta_auc, 4),
        "tta_test_auc_ci_lo": round(tta_lo, 4),
        "tta_test_auc_ci_hi": round(tta_hi, 4),
        "mcnemar_p_value": round(mcnemar_p, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_accuracy": round(test_acc, 4),
        "trivial_baseline_accuracy": round(baseline_acc, 4)
    }
    
    summary_path = os.path.join(docs_dir, 'summary_reduced.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Test Confusion Matrix (Reduced Model, Youden Threshold={optimal_threshold:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'confusion_matrix_reduced.png'), dpi=150)
    plt.close()

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(test_fpr, test_tpr, color='darkorange', lw=2,
             label=f'Test ROC (AUC = {test_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC - Reduced Capacity Model')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(docs_dir, 'roc_curve_reduced.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    main()

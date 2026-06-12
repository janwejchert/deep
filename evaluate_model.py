import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array
from scipy import stats

# ============================================================================
# Helpers
# ============================================================================

def evaluate_split(model, ds, split_name):
    """Deterministic evaluation on a single split."""
    print(f"\nEvaluating on {split_name} dataset...")
    y_true = np.concatenate([y for x, y in ds], axis=0).flatten()
    y_pred_probs = model.predict(ds).flatten()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"{split_name} ROC AUC Score: {roc_auc:.4f}")
    
    return y_true, y_pred_probs, fpr, tpr, thresholds, roc_auc


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, alpha=0.05, seed=42):
    """Bootstrap 95% confidence interval for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate samples
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lo, hi


def find_threshold_for_sensitivity(y_true, y_prob, target_sens):
    """Find the threshold that achieves at least target_sens sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # tpr is sensitivity; find smallest threshold where tpr >= target_sens
    valid = tpr >= target_sens
    if not np.any(valid):
        return thresholds[-1], tpr[-1], 1 - fpr[-1]
    # Among valid points, pick the one with highest specificity (lowest fpr)
    idx = np.where(valid)[0]
    # The last valid index gives highest specificity at that sensitivity
    best_idx = idx[0]  # first index where tpr crosses target (highest threshold)
    thresh = thresholds[best_idx]
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]
    return thresh, sens, spec


def mcnemar_test(y_true, y_pred_model, y_pred_baseline):
    """McNemar test comparing two classifiers."""
    # b = model correct, baseline wrong
    # c = model wrong, baseline correct
    model_correct = (y_pred_model == y_true)
    baseline_correct = (y_pred_baseline == y_true)
    b = np.sum(model_correct & ~baseline_correct)
    c = np.sum(~model_correct & baseline_correct)
    # McNemar statistic with continuity correction
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return p_value


def tta_predict(model, ds, img_height, img_width, n_augmentations=4):
    """Test-time augmentation: average predictions over augmented copies."""
    # Domain-safe TTA transforms
    augmentations = [
        lambda x: x,  # original
        lambda x: tf.image.adjust_brightness(x, 0.05),
        lambda x: tf.image.adjust_brightness(x, -0.05),
        lambda x: tf.roll(x, shift=int(img_width * 0.02), axis=2),   # small horizontal shift
        lambda x: tf.roll(x, shift=-int(img_width * 0.02), axis=2),  # small horizontal shift opposite
    ]
    
    all_preds = []
    for aug_fn in augmentations:
        preds = []
        for images, _ in ds:
            aug_images = aug_fn(images)
            pred = model.predict(aug_images, verbose=0)
            preds.append(pred.flatten())
        all_preds.append(np.concatenate(preds))
    
    # Average across augmentations
    return np.mean(all_preds, axis=0)


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    train_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/train'
    val_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/val'
    test_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/test'
    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/binary_ecg_model.h5'
    docs_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/docs'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Match training resolution exactly
    batch_size = 16
    img_height = 384
    img_width = 384

    # CRITICAL: shuffle=False to prevent misalignment
    kwargs = dict(
        seed=123, image_size=(img_height, img_width), batch_size=batch_size,
        label_mode='binary', class_names=['Normal', 'Abnormal'],
        crop_to_aspect_ratio=True, shuffle=False
    )

    print("Loading datasets with shuffle=False...")
    train_ds = image_dataset_from_directory(train_dir, **kwargs)
    val_ds = image_dataset_from_directory(val_dir, **kwargs)
    test_ds = image_dataset_from_directory(test_dir, **kwargs)

    class_names = test_ds.class_names
    print("Class names:", class_names)

    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # ========================================================================
    # 1. Training convergence check
    # ========================================================================
    train_y, train_prob, _, _, _, train_auc = evaluate_split(model, train_ds, "Train")

    # ========================================================================
    # 2. Validation — compute threshold (prevents test-set leakage)
    # ========================================================================
    val_y, val_prob, val_fpr, val_tpr, val_thresholds, val_auc = evaluate_split(model, val_ds, "Validation")
    
    # Youden's J
    J = val_tpr - val_fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = val_thresholds[optimal_idx]
    print(f"\nYouden's J Optimal Threshold (from Validation): {optimal_threshold:.4f}")

    # Operating points from validation
    thresh_90, sens_90, spec_90 = find_threshold_for_sensitivity(val_y, val_prob, 0.90)
    thresh_95, sens_95, spec_95 = find_threshold_for_sensitivity(val_y, val_prob, 0.95)
    print(f"Screening threshold (sens>=0.90): {thresh_90:.4f}")
    print(f"High-recall threshold (sens>=0.95): {thresh_95:.4f}")

    # ========================================================================
    # 3. Test evaluation — standard
    # ========================================================================
    test_y, test_prob, test_fpr, test_tpr, test_thresholds, test_auc = evaluate_split(model, test_ds, "Test")
    
    # Bootstrap CI for test AUC
    auc_lo, auc_hi = bootstrap_auc_ci(test_y, test_prob)
    print(f"Test AUC 95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]")

    # ========================================================================
    # 4. Test-Time Augmentation (TTA)
    # ========================================================================
    print("\nRunning Test-Time Augmentation (5 variants)...")
    tta_prob = tta_predict(model, test_ds, img_height, img_width)
    tta_auc = roc_auc_score(test_y, tta_prob)
    tta_lo, tta_hi = bootstrap_auc_ci(test_y, tta_prob)
    print(f"TTA Test AUC: {tta_auc:.4f} [95% CI: {tta_lo:.4f}, {tta_hi:.4f}]")

    # ========================================================================
    # 5. Classification report at Youden threshold
    # ========================================================================
    y_pred_classes = (test_prob >= optimal_threshold).astype(int)

    print("\n================ CLASSIFICATION REPORT (TEST SET) ================")
    report = classification_report(test_y, y_pred_classes, target_names=class_names, zero_division=0)
    print(report)

    cm = confusion_matrix(test_y, y_pred_classes)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    baseline_acc = np.sum(test_y) / len(test_y)
    test_acc = (tn + tp) / len(test_y)

    # ========================================================================
    # 6. Operating-point table on TEST set (using val-derived thresholds)
    # ========================================================================
    print("\n================ OPERATING-POINT TABLE (TEST SET) ================")
    
    def apply_threshold_on_test(threshold):
        preds = (test_prob >= threshold).astype(int)
        cm_t = confusion_matrix(test_y, preds)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        ppv_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        npv_t = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
        return sens_t, spec_t, ppv_t, npv_t
    
    youden_sens, youden_spec, youden_ppv, youden_npv = apply_threshold_on_test(optimal_threshold)
    screen_sens, screen_spec, screen_ppv, screen_npv = apply_threshold_on_test(thresh_90)
    high_sens, high_spec, high_ppv, high_npv = apply_threshold_on_test(thresh_95)

    print(f"{'Operating Point':<25} {'Threshold':<12} {'Sensitivity':<13} {'Specificity':<13} {'PPV':<8} {'NPV':<8}")
    print("-" * 79)
    print(f"{'Youden J':<25} {optimal_threshold:<12.4f} {youden_sens:<13.4f} {youden_spec:<13.4f} {youden_ppv:<8.4f} {youden_npv:<8.4f}")
    print(f"{'Screening (sens>=0.90)':<25} {thresh_90:<12.4f} {screen_sens:<13.4f} {screen_spec:<13.4f} {screen_ppv:<8.4f} {screen_npv:<8.4f}")
    print(f"{'High-recall (sens>=0.95)':<25} {thresh_95:<12.4f} {high_sens:<13.4f} {high_spec:<13.4f} {high_ppv:<8.4f} {high_npv:<8.4f}")

    # ========================================================================
    # 7. McNemar test vs trivial baseline
    # ========================================================================
    trivial_baseline = np.ones_like(test_y).astype(int)  # predict all Abnormal
    mcnemar_p = mcnemar_test(test_y.astype(int), y_pred_classes, trivial_baseline)
    print(f"\nMcNemar test (model vs. all-Abnormal baseline): p = {mcnemar_p:.4f}")

    # ========================================================================
    # 8. Detailed summary
    # ========================================================================
    print(f"\n--- Detailed Metrics (Test Set) ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Trivial Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Specificity (Normal Recall): {specificity:.4f}")
    print(f"Sensitivity (Abnormal Recall): {sensitivity:.4f}")
    print(f"Exact Counts - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # ========================================================================
    # 9. Save summary.json
    # ========================================================================
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
        "operating_points": {
            "youden": {
                "threshold": round(optimal_threshold, 4),
                "sensitivity": round(youden_sens, 4),
                "specificity": round(youden_spec, 4),
                "ppv": round(youden_ppv, 4),
                "npv": round(youden_npv, 4)
            },
            "screening_sens_90": {
                "threshold": round(thresh_90, 4),
                "sensitivity": round(screen_sens, 4),
                "specificity": round(screen_spec, 4),
                "ppv": round(screen_ppv, 4),
                "npv": round(screen_npv, 4)
            },
            "high_recall_sens_95": {
                "threshold": round(thresh_95, 4),
                "sensitivity": round(high_sens, 4),
                "specificity": round(high_spec, 4),
                "ppv": round(high_ppv, 4),
                "npv": round(high_npv, 4)
            }
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_accuracy": round(test_acc, 4),
        "trivial_baseline_accuracy": round(baseline_acc, 4)
    }
    
    summary_path = os.path.join(docs_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # ========================================================================
    # 10. Save text report
    # ========================================================================
    with open(os.path.join(docs_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("--- ROC AUC SCORES ---\n")
        f.write(f"Train AUC: {train_auc:.4f}\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f} [95% CI: {auc_lo:.4f}, {auc_hi:.4f}]\n")
        f.write(f"TTA Test AUC: {tta_auc:.4f} [95% CI: {tta_lo:.4f}, {tta_hi:.4f}]\n\n")
        f.write(f"McNemar p-value (vs. all-Abnormal baseline): {mcnemar_p:.4f}\n\n")
        f.write(f"Optimal Threshold (Youden, from Val): {optimal_threshold:.4f}\n\n")
        f.write("--- OPERATING-POINT TABLE (TEST SET) ---\n")
        f.write(f"Youden:          thresh={optimal_threshold:.4f}  sens={youden_sens:.4f}  spec={youden_spec:.4f}  PPV={youden_ppv:.4f}  NPV={youden_npv:.4f}\n")
        f.write(f"Screening>=0.90: thresh={thresh_90:.4f}  sens={screen_sens:.4f}  spec={screen_spec:.4f}  PPV={screen_ppv:.4f}  NPV={screen_npv:.4f}\n")
        f.write(f"High-recall>=0.95: thresh={thresh_95:.4f}  sens={high_sens:.4f}  spec={high_spec:.4f}  PPV={high_ppv:.4f}  NPV={high_npv:.4f}\n\n")
        f.write("--- TEST SET METRICS ---\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Trivial Baseline Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Confusion Matrix Counts -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nNote: Evaluated deterministically (shuffle=False) preventing misalignment.")

    # ========================================================================
    # 11. Plots
    # ========================================================================
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Test Confusion Matrix (Youden Threshold={optimal_threshold:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(test_fpr, test_tpr, color='darkorange', lw=2,
             label=f'Test ROC (AUC = {test_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.50)')
    # Mark operating points
    youden_fpr_pt = 1 - youden_spec
    plt.scatter([youden_fpr_pt], [youden_sens], color='green', s=100, zorder=5,
                label=f'Youden (thresh={optimal_threshold:.3f})')
    screen_fpr_pt = 1 - screen_spec
    plt.scatter([screen_fpr_pt], [screen_sens], color='red', s=100, zorder=5, marker='^',
                label=f'Screening sens>=0.90 (thresh={thresh_90:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Receiver Operating Characteristic')
    plt.legend(loc="lower right", fontsize=9)
    plt.savefig(os.path.join(docs_dir, 'roc_curve.png'), dpi=150)
    plt.close()

    print("\nEvaluation complete! Graphs and summary.json saved to docs/ directory.")

if __name__ == '__main__':
    main()

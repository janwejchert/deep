# ECG Classification Project Audit Report

## 1. Executive Summary
This project aimed to build a binary classification system (Normal vs. Abnormal) for electrocardiograms (ECGs). It began as a computer vision project processing 2D ECG plots (images) using advanced transfer learning architectures (EfficientNet, DenseNet). Although these 2D models achieved exceptionally high ROC-AUC scores (0.97–0.98), a rigorous methodological audit revealed that the performance was a statistical illusion caused by a **perfect source-label confound**: all "Normal" images were sourced from the Latidos dataset, while all "Abnormal" images were sourced from the PTB-XL dataset. The deep neural networks learned to distinguish the visual rendering artifacts (grid lines, font differences, baseline thicknesses) between the two sources rather than true cardiovascular physiology.

To resolve this confound, the project executed a structural pivot. The 2D image pipeline was archived, and the methodology was rebuilt from the ground up as a **1D raw-signal pipeline** using only the PTB-XL database. By enforcing patient-disjoint 5-fold cross-validation on a balanced pilot subset of 200 patients, the source confound was eliminated. 

The final 1D model (ResNet + Binary Focal Loss + Platt Scaling) achieved a cross-validated ROC-AUC of **0.8140**. While not clinically ready due to small-sample specificity instability, it serves as a methodologically sound Minimum Viable Product (MVP) and a clean foundation for future scaling.

## 2. Active Pipeline (1D Raw Signal)
The following scripts define the active, methodologically valid pipeline. **Future development should only use these scripts.**

| Script | Purpose | Status |
|---|---|---|
| `rebuild_1d_dataset.py` | Extracts raw `.dat` signals from the PTB-XL `records100` directory, scales them, applies bandpass filtering, enforces patient-level deduplication, and splits 200 records equally. | **Active** (Data Generation) |
| `train_1d_ecg_model.py` | Trains the 1D plain CNN baseline model. | **Active** (Baseline Training) |
| `ablation_experiments.py` | The definitive training script. Implements the 1D ResNet, `ECGDataGenerator` (augmentation), Binary Focal Loss, and 5-fold cross-validation. Calibrates via Platt Scaling and outputs per-fold metrics. | **Active** (Primary Model Training) |
| `get_1d_metrics.py` | Standalone inference script that loads the final `binary_1d_ecg_model.h5`, evaluates test data, and reports AUC and threshold metrics. | **Active** (Inference) |

## 3. Deprecated Pipeline (2D Image Confound)
The following scripts, notebooks, and models relate to the 2D image classification approach. They are retained for historical documentation and methodological reference but **must not be used to train clinical classifiers** because they suffer from the Latidos/PTB-XL source confound.

| Category | Files | Reason for Deprecation |
|---|---|---|
| **2D Model Training** | `train_binary_ecg_model.py`, `train_densenet_ecg.py`, `train_ecg_model.py`, `train_binary_ecg_model_reduced.py` | Trained on confounded image datasets. |
| **Image Preprocessing** | `preprocess_images.py`, `standardize_images_v2.py`, `std_images.py`, `split_data.py`, `rebuild_split.py` | Focuses on visual artifacts (grids, crops) which do not solve the underlying source confound. |
| **2D Evaluation** | `evaluate_model.py`, `evaluate_reduced_model.py`, `evaluate_densenet_model.py`, `gradcam_fixed.py` | Evaluates performance on confounded labels; Grad-CAM focuses on non-physiological spatial artifacts. |
| **Probes & Investigations** | `source_bias_probe.py`, `source_bias_probe_with_gradcam.py`, `app_2d_investigation.py` | Scripts used to uncover the confound. Their job is done. |
| **2D Applications** | `app.py` | Serving a confounded model to end-users is unsafe. |
| **Stored Models** | `binary_ecg_model.h5`, `densenet_ecg_model.h5`, `ecg_model.h5`, `binary_ecg_model_reduced.h5` | Confounded weights. |

## 4. Current State & Definitive Metrics

The active 1D model is defined in `train_1d_ecg_model.py`.
- **Architecture:** Smaller 1D ResNet (2 residual blocks, tighter capacity to prevent overfitting)
- **Augmentation:** On-the-fly Gaussian noise ($\sigma=0.01$), amplitude scaling ($\pm10\%$), temporal shifting ($\pm20$).
- **Loss:** Balanced Binary Focal Loss ($\gamma=2.0$, $\alpha=0.5$), no artificial class weights.
- **Calibration & Thresholding:** Nested validation Platt Scaling with out-of-fold (OOF) aggregation.

**Aggregate Out-Of-Fold (OOF) Performance (N=200, 95% Bootstrap CI):**
- **OOF ROC-AUC:** `0.7638` (95% CI: `0.6972 - 0.8297`)
- **OOF PR-AUC:** `0.7821` (95% CI: `0.7037 - 0.8508`)
- **OOF Accuracy:** `0.7100` (95% CI: `0.6500 - 0.7701`)
- **OOF Sensitivity:** `0.8300` (95% CI: `0.7500 - 0.9020`)
- **OOF Specificity:** `0.5900` (95% CI: `0.4951 - 0.6875`)

> [!NOTE]
> By eliminating artificial positive-class biases and computing metrics globally across all out-of-fold predictions rather than averaging noisy per-fold specificities, the model achieves a vastly more stable and defensible specificity. The ROC-AUC is statistically sound, though the pilot size limits precision.

## 5. Official Documentation Map
The project is documented through three primary markdown files:

1. **`methodology_guide.md`**
   - *Purpose:* Deep mathematical and theoretical explanation of the pipeline, from why 2D CNNs fail due to source-confounds, to the architecture of the 1D ResNet.
   - *Status:* Fully updated to reflect the 1D ResNet implementation and metrics.
2. **`validation_report.md`**
   - *Purpose:* Detailed metrics reporting for the 1D ResNet (ROC curves, PR curves, threshold sweeps, fold-by-fold breakdowns).
   - *Status:* Fully updated and reflects the honest limitations of the 200-patient pilot.
3. **`final_ecg_report.md`**
   - *Purpose:* The overarching project narrative, focusing on the audit of the 2D models and the subsequent pivot to 1D.
   - *Status:* Fully updated. Contains the definitive "Perfect Methodology Checklist."

## 6. Next Steps Roadmap
The 1D pipeline has successfully proven pilot feasibility without leakage or source confounds. The next steps for the incoming engineering or research team are:

1. **Scale Data:** Expand from 200 records to 1,000+ records from PTB-XL using `StratifiedGroupKFold` on `patient_id`.
2. **Component-wise Ablation:** Test the bundled upgrades (ResNet, augmentation, focal loss, Platt scaling) individually to isolate which component caused the performance improvement.
3. **Calibration Validation:** Independently validate Platt scaling quality using Brier scores and reliability curves.
4. **External Validation:** Test the frozen model on completely unseen external datasets (e.g., Chapman-Shaoxing or Georgia 12-lead ECG) to confirm true physiological generalization.

# ECG Classification & Multimodal Fusion MVP

Deep Learning Final Project · IE University

This repository contains the codebase for an automated ECG prescreening pipeline. The project is built entirely on raw 1D physiological signals and clinical metadata, avoiding visual shortcut confounds.

The repository maintains two distinct, complementary pipelines:

1. **The 1D Physiological Pipeline (ECG-Only):** An automated pipeline operating on raw 12-lead signal waveforms from a single clinical source (PTB-XL). Utilizing a 1D ResNet, Binary Focal Loss, and Out-of-Fold (OOF) Platt Scaling, this pipeline achieves an honest, confound-free cross-validated ROC-AUC of **0.9243**.
2. **The Multimodal Fusion Pipeline (Heartbreaker):** An advanced late-fusion architecture that leverages the frozen 1D ResNet features combined with patient demographic features (age, sex, BMI) to maximize classification sensitivity and specificity. The primary multimodal model achieves an OOF ROC-AUC of **0.9238** (Tier 1 LR) and **0.9218** (Tier 2 MLP).

---

## 📊 Dataset & Descriptive Analytics

The pipeline is evaluated on a standardized subset of **2,000 unique patient records** from the PTB-XL database, balanced exactly for binary classification:

* **Total Records:** 2,000
* **Class Balance:** 1,000 Normal (50.0%) | 1,000 Abnormal (50.0%)
  * *Abnormal includes Myocardial Infarction (MI), ST/T-change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).*

### Demographic Feature Distribution

* **Age Profile:** Mean = 61.5 years, Median = 61.0 years (range: 5.0 to 95.0 years, values clipped at 120.0).
* **Sex Profile:** 992 Male (49.6%) | 1,008 Female (50.4%).
* **Missingness & BMI:**
  * Height is missing in 63.8% of records (1,276 records).
  * Weight is missing in 50.1% of records (1,002 records).
  * BMI (mean = 25.4, median = 25.1) is derived dynamically when both height and weight are present.
  * *To prevent target leakage via missingness bias (where missing values correlate with clinical acquisition settings), binary missingness flags (`height_missing`, `weight_missing`, `bmi_missing`) are explicitly fed to the models as features.*

### Signal Quality Metrics

* **Baseline Drift:** Present in 7.6% of records.
* **Static Noise:** Present in 13.1% of records.
* **Burst Noise:** Present in 2.1% of records.
* **Electrode Problems:** Present in 0.1% of records.
* **Human Validation:** 77.5% of the waveforms are validated by a cardiologist.
  * *All signal quality and human validation flags are excluded from the primary multimodal model during safety ablations to prevent workflow-proxy leakage.*

---

## ⚙️ Signal Processing & Feature Engineering

To guarantee clean physiological representations, raw 12-lead signals undergo a rigorous signal processing pipeline before being modeled:

```
  Raw 12-Lead ECG (10s @ 100Hz)
            │
            ▼
   Bandpass Filter (0.5–40 Hz)   ───► Removes baseline wander and powerline interference
            │
            ▼
    Lead-Wise Z-Normalization    ───► Normalizes amplitude variances across patients
            │
            ▼
     Truncate or Pad (10s)       ───► Standardizes input to exactly 1,000 samples per lead
```

### Tabular Feature Preprocessing

1. **Imputation:** Missing continuous metrics (`age`, `height`, `weight`, `bmi`) are imputed fold-safely inside the cross-validation loop using training-fold medians.
2. **Standardization:** All continuous columns are scaled using a standard scaler fitted exclusively on the training fold.
3. **One-Hot Encoding:** Categorical variables such as `heart_axis` are converted to 9 binary categories.

---

## 🛡️ Leakage-Free Validation & Splitting

To prevent data leakage and metric inflation:

1. **Patient-Disjoint Splitting:** The dataset is split using a **5-Fold Stratified Group K-Fold** grouped by `patient_id`. This guarantees that recordings from the same patient never overlap between the training, validation, and test splits (0% patient overlap).
2. **Nested Platt Scaling:** Platt calibration is fitted exclusively on a separate validation-calibration slice inside each fold.
3. **Sensitivity-Constrained Thresholding:** The classification threshold is selected dynamically on the calibration slice to satisfy a clinical sensitivity floor of **$\ge 0.85$**, ensuring abnormal cases are not missed.

---

## 🫀 Model Architectures

The repository implements two tiers of model architectures:

### 1. ECG-Only 1D ResNet

- A 1D convolutional residual network operating directly on the processed waveforms.
* Contains two residual blocks utilizing Conv1D, Batch Normalization, ReLU, and MaxPooling.
* Trained with **Binary Focal Loss** ($\gamma=2.0$, $\alpha=0.5$) to counter gradient saturation.
* Feature representation: extracts a 128-dimensional embedding from the `GlobalAveragePooling1D` layer.

### 2. Heartbreaker Multimodal late-fusion

- **Tier 1 (Probability Fusion):** A calibrated Logistic Regression model combining the out-of-fold probability output of the 1D ResNet with the processed demographics.
* **Tier 2 (Embedding Fusion):** A Multi-Layer Perceptron (MLP) combining the 128-dimensional frozen ECG embedding with a dense metadata embedding branch, followed by dropout layers.

---

## 📊 Key Performance Metrics (5-Fold Stratified Cross-Validation)

All metrics are aggregated out-of-fold using Platt-calibrated predictions:

| Configuration | ROC-AUC | Sensitivity (Recall) | Specificity | Key Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **1D ResNet (ECG-Only)** | `0.9243 [95% CI: 0.9131–0.9350]` | `0.8580` | `0.8410` | Honest signal-only clinical baseline. |
| **Primary Multimodal (ECG + Demographics)** | **`0.9238 [95% CI: 0.9114–0.9348]`** | **`0.8660`** | **`0.8090`** | Highly robust, leakage-safe probability fusion (Tier 1 LR). |
| **Secondary Multimodal (MLP Fusion)** | `0.9223 [95% CI: 0.9103–0.9341]` | `0.8560` | `0.8340` | Embedding-level fusion MLP (Tier 2 MLP). |
| **Exploratory Multimodal (+ Report Text)** | `0.9565 [95% CI: 0.9482–0.9650]` | `0.8500` | `0.9320` | Structured + text (TF-IDF) (M7 report ablation ladder baseline). |

---

## 📈 Performance Visualizations

### 🫀 1D ResNet ECG-Only Model

| ROC Curve | Confusion Matrix |
| :---: | :---: |
| ![1D ECG ROC Curve](reports/figures/fig5_roc_curve.png) | ![1D ECG Confusion Matrix](reports/figures/fig3_confusion_matrix.png) |

### ⚡ Heartbreaker Multimodal Fusion Model

| Multimodal ROC Curve | Multimodal Confusion Matrix |
| :---: | :---: |
| ![Heartbreaker ROC Curve](reports/figures/hb_meta_fig3_roc_curve.png) | ![Heartbreaker Confusion Matrix](reports/figures/hb_meta_fig2_confusion_matrix.png) |

### 🔍 Robustness & Ablation Analysis

| Ablation Performance Ladder | Permutation Feature Importance |
| :---: | :---: |
| ![Ablation Ladder](reports/figures/ablation_ladder_chart.png) | ![Permutation Importance](reports/figures/permutation_test_chart.png) |

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download raw waveforms

Download the balanced 2,000 patient subset of PTB-XL raw physiological signals:

```bash
python src/data_processing/download_ptbxl_2000.py
```

### 3. Run the 1D Physiological Model (ECG-Only)

Train the 1D ResNet classifier on raw 10-second PTB-XL signal waveforms:

```bash
python src/model_training/train_1d_ecg_model.py
```

Launch the interactive Streamlit triage application for raw ECG signals:

```bash
streamlit run src/streamlit_dashboard/app.py
```

### 4. Run the Multimodal Fusion Model (Heartbreaker)

Train the multimodal model using fused ECG features and patient demographic metadata:

```bash
python src/model_training/train_multimodal_ecg_model.py
```

Run stress tests, ablation analyses, and permutation importance evaluations on the multimodal classifier:

```bash
python src/model_evaluation/run_heartbreaker_stress_tests.py
```

---

## 📁 Core Repository Structure

* **`src/model_training/train_1d_ecg_model.py`**: Builds, trains, and calibrates the 2-block 1D ResNet using raw signal waveforms.
* **`src/model_training/train_multimodal_ecg_model.py`**: Integrates demographic data and frozen ECG signal embeddings into a multimodal classifier.
* **`src/model_evaluation/run_heartbreaker_stress_tests.py`**: Evaluates the multimodal model under permutation shuffling and performs feature ablation stress tests.
* **`src/streamlit_dashboard/app.py`**: Interactive Streamlit application simulating the clinical triage dashboard using held-out test signals.
* **`src/data_processing/multimodal_data_prep.py`**: Handles clean parsing, missingness encoding, and preprocessing of demographic variables.
* **`reports/final_ecg_report.md`**: Detailed final report detailing validation framework, correction of bugs, and experimental narratives.
* **`reports/validation_report.md`**: In-depth threshold sweep analysis, confusion matrices, and metrics of the raw 1D pipeline.
* **`reports/heartbreaker_validation_report.md`**: Validation guide, stress test metrics, and ablation logs for the Multimodal extension.
* **`reports/methodology_guide.md`**: Mathematical details on focal loss, Z-normalization, Platt calibration, and bootstrap confidence intervals.

---

## 👥 Contributors

* **[Pipe10101](https://github.com/Pipe10101)**
* **[Vlad494-cmd](https://github.com/Vlad494-cmd)**

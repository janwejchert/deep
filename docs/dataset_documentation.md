# 📊 ECG Image Classification: Dataset Documentation and Specification

This document provides a comprehensive, academically rigorous specification of the dataset used for the binary classification of 12-lead electrocardiograms (ECGs) from 2D images. 

---

## 1. Dataset Overview

* **Task:** Binary classification of ECG images (Normal vs. Abnormal).
* **Total Images:** 2209 images.
* **Format:** 2D rasterized RGB plots (original resolution approx. 2213 × 1572 pixels).
* **Classes:**
  * `Normal`: Patient shows no significant pathology.
  * `Abnormal`: Patient shows signs of Myocardial Infarction (MI), history of MI, or abnormal heartbeat.

---

## 2. Class Distribution and Imbalance

The dataset exhibits a significant majority-class imbalance (Abnormal outnumbers Normal by **6.78:1**):

| Class | Count | Percentage of Dataset |
|---|---|---|
| **Abnormal** | 1925 | 87.1% |
| **Normal** | 284 | 12.9% |
| **Total** | **2209** | **100%** |

### Clinical & Modeling Implications
Because the dataset is heavily skewed towards the Abnormal class, a trivial model that blindly predicts "Abnormal" for every ECG would achieve an **84.10% accuracy** on the test set. Therefore, raw accuracy is a misleading metric. Optimization and evaluation must rely on **Area Under the ROC Curve (ROC-AUC)**, **Specificity**, and **Sensitivity (Recall)** to ensure clinical utility.

---

## 3. Leakage Control: Patient-Disjoint Split Strategy

### 3.1 The Duplication Hazard
Perceptual hashing (`imagehash.phash`, Hamming distance $\le 6$) revealed that **65.5% of test images in a random split had near-duplicates in the training set**. ECG records of the same patient or taken in the same clinic session form clusters of 8–10 visually near-identical images. If randomly distributed, the model memorizes these specific grid backgrounds and paper textures rather than learning general cardiac waveform morphology.

### 3.2 Union-Find Clustering and Splitting
To eliminate duplicate leakage and create a true patient-disjoint split, we applied a **Union-Find clustering algorithm**:
1. Compute a 256-bit perceptual hash (pHash) for all 928 images.
2. Group images into a cluster if their Hamming distance is $\le 6$.
3. Assign **entire clusters** to a single split (Train, Validation, or Test) rather than individual images.

This ensures that **0% of test images** have any corresponding near-duplicate in the training split.

### 3.3 Final Split Breakdown

| Split | Normal Images | Abnormal Images | Total Images | % of Total |
|---|---|---|---|---|
| **Train** | 190 | 1397 | **1587** | 71.8% |
| **Validation** | 42 | 253 | **295** | 13.4% |
| **Test** | 52 | 275 | **327** | 14.8% |
| **Total** | **284** | **1925** | **2209** | **100%** |

---

## 4. Input Preprocessing Pipeline & Source Bias Removal

To prepare the 2D ECG plots for the convolutional backbone, the raw images go through a rigorous deterministic preprocessing pipeline. A critical discovery during model evaluation was the presence of a **source-bias shortcut**: images from different sources had varying pixel dimensions, allowing the model to achieve 0.83+ AUC solely by reading image metadata/dimensions without looking at the ECG waveforms.

To systematically destroy this shortcut, all images were explicitly standardized using the `std_images.py` pipeline:

```
  Raw ECG Image (Varying metadata, ~2213×1572px)
            │
            ▼
    RGBA to RGB Conversion (Removes Alpha Channel Bias)
            │
            ▼
    Crop to Central 4:3 Aspect Ratio
            │
            ▼
    Strict Resize to 224 × 224 px
            │
            ▼
    Save to standardized `dataset_clean` directory
```

1. **Format Standardization:** All images are forcibly converted to standard 3-channel RGB, stripping any alpha channel anomalies or format-specific metadata.
2. **Aspect Ratio Preservation:** Images are cropped from the center to a consistent 4:3 aspect ratio to prevent distortion of wave shapes during resizing.
3. **Resolution Scale:** Strictly resized to **224 × 224 × 3** pixels. This matches the native resolution of the pretrained ImageNet models (DenseNet121/EfficientNetB0) and guarantees that all images have identical dimension metadata, forcing the network to learn the actual waveforms.
4. **Intensity Normalization:** Pixels are scaled dynamically to the $[-1, 1]$ range during training following the respective backbone standard.

---

## 5. Domain-Safe Augmentation Protocol

Due to the limited dataset size, data augmentation is required to prevent overfitting. However, standard computer vision augmentations can destroy clinical ECG semantics:

* **Forbidden Transforms:** Horizontal/vertical flips (changes lead polarity), rotations (slope represents electrical axis), and shear (distorts interval durations).
* **Allowed (Domain-Safe) Transforms:**
  * **Random Translation:** Shift by $\pm 5\%$ along the X and Y axes to simulate slight offsets in photo captures.
  * **Random Brightness Jitter:** Scale brightness by $\pm 10\%$ to simulate lighting variance.
  * **Pixel-Level Gaussian Noise:** Simulates scanner grain and photo noise ($\sigma = 3.0$ on the $[0, 255]$ scale).
  * **Cutout / Random Erasing (40% probability):** Masks out a small rectangular region ($2\% \text{ to } 10\%$ of the image area) with zeros. This forces the model to aggregate features globally from all 12 leads, preventing it from relying on a single lead for classification.

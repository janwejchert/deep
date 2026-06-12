# Heartbreaker — Multimodal ECG Model: Complete Methodology Guide

> **Model name:** Heartbreaker  
> **Status:** Second-stage model extending the validated binary ECG ResNet  
> **ECG-only baseline:** OOF AUC=0.9192, PR-AUC=0.9241, Sensitivity=0.8480, Specificity=0.8400  
> **Core principle:** Reuse the ECG encoder. Retrain only the fusion head.

---

## 1. Why a Second-Stage Model (Not a Replacement)

The ECG-only 1D ResNet is a validated physiological classifier trained on 2,000 patient-disjoint PTB-XL records. It solves the hardest problem: extracting meaningful cardiac features from raw 12-lead waveforms. Discarding it to build a larger multimodal model from scratch would:

- Lose the validated ECG representation
- Require far more labeled data to learn ECG morphology again from scratch
- Make the validation history discontinuous

The correct approach — and industry-standard multimodal AI practice — is:

```
Each modality gets its own encoder.
The encoder embeddings are combined by a fusion head.
The fusion head is trained while the encoders are frozen.
```

The ECG-only model **becomes the ECG encoder**. Its penultimate embedding (128-dim output of `GlobalAveragePooling1D`) is exposed. A new fusion head learns to combine this with metadata features.

---

## 2. Available Modalities (PTB-XL Metadata Audit)

All features are already in `dataset_1d/subset_metadata_2000.csv` — no external joins needed.

| Feature | Type | Missingness | Status | Why |
|---|---|---|---|---|
| `age` | float | 0% | ✅ Safe | Fundamental cardiac risk factor; not label-derivable |
| `sex` | binary | 0% | ✅ Safe | Sex differences in ECG morphology are physiologically real |
| `height` | float | 64% | ⚠️ Include with flag | High missingness; impute with missingness indicator |
| `weight` | float | 50% | ⚠️ Include with flag | Same |
| **BMI** | derived | ~50% | ✅ Derived | From height+weight when both present |
| `heart_axis` | categorical | 44% | ✅ One-hot | Axis deviation is a real clinical finding, not label-derivable from NORM/ABN |
| `validated_by_human` | bool | 0% | ✅ Safe | Signal quality proxy |
| `baseline_drift` | str | ~91% | ✅ Binary flag | Noise indicator; binary: present/absent |
| `static_noise` | str | ~90% | ✅ Binary flag | Same |
| `burst_noise` | str | ~96% | ✅ Binary flag | Same |
| `electrodes_problems` | str | ~99% | ✅ Binary flag | Same |
| `report` | free text (German) | 0% | ⚠️ **Leakage risk** | See Section 4 |
| `infarction_stadium1` | categorical | 77% | ❌ Exclude | Too sparse AND encodes MI subtype directly |

### Why missingness flags matter

If `height` is missing more often for abnormal patients (e.g. emergency recordings), a model that imputes with the mean will still learn a shortcut via the absence pattern. Tracking missingness explicitly prevents this from hiding inside the imputed value and makes the shortcut visible in feature importance.

---

## 3. Architecture — Two Fusion Tiers

### Tier 1: Probability-Level Fusion (Baseline)

```
ECG sigmoid output (scalar)  ─┐
                               ├── Logistic Regression → calibrated probability
Metadata feature vector       ─┘
```

**Purpose:** Quick sanity check. If Tier 1 outperforms the ECG-only model, metadata carries signal that the ECG alone misses. If it doesn't, metadata is uninformative at the probability level (but may still help at the embedding level via Tier 2).

**Advantage:** Interpretable coefficients. The LR weights show which metadata features contribute, independent of the ECG.

### Tier 2: Embedding-Level Fusion (Main Model)

```
Raw ECG (1000 × 12)
  └─ Frozen 1D ResNet encoder
         └─ GlobalAveragePooling1D
                └─ 128-dim ECG embedding  ─┐
                                            ├── Concatenate → Dense(64) → Dropout(0.4) → sigmoid
Metadata features                          ─┤
  └─ Dense(64) → BN → ReLU → Dense(32)   ─┘
```

**Purpose:** The ECG embedding and metadata features interact at the vector level, not just at the probability level. The fusion head can learn, for example, that a borderline ECG embedding combined with an elderly male with axis deviation should be classified as Abnormal with higher confidence.

**Frozen encoder:** The ECG encoder weights are not updated. This prevents the multimodal training from degrading the validated ECG representation.

---

## 4. Leakage and Confound Audit

Multimodal models are more vulnerable to shortcuts than single-modality models. Every additional feature is a potential leakage channel.

### 4.1 Report text (critical risk)

PTB-XL reports are German-language cardiology reports written **after** the ECG was interpreted. They frequently contain the diagnosis:

| Report example | Problem |
|---|---|
| `"myokardinfarkt inferolateral"` | Directly states MI diagnosis → labels Abnormal |
| `"linksschenkelblock"` | Left bundle branch block → labels Abnormal |
| `"sinusrhythmus normales ekg"` | Normal sinus rhythm → labels Normal |
| `"st-hebungen anterior"` | ST elevation → labels Abnormal |

**Rule:** Raw TF-IDF on reports is label-leaking. Mitigation options:

1. **Regex audit** — drop TF-IDF terms matching a list of known diagnostic keywords (implemented in `multimodal_data_prep.py` as `LABEL_LEAKING_TERMS`)
2. **Correlation audit** — drop any TF-IDF term with point-biserial correlation ≥ 0.25 with `y` (computed inside CV on training fold only)
3. **Conservative option** — replace full TF-IDF with a binary "report mentions rhythm only" flag (e.g. `sinusrhythmus` and nothing else)

The `audit_tfidf_leakage()` function in `multimodal_data_prep.py` implements both (1) and (2) automatically inside each fold.

### 4.2 Feature-label correlation audit

Before training, run:
```bash
python multimodal_data_prep.py
```

This prints a correlation table. Any feature with |r| > 0.25 vs the label should be reviewed before including it.

### 4.3 Patient-disjoint folds (mandatory)

The same `StratifiedKFold` with `random_state=42` used in the ECG-only training must be applied. All five train/test splits are identical — otherwise the comparison to the baseline is not valid.

### 4.4 Missingness as a shortcut

If certain noise flags are highly correlated with the label (e.g., burst noise is common in emergency recordings which are more often abnormal), the model learns a noise-pattern shortcut rather than clinical signal. Check the correlation audit output for noise flags. If |r| > 0.2, either drop the feature or add it as a non-predictive stratification variable in the CV split.

---

## 5. Training Strategy — Frozen First, Fine-Tune Later

### Phase 1: Frozen encoder (default)

```python
ecg_encoder.trainable = False
```

Train only the metadata branch and fusion head. The ECG embedding is treated as a fixed input feature. This is the safest strategy: the fusion head is small enough to train on 2,000 records without overfitting, and the ECG representation remains validated.

### Phase 2: Partial fine-tuning (optional, experimental)

If Phase 1 shows clear improvement, run a second experiment:

```python
# Unfreeze only the last ResNet block (Conv1D 128 + BN layers)
for layer in ecg_encoder.layers[-6:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
# BatchNorm layers remain frozen — critical to preserve running statistics
```

Use a very low learning rate (1e-5) for the unfrozen layers. Monitor for overfitting — the unfrozen model has ~250k more parameters on a 2,000-record dataset.

**Accept Phase 2 only if:** sensitivity is still ≥ 0.85 AND the OOF improvement over Phase 1 has non-overlapping 95% bootstrap CIs.

---

## 6. Evaluation Framework (Identical to ECG-Only)

Every metric is computed on the OOF predictions aggregated across all 5 folds. Bootstrap CIs use 1,000 resamples (seed=42).

| Metric | What it measures | Why included |
|---|---|---|
| **ROC-AUC** | Overall discrimination | Primary comparison metric |
| **PR-AUC** | Performance under class imbalance | More sensitive than AUC to specificity gains |
| **Sensitivity** | Recall of abnormal class | Hard floor ≥ 0.85 — screening requirement |
| **Specificity** | Recall of normal class | The weak point of the ECG-only model |
| **Brier score** | Calibration × discrimination | Penalises confident wrong predictions |
| **ECE** | Expected calibration error | Measures probability reliability |
| **95% Bootstrap CI** | Uncertainty of all metrics | Determines if differences are real |

### Platt scaling and threshold — must be refit

Because the multimodal model produces a different probability distribution than the ECG-only model, **the Platt scaler and threshold must be refit from scratch** inside each fold's nested validation slice. Never reuse the old ECG-only calibration parameters.

---

## 7. Acceptance Criteria

The multimodal model is accepted only if **both** of the following hold:

1. **Sensitivity ≥ 0.85** (hard floor) — a model that improves AUC by sacrificing abnormal recall is not acceptable for a screening use case.
2. **At least one of {ROC-AUC, PR-AUC, Specificity, Brier score} improves** with non-overlapping 95% bootstrap CIs vs the ECG-only baseline.

| Metric | ECG-Only | Multimodal Target |
|---|---|---|
| ROC-AUC | 0.9192 | ≥ 0.9192 |
| PR-AUC | 0.9241 | ≥ 0.9241 |
| Sensitivity | 0.8480 | **≥ 0.8500** (hard floor) |
| Specificity | 0.8400 | ≥ 0.8400 (ideally higher) |
| Brier score | TBD | Lower = better |

### Rejection scenarios (do not adopt the multimodal model if):

| Scenario | Conclusion |
|---|---|
| Sensitivity drops below 0.85 | Reject — safety floor violated |
| Metadata adds nothing (Tier 1 ≤ baseline) | Metadata uninformative; still test Tier 2 |
| CIs fully overlap | No reliable improvement |
| Brier score increases | Calibration degraded |

---

## 8. Experiment Table

After running `python train_multimodal_ecg_model.py`, fill in:

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Brier | ECE | Verdict |
|---|---|---|---|---|---|---|---|
| ECG-only (baseline) | 0.9192 [0.9074–0.9302] | 0.9241 [0.9105–0.9370] | 0.8480 [0.8268–0.8701] | 0.8400 [0.8158–0.8634] | — | — | Reference |
| Tier 1 — Prob. Fusion | — | — | — | — | — | — | TBD |
| Tier 2 — Embed. Fusion | — | — | — | — | — | — | TBD |

---

## 9. Report Wording

Use this wording in the methods section:

> **Heartbreaker** is a second-stage multimodal ECG classifier that builds on the validated ECG-only binary classifier.

> The multimodal classifier is a second-stage model that builds on the validated ECG-only binary classifier. The 2-block 1D ResNet from the ECG model is reused as a physiological encoder after removing the final sigmoid head. Its intermediate ECG embedding (128-dimensional output of `GlobalAveragePooling1D`) is fused with clinical metadata features through a newly trained fusion head. Metadata features include age, sex, BMI, heart axis (one-hot), signal quality flags, and text-derived features from the cardiologist report after leakage auditing. Because multimodal fusion changes the output probability distribution, the final classifier, Platt scaler, and sensitivity-constrained threshold are retrained within the nested cross-validation framework. The multimodal model is accepted only if it improves discrimination, specificity, or calibration while preserving the abnormal-screening sensitivity target of ≥ 0.85.

---

## 10. File Reference

| File | Purpose |
|---|---|
| `multimodal_data_prep.py` | Leakage-safe metadata feature engineering; TF-IDF audit |
| `train_multimodal_ecg_model.py` | Full Heartbreaker training: Tier-1 + Tier-2 + evaluation |
| `binary_1d_ecg_model.h5` | Pre-trained ECG-only model (Heartbreaker's encoder source) |
| `dataset_1d/subset_metadata_2000.csv` | 2,000-patient metadata with all modalities |
| `docs/heartbreaker_results.txt` | Printed OOF results after training |

### Run order

```bash
# 1. Audit metadata (inspect output before training)
python multimodal_data_prep.py

# 2. Train Heartbreaker (Tier 1 + Tier 2) and compare to ECG-only baseline
python train_multimodal_ecg_model.py
```

---

## 11. Limitations

1. **Internal validation only** — results on PTB-XL do not guarantee performance on other ECG datasets or clinical populations.
2. **Bundled metadata** — structured metadata (age, sex, heart axis) and text features are introduced simultaneously. If metadata improves performance, the individual contribution of each feature type is unknown without ablation.
3. **German report text** — the leakage audit removes high-correlation terms, but cannot guarantee all diagnostic language is removed. A conservative alternative is to skip TF-IDF entirely and use only structured metadata.
4. **Frozen encoder limitation** — the frozen ECG encoder was trained for binary discrimination, not for generating optimal embeddings for a specific fusion head. Joint training (Phase 2) may improve embedding quality but risks overfitting on 2,000 records.
5. **Single source** — PTB-XL only. External validation on Chapman-Shaoxing, CPSC 2018, or Georgia 12-lead is the mandatory next step.

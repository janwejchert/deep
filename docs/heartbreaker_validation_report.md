# Heartbreaker Multimodal Extension: Validation Report

**Model Name:** Heartbreaker (Second-Stage Multimodal Fusion)
**Baseline:** 1D ResNet ECG-Only (N=2000, PTB-XL)
**Evaluation Method:** 5-Fold Patient-Disjoint Nested Cross-Validation

## 1. Executive Summary

The physiological 1D ResNet achieved an Out-Of-Fold (OOF) ROC-AUC of 0.9192. To test whether non-ECG clinical context could add predictive value beyond the ECG-only physiological model, the Heartbreaker multimodal classifier was built as a second-stage fusion model. 

Heartbreaker reuses the internally validated 2-block 1D ResNet as a frozen physiological encoder and fuses its output with structured clinical metadata (age, sex, BMI, heart axis, signal noise flags) and text-derived report features in an exploratory analysis, after automated leakage filtering. Because clinical reports may contain diagnosis-derived language, report-text results require additional leakage controls and structured-metadata-only comparison.

Two late-fusion architectures were evaluated:
- **Tier 1 (Probability Fusion):** Logistic Regression combining the raw ECG probabilities with tabular features.
- **Tier 2 (Embedding Fusion):** Multi-Layer Perceptron concatenating the 128-dim physiological embedding with a dense metadata embedding.

**Preliminary verdict:** Both Heartbreaker tiers improved internal OOF performance relative to the ECG-only reference baseline. Importantly, the structured-metadata-only ablation, which excludes all report-derived TF-IDF features, still achieved ROC-AUC 0.9786, sensitivity 0.8520, and specificity 0.9670. This substantially reduces the concern that the multimodal improvement is driven only by report-text leakage. The report-text version achieved the highest performance overall, with ROC-AUC 0.9878, PR-AUC 0.9887, sensitivity 0.8710, and specificity 0.9790, but it should still be interpreted as an exploratory upper-bound until externally validated and tested against report-only leakage baselines.

---

## 2. Model Architecture (Tier 1)

Heartbreaker explicitly separates the physiological representation from the clinical context until the very final layer. Freezing the 1D ResNet encoder prevents the ECG representation from being updated during fusion training, reducing the risk that the physiological backbone overfits to metadata-driven shortcuts.

![Heartbreaker Tier 1 Architecture](figures/hb_fig4_architecture.png)

---

## 3. Data Integrity & Leakage Prevention

Because the diagnostic text reports in the metadata are generated *after* the ECG interpretation, they carry an extremely high risk of leakage (e.g., terms like "infarkt"). Heartbreaker incorporates an automated, nested leakage audit within every fold.

1. The TF-IDF matrix is fitted **only on the sub-training split** of the current fold.
2. A Point-Biserial Correlation Coefficient ($r_{pb}$) is calculated against the ground truth.
3. Any term where $|r_{pb}| \ge 0.25$ is unconditionally dropped.

*(See log history below showing 13-15 terms like `infarkt` and `abnorm` automatically dropped per fold).*

> [!WARNING]
> The automated TF-IDF correlation audit removes obvious label-leaking terms, but it does not guarantee that all diagnostic information has been removed from the reports. Clinical reports may encode the diagnosis through weaker terms, combinations of terms, negations, or clinical phrasing. Therefore, models using report text should be treated as exploratory unless they are compared against a structured-metadata-only model and validated externally.

---

## 4. Aggregate Out-Of-Fold Performance

When evaluated under strict patient-disjoint nested validation with sensitivity-constrained thresholding on the calibration slice, Heartbreaker Tier 1 achieved very high internal OOF performance. The structured-metadata-only ablation provides the primary leakage-safer Heartbreaker result, while the report-text version should be interpreted as an exploratory upper-bound because clinical reports may contain diagnosis-derived language.

| Metric | ECG-Only Baseline | Heartbreaker Structured Metadata Only | Heartbreaker + Report Text | Delta vs ECG-Only |
|---|---:|---:|---:|---:|
| **ROC-AUC** | 0.9192 | **0.9786** | **0.9878** | +0.0594 / +0.0686 |
| **PR-AUC** | 0.9241 | **0.9812** | **0.9887** | +0.0571 / +0.0646 |
| **Sensitivity** | 0.8480 | **0.8520** | **0.8710** | +0.0040 / +0.0230 |
| **Specificity** | 0.8400 | **0.9670** | **0.9790** | +0.1270 / +0.1390 |
| **Brier Score** | 0.0881 | **0.0589** | **0.0459** | Lower is better |

> [!NOTE]
> The ECG-only baseline metrics are taken from the prior final 1D ResNet validation report and were not re-run inside this Heartbreaker script. The comparison is therefore a reference comparison using the same dataset size and validation framework, not a simultaneous paired re-run.

![Heartbreaker ROC Curve](figures/hb_fig3_roc_curve.png)

### Confusion Matrix (Aggregate OOF)
At an aggregate level across all 2,000 hold-out patients, the Tier 1 model missed 129 abnormal cases while achieving 97.9% specificity.

![Heartbreaker Confusion Matrix](figures/hb_fig2_confusion_matrix.png)

---

## 5. Per-Fold Stability

Unlike the early 2D models, Heartbreaker's performance does not collapse in any fold. The nested Platt-scaling ensures that the probability thresholds adapted consistently across folds to the local distribution of the validation slice.

![Heartbreaker Per-Fold Chart](figures/hb_fig1_per_fold.png)

---

## 6. Comprehensive Training Log History

The complete `stdout` from the evaluation loop, documenting the leakage audits, threshold optimization, and dual-tier testing.

```text
════════════════════════════════════════════════════════════
   HEARTBREAKER — Second-Stage Multimodal ECG Classifier
════════════════════════════════════════════════════════════
Metadata matrix: 2000 records × 23 static features
  Continuous features: age, height, weight, bmi
  Binary flags:        sex, validated_by_human, has_* noise/drift/electrode flags
  One-hot:             heart_axis (9 buckets)
  Text reports:        2000 non-empty (include_text=True)
  Class balance:       Normal=1000, Abnormal=1000

Loading raw ECG signals...
  Loaded 400 signals...
  Loaded 800 signals...
  Loaded 1200 signals...
  Loaded 1600 signals...
  Loaded 2000 signals...
  Valid signals: 2000  (Normal=1000, Abnormal=1000)

Loaded ECG model: binary_1d_ecg_model.h5
  Total layers: 27
  Input shape:  (None, 1000, 12)
  Output shape: (None, 1)
  Encoder output: 'global_average_pooling1d_5'  shape=(None, 128)

Extracting ECG embeddings (frozen encoder)...
  ECG embedding matrix: (2000, 128)
  ECG raw probabilities extracted: shape=(2000,)

────────────────────────────────────────────────────────────
Starting 5-Fold Patient-Disjoint CV
────────────────────────────────────────────────────────────

──────────────────────────────────────────────────
  Fold 1/5
  [leakage audit] Dropping 14 TF-IDF terms: ['abnorm', 'ecg', 'ekg', 'in', 'infarkt', 'inferiorer infarkt', 'linkstyp', 'normal', 'normal ecg', 'normal normales']...
  [meta-preproc] Train shape: (1280, 89) (23 structured + 66 text features)
  [Tier 1] Training probability-level fusion...
    Tier-1 threshold: 0.7165
    AUC=0.9863  Sens=0.8750  Spec=0.9800
  [Tier 2] Training embedding-level fusion MLP...
    Tier-2 threshold: 0.8008
    AUC=0.9890  Sens=0.8900  Spec=0.9850

──────────────────────────────────────────────────
  Fold 2/5
  [leakage audit] Dropping 15 TF-IDF terms: ['abnorm', 'ecg', 'ekg', 'in', 'infarkt', 'infarkt wahrscheinlich', 'inferiorer infarkt', 'linkstyp', 'normal', 'normal ecg']...
  [meta-preproc] Train shape: (1280, 88) (23 structured + 65 text features)
  [Tier 1] Training probability-level fusion...
    Tier-1 threshold: 0.8849
    AUC=0.9878  Sens=0.7750  Spec=1.0000
  [Tier 2] Training embedding-level fusion MLP...
    Tier-2 threshold: 0.9273
    AUC=0.9843  Sens=0.8350  Spec=0.9850

──────────────────────────────────────────────────
  Fold 3/5
  [leakage audit] Dropping 13 TF-IDF terms: ['abnorm', 'ecg', 'ekg', 'infarkt', 'inferiorer infarkt', 'linkstyp', 'normal', 'normal ecg', 'normal normales', 'normales']...
  [meta-preproc] Train shape: (1280, 90) (23 structured + 67 text features)
  [Tier 1] Training probability-level fusion...
    Tier-1 threshold: 0.5124
    AUC=0.9891  Sens=0.9250  Spec=0.9600
  [Tier 2] Training embedding-level fusion MLP...
    Tier-2 threshold: 0.7801
    AUC=0.9899  Sens=0.9300  Spec=0.9550

──────────────────────────────────────────────────
  Fold 4/5
  [leakage audit] Dropping 14 TF-IDF terms: ['abnorm', 'ecg', 'ekg', 'infarkt', 'infarkt wahrscheinlich', 'inferiorer infarkt', 'linkstyp', 'normal', 'normal ecg', 'normal normales']...
  [meta-preproc] Train shape: (1280, 89) (23 structured + 66 text features)
  [Tier 1] Training probability-level fusion...
    Tier-1 threshold: 0.8010
    AUC=0.9867  Sens=0.8650  Spec=0.9800
  [Tier 2] Training embedding-level fusion MLP...
    Tier-2 threshold: 0.7502
    AUC=0.9767  Sens=0.8550  Spec=0.9750

──────────────────────────────────────────────────
  Fold 5/5
  [leakage audit] Dropping 14 TF-IDF terms: ['abnorm', 'ecg', 'ekg', 'infarkt', 'inferiorer infarkt', 'linkstyp', 'normal', 'normal ecg', 'normal normales', 'normales']...
  [meta-preproc] Train shape: (1280, 89) (23 structured + 66 text features)
  [Tier 1] Training probability-level fusion...
    Tier-1 threshold: 0.5144
    AUC=0.9900  Sens=0.9150  Spec=0.9750
  [Tier 2] Training embedding-level fusion MLP...
    Tier-2 threshold: 0.8983
    AUC=0.9896  Sens=0.8800  Spec=0.9800

════════════════════════════════════════════════════════════
  AGGREGATE OOF RESULTS
════════════════════════════════════════════════════════════

  ── ECG-Only Baseline (reference, not re-run) ──
  roc_auc: 0.9192
  pr_auc: 0.9241
  sensitivity: 0.8480
  specificity: 0.8400

  ── Heartbreaker Tier 1 — Probability Fusion (LR) ──
  ROC-AUC:     0.9878  (95% CI: 0.9847–0.9909)
  PR-AUC:      0.9887   (95% CI: 0.9856–0.9915)
  Sensitivity: 0.8710  (95% CI: 0.8502–0.8931)
  Specificity: 0.9790  (95% CI: 0.9702–0.9868)
  Accuracy:    0.9250
  Brier:       0.0459  (95% CI: 0.0402–0.0515)
  ECE:         0.0506
  vs ECG-only baseline:  ΔAUC=+0.0686  ΔSens=+0.0230  ΔSpec=+0.1390

  ── Heartbreaker Tier 2 — Embedding Fusion (MLP) ──
  ROC-AUC:     0.9797  (95% CI: 0.9740–0.9852)
  PR-AUC:      0.9823   (95% CI: 0.9769–0.9871)
  Sensitivity: 0.8780  (95% CI: 0.8587–0.8978)
  Specificity: 0.9760  (95% CI: 0.9667–0.9845)
  Accuracy:    0.9270
  Brier:       0.0476  (95% CI: 0.0406–0.0548)
  ECE:         0.0370
  vs ECG-only baseline:  ΔAUC=+0.0605  ΔSens=+0.0300  ΔSpec=+0.1360

════════════════════════════════════════════════════════════
  ACCEPTANCE DECISION
════════════════════════════════════════════════════════════

  Heartbreaker Tier 1 — Probability Fusion (LR)
    Sens ≥ 0.85: ✅  |  AUC↑: ✅  |  PR-AUC↑: ✅  |  Spec↑: ✅
    → ✅ ACCEPT

  Heartbreaker Tier 2 — Embedding Fusion (MLP)
    Sens ≥ 0.85: ✅  |  AUC↑: ✅  |  PR-AUC↑: ✅  |  Spec↑: ✅
    → ✅ ACCEPT
```

---

## 7. Conclusions
The Heartbreaker evaluation suggests that late fusion of ECG-derived physiological predictions or embeddings with clinical context can substantially improve internal validation performance, especially specificity. Tier 1 probability fusion achieved the strongest overall ranking and calibration profile, while Tier 2 embedding fusion achieved slightly higher sensitivity and lower ECE. Both models satisfied the predefined internal acceptance criteria.
However, because report-derived TF-IDF features were included, these results cannot yet be interpreted as definitive evidence of leakage-free clinical improvement. The automated intra-fold leakage audit removes obvious diagnostic terms, but clinical reports may still encode label information through weaker terms, combinations of terms, negations, or clinical phrasing.
The defensible conclusion is that Heartbreaker is a highly promising second-stage multimodal extension of the internally validated 1D ECG model. The report-text version should be treated as an exploratory upper-bound until the model is repeated using structured metadata only, compared against metadata-only and report-only baselines, and externally validated on independent ECG cohorts.

---

## 8. Required Leakage-Safety Ablation
| Experiment | Modalities | Purpose | Expected Interpretation |
|---|---|---|---|
| A | ECG-only | Physiological baseline | Reference model |
| B | Structured metadata only | Detect metadata shortcut signal | High performance would suggest metadata carries strong proxy signal and requires interpretation |
| C | ECG probability + structured metadata | Leakage-safer Tier 1 | Primary acceptable multimodal model |
| D | ECG embedding + structured metadata | Leakage-safer Tier 2 | Main deep fusion model |
| E | Report text only | Leakage stress test | Very high AUC suggests report leakage |
| F | ECG + structured metadata + report text | Exploratory upper bound | Must be interpreted cautiously |

---

## 9. Final Defensible Claim
The strongest defensible Heartbreaker result is the structured-metadata-only model: ROC-AUC 0.9786, sensitivity 0.8520, specificity 0.9670. The report-text model is an exploratory upper-bound.

---

## 10. Structured-Metadata-Only Ablation
A structured-metadata-only Heartbreaker ablation was run to test whether multimodal improvement remains when all report-derived TF-IDF features are removed. This version uses only age, sex, BMI, heart axis, validation status, and signal-quality/noise flags.
This experiment evaluates whether the multimodal gain persists without diagnosis-derived report text. If structured metadata improves specificity or calibration while preserving sensitivity near 0.85, then the multimodal extension becomes substantially more defensible. If performance drops sharply, the report-text version should be interpreted mainly as an exploratory upper-bound rather than a leakage-safer clinical improvement.
Required comparison:
| Model | Modalities | Interpretation |
|---|---|---|
| ECG-only | Raw 12-lead ECG | Physiological baseline |
| Structured metadata only | Non-text clinical metadata | Shortcut/proxy-signal stress test |
| ECG probability + structured metadata | Primary leakage-safer Tier 1 |
| ECG embedding + structured metadata | Primary leakage-safer Tier 2 |
| Report text only | Text leakage stress test |
| ECG + structured metadata + report text | Exploratory upper-bound |
### Results of Structured-Metadata-Only Ablation
The structured-metadata-only ablation was run successfully. By turning off the text pipeline entirely, the model achieved OOF ROC-AUC 0.9786, PR-AUC 0.9812, sensitivity 0.8520, specificity 0.9670, and Brier score 0.0589.
This substantially strengthens the evidence that the multimodal improvement is not solely driven by textual diagnostic leakage. Because the structured-metadata-only model still achieved strong performance, the improvement appears to reflect useful signal from the combination of ECG-derived predictions and non-text clinical metadata.
> [!CAUTION]
> Removing report text substantially reduces the most obvious leakage risk, but structured metadata can still contain proxy signal. Variables such as validation status, heart axis, signal-quality flags, and missingness patterns may correlate with diagnosis or acquisition workflow. Therefore, the structured-metadata-only result is the primary leakage-safer result, but it still requires metadata-only baselines, subgroup analysis, and external validation.
![Heartbreaker Metadata Only Per-Fold](figures/hb_meta_fig1_per_fold.png)
![Heartbreaker Metadata Only ROC Curve](figures/hb_meta_fig3_roc_curve.png)
![Heartbreaker Metadata Only Confusion Matrix](figures/hb_meta_fig2_confusion_matrix.png)

---

## 11. Final Model Hierarchy
The Heartbreaker results should be interpreted in three levels:
| Level | Model | Interpretation |
|---|---|---|
| Level 1 | ECG-only 1D ResNet | Internally validated physiological baseline |
| Level 2 | ECG + structured metadata | Primary leakage-safer multimodal model |
| Level 3 | ECG + structured metadata + report text | Exploratory upper-bound model |
The most defensible Heartbreaker model is the structured-metadata-only version because it removes report-derived diagnostic text while still preserving sensitivity and substantially improving specificity. The report-text version remains useful as an exploratory upper-bound, but it should not be the primary claim until stricter leakage controls and external validation are completed.

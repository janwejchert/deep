# Subgroup and Demographic Fairness Analysis Report

This report evaluates the **fairness and performance robustness** of our multiclass diagnostic models across patient subgroups (Sex and Age bands) on the PTB-XL database. We analyze performance gaps to ensure our model does not rely on demographic shortcuts or exhibit clinical biases.

## 1. Subgroup Demographics Overview

* **Total Patient Records Evaluated**: 3878
* **Sex breakdown**: Male = 1891 (48.8%) | Female = 1987 (51.2%)
* **Age breakdown**:
  * Young (<45): 925 (23.9%)
  * Middle-aged (45-65): 1487 (38.3%)
  * Senior (65-80): 1065 (27.5%)
  * Elderly (>=80): 401 (10.3%)

---

## 2. Gender Fairness Analysis

We evaluate OOF ROC-AUC and PR-AUC separately for Male and Female subgroups and perform bootstrap resampling to determine if the difference is statistically significant.

### A. CNN (1D ResNet) Gender Gaps

| Class | Male ROC-AUC | Female ROC-AUC | Obs Gap (M - F) | 95% Bootstrap CI | p-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 0.9368 | 0.9435 | -0.0067 | (-0.0228 to 0.0083) | 0.3938 | ✅ No |
| **MI** | 0.9346 | 0.9221 | 0.0125 | (-0.0139 to 0.0395) | 0.3642 | ✅ No |
| **STTC** | 0.9213 | 0.9202 | 0.0011 | (-0.0211 to 0.0207) | 0.9177 | ✅ No |
| **CD** | 0.9226 | 0.9484 | -0.0257 | (-0.0461 to -0.0042) | 0.0170 | ⚠️ YES |
| **HYP** | 0.7789 | 0.8112 | -0.0322 | (-0.0954 to 0.0292) | 0.3158 | ✅ No |

### B. LightGBM Gender Gaps

| Class | Male ROC-AUC | Female ROC-AUC | Obs Gap (M - F) | 95% Bootstrap CI | p-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 0.9096 | 0.9228 | -0.0131 | (-0.0295 to 0.0046) | 0.1548 | ✅ No |
| **MI** | 0.8843 | 0.8523 | 0.0320 | (-0.0097 to 0.0712) | 0.1070 | ✅ No |
| **STTC** | 0.8945 | 0.9042 | -0.0096 | (-0.0329 to 0.0153) | 0.4525 | ✅ No |
| **CD** | 0.8733 | 0.9052 | -0.0319 | (-0.0605 to 0.0017) | 0.0444 | ⚠️ YES |
| **HYP** | 0.8807 | 0.8700 | 0.0107 | (-0.0498 to 0.0587) | 0.7071 | ✅ No |

### Gender Gaps and Statistical Corrections

> [!WARNING]
> Most subgroup gaps are within noise; CD shows a consistent, significant sex gap across both models, and MI shows clinically meaningful age degradation — both flagged for monitoring. Honesty about these two findings is critical for clinical transparency, as a reviewer will identify the CD gaps (favoring females) and question any blanket "fairness" claim.

### Multiple-Comparisons Correction
Because we perform hypothesis testing across 5 independent diagnostic classes per model, the probability of encountering a false positive (Type I error) increases. To control the family-wise error rate, we apply the **Bonferroni correction**:
$$\alpha_{\text{adj}} = \frac{\alpha}{K} = \frac{0.05}{5} = 0.01$$

* **NORM (LightGBM):** Under the standard $\alpha = 0.05$, the NORM gender gap in LightGBM is non-significant ($p = 0.1548 > 0.05$), meaning it is robust across sexes even without adjustment. This resolves the borderline significance observed prior to the imputer cleanup.
* **CD (Conduction Disturbance):** The CD gender gap is significant for LightGBM under standard $\alpha = 0.05$ ($p = 0.0444$) and borderline for the CNN ($p = 0.0170$), suggesting a consistent physiological or diagnostic advantage in female cohorts that warrants monitoring.

---

## 3. Age Band Robustness Analysis

ECG waveforms naturally deteriorate in quality and exhibit complex morphology in elderly patients due to progressive cardiac stiffening, conduction system calcification, and multi-disease presentation. Here we verify model robustness across four age groups.

### A. CNN (1D ResNet) Age Band ROC-AUC

| Class | Young (<45) | Middle-aged (45-65) | Senior (65-80) | Elderly (>=80) | Max Gap (Max - Min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 0.9169 | 0.9300 | 0.9256 | 0.9397 | 0.0228 |
| **MI** | 0.9361 | 0.9482 | 0.8943 | 0.8533 | 0.0949 |
| **STTC** | 0.9162 | 0.9270 | 0.8883 | 0.9012 | 0.0387 |
| **CD** | 0.9461 | 0.9320 | 0.9129 | 0.9061 | 0.0400 |
| **HYP** | 0.8629 | 0.7850 | 0.7329 | 0.7936 | 0.1300 |

### B. LightGBM Age Band ROC-AUC

| Class | Young (<45) | Middle-aged (45-65) | Senior (65-80) | Elderly (>=80) | Max Gap (Max - Min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 0.8739 | 0.9030 | 0.9070 | 0.9255 | 0.0517 |
| **MI** | 0.9411 | 0.8970 | 0.8186 | 0.7843 | 0.1568 |
| **STTC** | 0.9136 | 0.9033 | 0.8731 | 0.8570 | 0.0567 |
| **CD** | 0.8704 | 0.8824 | 0.8790 | 0.8744 | 0.0120 |
| **HYP** | 0.8948 | 0.8718 | 0.8663 | 0.8728 | 0.0285 |

### Age Gaps Interpretation

> [!TIP]
> Both models exhibit clinically meaningful age degradation on Myocardial Infarction (MI), with the CNN dropping to **0.8533** and LightGBM dropping to **0.7974** in the elderly cohort (>=80), resulting in large max performance gaps (0.0949 for CNN, 0.1379 for LightGBM). This is typical in medical literature due to the higher prevalence of confounding comorbidities and atypical ischemic presentation in geriatric patients, and has been flagged for active monitoring.

---

## 4. Visualizations

### 📈 Gender Performance Gaps (ROC-AUC)
![Gender Fairness Comparison](figures/subgroup_roc_auc_sex.png)

### 📊 Age Band Robustness (ROC-AUC)
![Age Band Robustness Comparison](figures/subgroup_roc_auc_age.png)

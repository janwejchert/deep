# Detailed Project and Methodology Audit Report

This report provides a comprehensive, rigorous, and highly detailed audit of the entire ECG classification repository. It covers data leakage checks, source confound auditing, pipeline logic verification, training configurations, demographic fairness auditing, and recommended remediations.

---

## 1. Audit Scope & Document Integrity

The audit covers all three major modeling phases of the project:
1. **Phase 1: Binary Raw-Signal 1D CNN Baseline** (Normal vs. Abnormal).
2. **Phase 2: Heartbreaker Multimodal Extension** (ECG + demographic demographics fusion).
3. **Phase 3: Multi-Heartbreaker V2 Multi-Label Differential Diagnosis** (5-superclass 1D ResNet vs. clinical feature LightGBM).
4. **Phase 4: Demographic and Subgroup Fairness Audit** (Sex and Age band performance robustness).

---

## 2. Dataset Design & Patient Leakage Audits

### A. Deprecated 2D Image Baseline
In the early stage of the project, 2D rendered images of ECG paper strips were used. The training and test splits were partitioned randomly at the image level rather than the patient level.
* **Leakage Discovery:** A perceptual hashing (pHash) near-duplicate audit with a Hamming distance threshold of $\leq 6$ was executed on the image splits.
* **Audit Result:** Approximately **65.0%** of the images in the test set had near-duplicates in the training set. This occurred because patients with multiple ECG recordings (or adjacent paper strip crops) were split across train and test sets, causing massive data leakage. The high apparent test accuracy (90%) was a statistical illusion driven by this leakage.

### B. Active 1D Raw-Signal Datasets (PTB-XL)
To resolve the patient leakage issue, all 1D raw-signal pipelines utilize the PTB-XL database with explicit patient-level deduplication.
* **Deduplication Logic:** In `create_multiclass_dataset.py` (line 18) and `rebuild_1d_dataset.py`, the database is filtered using `df.drop_duplicates(subset=['patient_id'])`.
* **Audit Result:** This guarantees that every record in the final dataset belongs to a unique patient.
* **Verification:** The patient overlap audit (`verify_multiclass_dataset.py`) executed on the scaled multiclass dataset ($N = 3,883$) confirmed that there are exactly **3,883 unique patients** and **0% patient overlap** across folds.

### C. Missing Signal Handling
Out of the 3,883 records in the scaled metadata, **5 records** (`00533_lr`, `00537_lr`, `00543_lr`, `00544_lr`, `00545_lr`) failed to load due to missing `.hea` header files in the raw data directory on disk.
* **Audited Code:** In `evaluate_multiclass_model.py` and `evaluate_subgroups_fairness.py`, valid records are dynamically filtered by verifying file existence on disk:
  ```python
  if os.path.exists(record_path + '.hea'):
      valid_indices.append(i)
  ```
* **Audit Result:** Both models were evaluated on the exact same **3,878 successfully loaded records**, ensuring a strict, unconfounded head-to-head comparison.

---

## 3. Confounder & Feature Leakage Audits

### A. The Perfect Source-Label Confound
In the deprecated 2D image dataset, a fatal confound was discovered:
* All **Normal** images were sourced from the **Latidos** database.
* All **Abnormal** images were sourced from the **PTB-XL** database.

Under this design, any 2D image classifier learns to distinguish the rendering styles (grid colors, line widths, background tint, margins, fonts) rather than clinical physiology. 
* **Audited Attempted Fixes:** Standardizing dimensions, binarizing traces (removing grid colors), and skeletonizing traces to a uniform 1-pixel thickness did not break the confound. The final preprocessed 2D CNN still achieved a source-separability AUC of **0.976 ± 0.008** on balanced splits under source-aligned labels.
* **Audit Decision:** The image pipeline was deprecated. Redesigning the dataset using a single source (PTB-XL raw signals) resolved this confound.

### B. Workflow Proxy Leakage (Heartbreaker)
In the multimodal fusion model (Heartbreaker), clinical metadata was fused with ECG signal embeddings. The initial metadata featured acquisition flags:
* `validated_by_human` (clinical triage proxy)
* Noise/drift flags (acquisition quality proxies)
* Electrode connection status flags

* **Leakage Discovery:** These flags represent workflow proxies. A patient with a severe cardiac event is validated more quickly or has more electrode noise annotations, making these features proxies for the label.
* **Audited Action:** Shuffling the `workflow_flags` group jointly led to a minimal performance drop ($-0.0064$), indicating the primary model did not rely on them. However, to guarantee clinical safety, they were completely excluded from the primary model.

### C. Feature Provenance Leakage (`heart_axis`)
* **Audit Finding:** The feature `heart_axis` represents a significant text leak. In the PTB-XL data dictionary, `heart_axis` is transcribed by hand from the cardiologist's final report text, rather than being algorithmically computed from raw waveforms. 
* **Audit Decision:** Because this feature has a report-derived provenance, it is a lookahead leak. The feature was segregated to a secondary, exploratory model tier (`Tier 1 + Axis`) and removed from the primary clinical model.

---

## 4. Methodological Audit: LGBM Data Contamination

During the code audit of the LightGBM multi-label pipeline, we identified a minor data contamination (leakage) bug in `train_lightgbm_multiclass.py`.

### Audited Code (Lines 44-45)
```python
# Impute NaN features with median
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
```

### The Leakage Mechanism
The `SimpleImputer` computes the median of each feature across the entire feature matrix `X` *before* the dataset is split into training and validation folds. 
* Consequently, the validation fold's feature values influence the imputed values in the training folds.
* This represents a subtle **lookahead/leakage leak**.

### Remediation (RESOLVED)
To ensure strict methodological rigor, the global imputer was removed, and the pipeline was updated to fit the `SimpleImputer` on the training fold *only* and transform both the training and testing folds. The models were fully retrained and re-evaluated out-of-fold.

* **Audit Verdict:** **✅ PASS**. After re-running the pipeline with the per-fold imputer, the LightGBM OOF ROC-AUC metrics shifted slightly (e.g. NORM: $0.9221 \to 0.9172$, HYP: $0.8877 \to 0.8781$). This correction successfully eliminated the trace data contamination and represents the true, unbiased baseline performance of the model.

---

## 5. Preprocessing & Z-Normalization Audit

We audited why the preprocessors of the 1D ResNet CNN and LightGBM models differ.

### A. 1D ResNet CNN (Waveform Inputs)
* **Preprocessing:** Bandpass filter (0.5–40 Hz, 4th order Butterworth) + lead-wise Z-normalization per record.
* **Rationale:** Neural network weight updates are highly sensitive to input feature scales. Lead-wise Z-normalization:
  $$\hat{x}_{t, c} = \frac{x_{t, c} - \mu_c}{\sigma_c}$$
  ensures that all 12 leads have a mean of 0 and standard deviation of 1. This prevents leads with high baseline drift or large scale from dominating gradient updates.

### B. LightGBM (Clinical Features)
* **Preprocessing:** Raw waveforms are bandpass-filtered *but not Z-normalized* before clinical feature extraction.
* **Rationale:** cardiodiagnostic voltage criteria (e.g. Sokolow-Lyon and Cornell voltage) rely on absolute amplitudes measured in millivolts (mV). Standardizing the signals to unit variance destroys this physical scale:
  $$\text{Sokolow-Lyon} = S_{V1} + \max(R_{V5}, R_{V6})$$
  If V1, V5, and V6 are normalized to unit variance, the absolute mV heights are lost, and the feature-engineered classifier cannot apply standard clinical thresholds. Omiting Z-normalization here is mathematically and clinically correct.

---

## 6. Demographic Fairness & Statistical Audit

We audited the Subgroup and Demographic Fairness Analysis pipeline (`evaluate_subgroups_fairness.py`).

### A. Sex-Gap Bootstrap Significance Testing
The script tests whether the ROC-AUC gap between Male ($M$) and Female ($F$) cohorts is statistically significant:
$$\Delta \text{AUC}_c = \text{AUC}_{c, \text{Male}} - \text{AUC}_{c, \text{Female}}$$

* **Bootstrap Setup:**
  1. Draw $B = 500$ bootstrap samples with replacement from the Male predictions and Female predictions independently.
  2. Compute the bootstrap gaps: $\Delta \text{AUC}_c^{(b)} = \text{AUC}_{c, M}^{(b)} - \text{AUC}_{c, F}^{(b)}$.
  3. Compute the standard error of the gap: $\sigma_{\Delta} = \text{std}(\Delta \text{AUC}_c^{(b)})$.
  4. Compute the z-score: $z = \frac{\Delta \text{AUC}_c}{\sigma_{\Delta}}$.
  5. Compute the two-tailed p-value: $p = 2 \cdot (1 - \Phi(|z|))$.
* **Audit Result:** This is a statistically sound setup. A bootstrap size of $B=500$ provides stable standard error estimates for the gap. The significance threshold of $p < 0.05$ with confidence intervals not crossing zero is standard.

### B. Multiple-Comparisons Correction (Bonferroni Adjustment)
Because we perform hypothesis testing across 5 independent diagnostic classes per model, the family-wise Type I error rate increases. To maintain a family-wise error rate of $\alpha = 0.05$, we apply the Bonferroni correction ($\alpha_{\text{adj}} = 0.05 / 5 = 0.01$):
* **NORM (LightGBM):** Under the standard uncorrected $\alpha = 0.05$, the NORM gender gap in LightGBM is non-significant ($p = 0.1548 > 0.05$), meaning it is robust across sexes even without adjustment. This resolves the borderline significance observed prior to the imputer cleanup.
* **CD (Conduction Disturbance):** The CD gender gap is significant for LightGBM under standard $\alpha = 0.05$ ($p = 0.0444$) and borderline for the CNN ($p = 0.0170$). Under the strict Bonferroni-corrected threshold ($\alpha_{\text{adj}} = 0.01$), these gaps are defused as non-significant, suggesting they may represent minor statistical fluctuations, though their consistency across both architectures warrants monitoring.

### C. Honest Performance Gaps & Demographic Monitoring
Rather than claiming universal fairness across cohorts, the audit flags two clear performance patterns that must be highlighted for ongoing clinical monitoring:
1. **Conduction Disturbance (CD) Sex Gap:** Both models show a consistent performance gap in CD favoring female patients (obs gap of -0.0257 for CNN, $p = 0.0170$; and -0.0319 for LightGBM, $p = 0.0444$).
2. **Myocardial Infarction (MI) Age Degradation:** Both models exhibit clinically meaningful age-related performance degradation on MI. CNN performance drops from 0.9361 (young) to 0.8533 (elderly), while LightGBM drops from 0.9411 (young) to 0.7843 (elderly). This drop is typical in cardiology literature due to the higher prevalence of confounding comorbidities and silent/atypical ischemic presentation in geriatric patients.

### D. Age Band Robustness finding (HYP Class)
The audit highlighted a significant finding in Section 21 of the methodology guide:
* **Raw Signal CNN:** Exhibits a large performance drop in Senior cohorts on the Hypertrophy (HYP) class (ROC-AUC drops from **0.8629** in young to **0.7329** in seniors, gap of **0.1300**).
* **LightGBM (Clinical Features):** Maintains high robustness across all age bands (ROC-AUC ranges from **0.8663** to **0.8948**, max gap of only **0.0285**).
* **Clinical Rationale:** Ventricular geometry, baseline voltage heights, and heart muscle stiffness change continuously with age. The 1D ResNet CNN, lacking sample volume (only 240 positive cases of HYP), cannot extract invariant morphology across age bands. LightGBM, utilizing Sokolow-Lyon and Cornell voltage priors, bypasses this raw waveform variance, showing that clinical priors insulate models against demographic sparseness.

---

## 7. Operational Checklist & Recommendations

| Parameter/Area | Current Setup | Audit Verdict | Recommended Action |
|---|---|---|---|
| **Patient Leakage** | Group by `patient_id` | **✅ PASS** | Maintain strict single-record-per-patient limit. |
| **Imputer Location** | Per-fold partition | **✅ PASS** | SimpleImputer fit on training fold only, resolving contamination. |
| **Z-Normalization** | Raw CNN only | **✅ PASS** | Do not normalize signals prior to feature extraction. |
| **Confounder Check** | Single source (PTB-XL) | **✅ PASS** | Continue using PTB-XL-only 1D data. |
| **Workflow variables**| Excluded | **✅ PASS** | Keep workflow flags out of clinical models. |
| **Heart Axis feature**| Excluded | **✅ PASS** | Keep heart axis in secondary/exploratory tier only. |
| **Bootstrap Size** | $B=500$ (gaps) / $B=1000$ (CIs) | **✅ PASS** | Standard statistical resolution is satisfied. |

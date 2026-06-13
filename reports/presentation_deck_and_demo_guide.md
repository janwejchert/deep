# Presentation Slide Deck & Live Demo Rehearsal Guide

This document provides a comprehensive, professional slide-by-slide outline, script, and interactive demo rehearsal walkthrough for the **Multi-Heartbreaker ECG Classification Project**. 

Use this guide to build your visual slides and practice the live presentation. It connects the deep technical science of the project to the medical business case, presentation delivery, and live demo segments (which comprise 55% of your final grading rubric).

---

## Part 1: Slide-by-Slide Deck Outline & Script

### Slide 1: Title Slide
* **Slide Title:** Multi-Heartbreaker: Raw 1D ECG Neural Diagnosis and Methodological Integrity
* **Subtitle:** Moving from Cheating 2D Computer Vision to Leakage-Free 1D Physiological Screening
* **Visual Concept:** Minimalist dark mode layout. A clean, single red ECG waveform trace on a dark grid backdrop. Accent colors: Crimson Red (#E53935) and Slate Blue.
* **Presenter Script:**
  > *"Good morning, everyone. Today, my partner and I are excited to present the Multi-Heartbreaker ECG Diagnostic Suite. Our project represents a rigorous journey in clinical machine learning—moving from an early 2D computer vision pipeline that achieved a 'too-good-to-be-true' 98% accuracy by accidentally learning dataset artifacts, to a methodologically sound, raw 1D signal classification system. We will show you how we audited our models for data leakage, resolved dataset confounds, scaled our cohort to 3,878 patients, and integrated expert clinical priors to solve the rare-class bottleneck."*

---

### Slide 2: Clinical Motivation & The Triage Bottleneck
* **Slide Title:** The Clinical Problem: Triage Under Shortage
* **Key Bullet Points:**
  * Cardiovascular Disease (CVD) is the leading cause of global mortality (17.9M deaths/year).
  * 12-lead ECG is the gold-standard bedside diagnostic, but interpretation is a major clinical bottleneck.
  * Delayed readings in emergency departments lead to critical diagnostic windows closing.
  * **Business Case:** An automated triage classifier can scan incoming telemetry at the hardware level, flagging high-risk abnormalities to route them to specialists first.
* **Visual Concept:** A split-screen layout. Left: A graphic representing emergency department backlog. Right: A flowchart showing an automated triage queue ranking patients by abnormality probability.
* **Presenter Script:**
  > *"Cardiovascular disease remains the leading cause of mortality worldwide. While the 12-lead ECG is an inexpensive, ubiquitous bedside test, reading them is a major clinical bottleneck. In a typical hospital, thousands of ECGs are printed daily. A patient presenting with silent myocardial ischemia might wait hours in a stack of papers before a cardiologist reviews their tracing. Our goal is not to replace the doctor, but to automate triage. By ranking incoming telemetry by abnormality probability, we can ensure that high-risk patients are seen in minutes, saving lives."*

---

### Slide 3: Phase 1 — The Deprecated 2D Computer Vision Pipeline
* **Slide Title:** Phase 1: 2D Image-Based ECG Classification
* **Key Bullet Points:**
  * Inputs: 2D rendered images of ECG paper strips.
  * Architecture: DenseNet-121 pretrained on ImageNet.
  * Initial Performance: **97.6% ROC-AUC** on balanced splits.
  * **The Fatal Flaw:** The model was not learning cardiac physics; it was cheating.
* **Visual Concept:** Examples of the 2D rendered ECG strips. Highlights indicating differences in grid lines, fonts, and borders. A red warning banner over the 97.6% AUC score.
* **Presenter Script:**
  > *"We began by replicating common academic approaches: rendering digital ECG signals onto 2D paper strip images and training a DenseNet-121 CNN. The results were immediate and spectacular—97.6% ROC-AUC. However, as clinical engineers, we knew that human electrocardiography is highly complex. Such immediate high accuracy was suspicious. We launched a deep methodological audit to find out if the neural network was actually reading cardiac morphology, or simply identifying visual shortcuts."*

---

### Slide 4: The Audit: Unmasking the Perfect Source-Label Confound
* **Slide Title:** The Audit: How the Model Cheated
* **Key Bullet Points:**
  * **Data Leakage:** Perceptual hashing (pHash) revealed **65.0%** of test images had near-duplicates in training due to random record-level splits instead of patient-level splits.
  * **Source-Label Confound:** 100% of Normal images came from the Latidos database; 100% of Abnormal images came from the PTB-XL database.
  * **Cheating Mechanism:** CNN classified image resolution, border margins, grid colors, and font styles.
  * **Remediation:** Complete deprecation of the 2D image pipeline.
* **Visual Concept:** A Grad-CAM heatmap showing the CNN focusing on the printed text header and border lines rather than the actual QRS complexes.
* **Presenter Script:**
  > *"Our audit unmasked a perfect source-label confound. Every single 'Normal' image had been rendered from the Latidos database, while every 'Abnormal' image came from PTB-XL. Because database source and clinical label were perfectly aligned, the model had no reason to learn medicine. It merely classified differences in margins, background tint, and rendering fonts. Grad-CAM confirmed this, showing the model's attention focused on text headers and borders. Furthermore, random splitting allowed different records from the same patient to cross the train-test boundary. We immediately deprecated the entire 2D pipeline. High accuracy means nothing if it is a statistical illusion."*

---

### Slide 5: The Pivot: 1D Raw-Signal Deep Learning
* **Slide Title:** Phase 2: 1D Raw-Signal ResNet Pipeline
* **Key Bullet Points:**
  * **Data Source:** Raw 12-lead digital telemetry from PTB-XL only.
  * **Leakage Control:** Grouping by `patient_id` ensures 0% patient overlap across folds.
  * **Standardization Preprocessing:** 
    * 4th-order Butterworth filter (0.5–40 Hz) removes baseline drift and muscle artifact.
    * Lead-wise Z-score standardization per patient prevents voltage scale bias.
  * **Architecture:** 1D ResNet with Residual Blocks + Dropout + Global Average Pooling.
* **Visual Concept:** A clean architectural flowchart showing: Raw Waveform (1000x12) ➔ Butterworth Bandpass ➔ Z-score Standardization ➔ 1D ResNet ➔ Sigmoid.
* **Presenter Script:**
  > *"To build a clinically honest classifier, we pivoted to a raw 1D signal pipeline using only the PTB-XL database. We implemented strict patient-level GroupKFold cross-validation, guaranteeing zero patient overlap. The raw waveforms are preprocessed using a Butterworth bandpass filter to eliminate baseline drift and standardized lead-by-lead. This ensures the model is scale-invariant and immune to visual artifacts. We trained a customized 1D ResNet operating directly on these 12-lead digital signals, establishing a valid, robust baseline."*

---

### Slide 6: Multi-Label Scaling & Threshold Calibration
* **Slide Title:** Phase 3: Scaling to Multi-Label Differential Diagnosis
* **Key Bullet Points:**
  * Scaled cohort to **3,878 unique patient records** (progressing from 2,000-record binary model).
  * Target: 5 co-occurring superclasses: Normal (NORM), Infarction (MI), ST/T-Change (STTC), Conduction (CD), and Hypertrophy (HYP).
  * **Independent Sigmoid Outputs** allow detection of multiple pathologies.
  * **Platt Calibration & Youden's J Optimization** set independent thresholds for each class to handle clinical prevalence variations.
* **Visual Concept:** A bar chart showing the class prevalence (NORM: 56.5%, MI: 11.2%, STTC: 17.7%, CD: 18.2%, HYP: 6.2%) to illustrate class imbalance.
* **Presenter Script:**
  > *"To move beyond a simple normal/abnormal toggle, we scaled our dataset to 3,878 patients and transitioned to a multi-label classification system. Because patients often present with multiple overlapping pathologies—such as an old myocardial infarction alongside acute ST/T-wave changes—we used independent sigmoid outputs. Due to the high prevalence imbalance, applying a uniform 0.5 threshold is clinically incorrect. Instead, we applied Platt calibration and swept thresholds out-of-fold to find the optimal operating point for each disease using Youden's J statistic."*

---

### Slide 7: ResNet vs. LightGBM — Bypassing the Rare-Class Bottleneck
* **Slide Title:** Deep Learning vs. Clinical Prior Feature Engineering
* **Key Bullet Points:**
  * **The Problem:** The 1D CNN struggled on the rarest class, Hypertrophy (HYP, 240 positive cases), yielding a PR-AUC of **0.2754**.
  * **The Fix:** LightGBM trained on **59 expert cardiology features** (voltage, intervals, ST deviations).
  * **The Result:** LightGBM boosted HYP ROC-AUC to **0.8781** (+0.0822) and PR-AUC to **0.4772** (+0.2018).
  * **Conclusion:** Cardiology priors insulate models against rare-class data sparseness.
* **Visual Concept:** A head-to-head metrics table showing the CNN winning on common classes (NORM, MI, CD) but LightGBM dominating on the rare HYP class.
* **Presenter Script:**
  > *"When we evaluated our deep learning model, we encountered a classic machine learning bottleneck. On the rarest class, Hypertrophy, which had only 240 positive examples, the CNN struggled, achieving a low PR-AUC of 0.2754. Deep neural networks are data-hungry and fail to learn complex features from rare samples. To solve this, we built a second model: a LightGBM classifier trained on 59 cardiology-standard features, such as the Sokolow-Lyon index and Cornell voltage. Because these clinical formulas encode expert priors, LightGBM bypassed the sample size limitation, boosting Hypertrophy ROC-AUC to 0.8781 and PR-AUC to 0.4772. This proves that expert feature engineering remains crucial in low-data medical regimes."*

---

### Slide 8: Demographic Fairness & Statistical Rigor
* **Slide Title:** Subgroup Fairness & Statistical Auditing
* **Key Bullet Points:**
  * **Bootstrap Gap Testing:** Audited sex and age bands to identify clinical biases.
  * **Multiple-Comparisons Correction:** Applied Bonferroni correction ($\alpha_{\text{adj}} = 0.01$).
  * **Key Finding 1:** The CD sex gap (favoring females) is consistent across models.
  * **Key Finding 2:** MI classification degrades in elderly patients due to silent comorbidities.
  * **HYP Age Generalization:** CNN's performance dropped sharply by **0.1300** in seniors; LightGBM maintained a tiny **0.0285** gap.
* **Visual Concept:** Subgroup ROC-AUC comparison bar charts showing gender gaps and age-band performance lines.
* **Presenter Script:**
  > *"To ensure clinical safety, we audited our models across patient sex and age subgroups using bootstrap gap significance testing. Rather than declaring our model universally 'fair,' we want to be transparent about two clinical findings. First, both models show a significant Conduction Disturbance gap favoring female patients. Second, both models exhibit performance degradation on Myocardial Infarction in elderly patients—a pattern well-documented in medical literature due to silent ischemic presentations in geriatric cohorts. Finally, on Hypertrophy, the CNN's performance dropped by 0.13 in senior cohorts, whereas the LightGBM clinical-feature model maintained high robustness with a maximum gap of only 0.028, proving again that clinical voltage priors insulate classifiers against age-related waveform drift."*

---

### Slide 9: The Product: Automated Triage Dashboard
* **Slide Title:** Interactive Triage & Diagnostic MVP
* **Key Bullet Points:**
  * Streamlit application serving models end-to-end.
  * Preprocesses raw 12-lead WFDB binary telemetry in real-time.
  * Visualizes raw vs. standardized signals for clinical review.
  * Displays triage verdicts and multi-label probability charts with Youden's J cutoffs.
  * **Live Demo Alert:** Showcasing the interface.
* **Visual Concept:** Screenshots of the Streamlit dashboard app interface, highlighting the waveform plot, the model selector, and prediction outcome cards.
* **Presenter Script:**
  > *"We have packaged our final models into an interactive, clinical-ready dashboard. This MVP operates on raw binary ECG telemetry. It displays both the raw signals and the standardized outputs so clinicians can verify the tracings. It lets users run automated triage diagnostics, showing clear flagging cards for abnormal readings, and provides differential diagnostic probability bars with Youden's J threshold lines. Let us transition to the live demo to show the system running in real-time."*

---

### Slide 10: Business Case & Future Roadmap
* **Slide Title:** Business Impact & The Path to Clinical Deployment
* **Key Bullet Points:**
  * **Triage Value:** Reduces time-to-interpretation for critical anomalies from hours to seconds.
  * **Integrity Guarantee:** Direct 1D execution allows edge-computing directly on ECG cart hardware, ensuring zero dependency on cloud rendering.
  * **Future Roadmap:**
    1. Scale to the full PTB-XL database (~21,837 records) to stabilize the HYP class.
    2. Deploy Brier calibration curves to evaluate probability reliability.
    3. Validate externally on Chapman-Shaoxing or MIMIC-IV-ECG cohorts.
* **Visual Concept:** A timeline arrow detailing: Scaled MVP (N=3.8k) ➔ Full PTB-XL Scale (N=21k) ➔ External Cohort Validation ➔ Hospital Pilot.
* **Presenter Script:**
  > *"From a business perspective, the value is clear. By automating triage, we prevent critical patient delay. Because our model processes raw 1D digital telemetry, it can run as a low-latency edge application directly on bedside ECG cart hardware, keeping patient data secure. Our roadmap to clinical deployment has three stages: first, scaling to the full 21,800 PTB-XL database to resolve the hypertrophy clinical ready status; second, validating probability calibration using Brier scores; and third, performing external validation on independent databases like Chapman-Shaoxing. Thank you, and we are happy to take your questions."*

---

## Part 2: Live Demo Rehearsal Walkthrough

Follow this step-by-step checklist during your demo rehearsal to ensure a flawless, high-scoring live presentation.

### Step 1: Set up the Environment
* [ ] Run the Streamlit server in your terminal:
  ```bash
  streamlit run src/streamlit_dashboard/app.py
  ```
* [ ] Open the dashboard at `http://localhost:8501`.
* [ ] Zoom the browser slightly (e.g., 110%) so the text and plots are clear on the projector screen.

### Step 2: Demonstrate Binary Triage
* [ ] Select **Triage Classifier (Binary CNN)** in the sidebar model selection.
* [ ] Filter by ground-truth pathology: Select **Normal (NORM)**.
* [ ] Choose a patient record from the list.
* [ ] Point out the **ECG Waveform Plots**:
  * Explain the difference: The top plot is the raw amplitude in millivolts, showing baseline drift. The bottom plot is the bandpass-filtered and Z-score standardized signal that the CNN actually sees.
  * Select different leads (e.g., Lead II, V1, V5) to show that the dashboard loads and renders them instantly.
* [ ] Click **Run Neural Network Diagnostic**.
* [ ] Show the green **✅ NORMAL ECG** verdict card. Point out the probability score (which should be low, e.g. 26%).

### Step 3: Demonstrate Multi-Label Diagnosis & The HYP Warning
* [ ] Switch the model mode in the sidebar to **Differential Diagnosis (Multi-Label CNN)**.
* [ ] Filter by ground-truth pathology: Select **Myocardial Infarction (MI)**.
* [ ] Choose a patient record from the list.
* [ ] Click **Run Neural Network Diagnostic**.
* [ ] Point out the **Probability Bar Chart**:
  * Show the blue bars (probabilities) and the red dotted vertical lines (**Youden's J Cutoffs**).
  * Explain that if a bar crosses the red dotted line, it is flagged as positive. Show that the MI bar has crossed the threshold, triggering a red bar color and a `⚠️ POSITIVE` verdict in the list below.
* [ ] **Highlight the HYP Warning:**
  * Scroll down to the yellow warning box.
  * Verbally explain: *"You will notice a warning regarding Left Ventricular Hypertrophy (HYP). Although our LightGBM model achieves strong performance, we want to be transparent: the HYP class has low prevalence. We do not consider it clinically usable yet. A key part of our future roadmap is scaling to the full 21k database to stabilize this class."* (This shows high integrity and methodological maturity, which evaluators love).

### Step 4: Summarize the Business Case
* [ ] Point to the grey card at the bottom right: **Cardiology MVP Business Case**.
* [ ] Reiterate that the application executes at the raw signal level, ensuring it is immune to visual rendering confounds and can run directly on edge hardware.

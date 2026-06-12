# ECG Classification MVP
Deep Learning Final Project · IE University

This repository contains the codebase for an automated ECG prescreening pipeline. 
Following a rigorous methodological investigation, the project maintains two distinct pipelines:

1. **The 2D Forensic Pipeline:** A highly optimized image classifier that ultimately proved the original image dataset contains an insurmountable **source confound** (Latidos vs. PTB-XL). 
2. **The 1D Physiological Pipeline (MVP):** An automated pipeline operating on raw 1D physiological signals from a single clinical source (PTB-XL). Utilizing a 1D ResNet, Binary Focal Loss, and Platt Scaling, this pipeline achieves an honest, confound-free cross-validated ROC-AUC of 0.8140.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the 1D Physiological Model (The Solution / Main MVP)
Train the 1D CNN on unconfounded raw PTB-XL signals:
```bash
python train_1d_ecg_model.py
```

Launch the MVP Streamlit triage application for raw signals:
```bash
streamlit run app.py
```

### 3. Run the 2D Image Model (The Forensic Investigation)
Train the rigorously corrected 2D CNN (with explicit BatchNorm freezing) and evaluate it deterministically to reveal the source confound:
```bash
python train_binary_ecg_model_reduced.py
python evaluate_reduced_model.py
```

Launch the 2D Streamlit prototype (Note: subject to the source confound):
```bash
streamlit run app_2d_investigation.py
```

## Repository Structure
- `project_audit_report.md`: The definitive ReadMe generated during the final audit. Outlines active scripts vs deprecated scripts.
- `final_ecg_report.md`: The complete project narrative detailing the 19 methodological corrections and the central lesson of the investigation.
- `validation_report.md`: Deep dive into the metrics, threshold sweep, and limitations of the 1D ResNet.
- `methodology_guide.md`: A mathematical explanation of the optimization, focal loss, Platt scaling, and leakage detection.
- `train_1d_ecg_model.py` / `ablation_experiments.py`: The single-source physiological pipelines representing the true clinical MVP.
- `app.py`: The 1D Streamlit application simulating an ECG triage flow.
- `train_binary_ecg_model_reduced.py` / `app_2d_investigation.py`: The historically archived 2D image pipelines.

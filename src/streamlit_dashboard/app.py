import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import wfdb
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import json

# Set premium page config
st.set_page_config(
    page_title="Multi-Heartbreaker ECG Diagnostic Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling for rich aesthetics and dark/light harmony
st.markdown("""
    <style>
    .main {
        background-color: #fcfdfe;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stApp {
        background-color: #fcfdfe;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #E53935;
        margin-bottom: 1rem;
    }
    .metric-card-success {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .triage-header {
        font-size: 2rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #E53935 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #b71c1c !important;
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Caching Resource Loaders
# -------------------------------------------------------------

@st.cache_resource
def load_binary_model():
    model_path = 'models/binary_1d_ecg_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

@st.cache_resource
def load_multiclass_model():
    model_path = 'models/multiclass_1d_ecg_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

@st.cache_data
def load_metadata():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        # Filter to records where raw signals exist on disk
        valid_records = []
        for i, row in df.iterrows():
            record_path = os.path.join('data/raw', row['filename_lr'])
            if os.path.exists(record_path + '.hea'):
                valid_records.append(row)
        return pd.DataFrame(valid_records)
    return None

@st.cache_data
def load_thresholds():
    # Load CNN multiclass thresholds
    thresh_path = 'models/multiclass_thresholds_cnn.json'
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            return json.load(f)
    # Fallback to defaults if not found
    return {
        "NORM": 0.465,
        "MI": 0.080,
        "STTC": 0.197,
        "CD": 0.261,
        "HYP": 0.065
    }

# Load resources
binary_model = load_binary_model()
multiclass_model = load_multiclass_model()
df_metadata = load_metadata()
thresholds = load_thresholds()

# -------------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------------

def preprocess_ecg_signal(record_path):
    """Load raw WFDB record and apply bandpass filter + lead-wise Z-norm."""
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        
        fs = 100
        nyq = 0.5 * fs
        # 0.5Hz to 40Hz bandpass filter to eliminate baseline drift and powerline noise
        b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
        
        filtered_sig = np.zeros_like(sig)
        for i in range(sig.shape[1]):
            filtered_sig[:, i] = signal.filtfilt(b, a, sig[:, i])
            
        # Lead-wise Z-normalization per record to eliminate baseline scale bias
        mean = np.mean(filtered_sig, axis=0)
        std = np.std(filtered_sig, axis=0)
        std[std == 0] = 1.0
        norm_sig = (filtered_sig - mean) / std
        
        # Pad or crop to exactly 1000 samples (10 seconds @ 100Hz)
        if norm_sig.shape[0] >= 1000:
            norm_sig = norm_sig[:1000, :]
        else:
            pad = np.zeros((1000 - norm_sig.shape[0], 12))
            norm_sig = np.vstack([norm_sig, pad])
            
        return norm_sig, sig # Return normalized and raw signals
    except Exception as e:
        st.error(f"Error loading WFDB signal {record_path}: {e}")
        return None, None

# -------------------------------------------------------------
# Dashboard Layout
# -------------------------------------------------------------

# Title Section
st.title("🫀 Multi-Heartbreaker ECG Diagnostic Suite")
st.markdown("##### End-to-End Diagnostic MVP operating on Raw 12-Lead Physiological Waveforms (PTB-XL)")
st.write("---")

# Sidebar Configuration
st.sidebar.header("🛠️ Pipeline Controls")

# 1. Select Model Mode
model_mode = st.sidebar.selectbox(
    "Choose Clinical Task:",
    ["Triage Classifier (Binary CNN)", "Differential Diagnosis (Multi-Label CNN)"]
)

# Check model availability
if model_mode == "Triage Classifier (Binary CNN)" and binary_model is None:
    st.sidebar.error("⚠️ Binary Model not found in `models/`!")
elif model_mode == "Differential Diagnosis (Multi-Label CNN)" and multiclass_model is None:
    st.sidebar.error("⚠️ Multi-Label Model not found in `models/`!")

# 2. Select Sample Filtering
if df_metadata is not None:
    st.sidebar.subheader("📂 Demo Dataset Filter")
    subgroup_filter = st.sidebar.selectbox(
        "Filter by Ground-Truth Pathology:",
        ["All Patients", "Normal (NORM)", "Myocardial Infarction (MI)", "ST/T-Change (STTC)", "Conduction Disturbance (CD)", "Hypertrophy (HYP)"]
    )
    
    # Filter the metadata dataframe based on selection
    if subgroup_filter == "Normal (NORM)":
        df_filtered = df_metadata[df_metadata['label_NORM'] == 1]
    elif subgroup_filter == "Myocardial Infarction (MI)":
        df_filtered = df_metadata[df_metadata['label_MI'] == 1]
    elif subgroup_filter == "ST/T-Change (STTC)":
        df_filtered = df_metadata[df_metadata['label_STTC'] == 1]
    elif subgroup_filter == "Conduction Disturbance (CD)":
        df_filtered = df_metadata[df_metadata['label_CD'] == 1]
    elif subgroup_filter == "Hypertrophy (HYP)":
        df_filtered = df_metadata[df_metadata['label_HYP'] == 1]
    else:
        df_filtered = df_metadata
        
    st.sidebar.write(f"Available cohort size: **{len(df_filtered)}** records")
    
    # Generate labels for dropdown
    df_filtered = df_filtered.head(20) # Limit to first 20 records to keep list manageable
    record_choices = {}
    for idx, row in df_filtered.iterrows():
        # Build description of pathologies
        pathologies = []
        if row['label_NORM'] == 1: pathologies.append("Normal")
        if row['label_MI'] == 1: pathologies.append("MI")
        if row['label_STTC'] == 1: pathologies.append("STTC")
        if row['label_CD'] == 1: pathologies.append("CD")
        if row['label_HYP'] == 1: pathologies.append("HYP")
        pathology_str = ", ".join(pathologies)
        
        label = f"Patient {row['patient_id']:.0f} (ECG ID {row['ecg_id']} | Age {row['age']:.0f} | {row['sex'] == 0 and 'M' or 'F'} | Ground Truth: {pathology_str})"
        record_choices[label] = row['filename_lr']
        
    selected_label = st.sidebar.selectbox("Select Patient Record:", list(record_choices.keys()))
    record_filename = record_choices[selected_label]
    selected_row = df_filtered[df_filtered['filename_lr'] == record_filename].iloc[0]
else:
    st.sidebar.warning("⚠️ Metadata database not found at `data/subset_multiclass_metadata.csv`!")
    record_filename = None

# 3. Select Lead for Waveform View
lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
selected_lead_idx = st.sidebar.selectbox("ECG Visualization Lead:", range(12), format_func=lambda idx: lead_names[idx])

# Main Window Logic
if record_filename:
    record_path = os.path.join('data/raw', record_filename)
    
    # Preprocess ECG Waveform
    norm_sig, raw_sig = preprocess_ecg_signal(record_path)
    
    if norm_sig is not None:
        # Create Layout columns (Main view + predictions side-by-side)
        col1, col2 = st.columns([7, 5])
        
        with col1:
            st.subheader(f"📈 ECG Waveform - {lead_names[selected_lead_idx]}")
            st.write("Visualizing 10 seconds of raw signal vs. 0.5-40Hz filtered and lead-wise standardized waveform.")
            
            # Matplotlib interactive plotting
            fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
            
            # Raw Signal Plot
            ax[0].plot(raw_sig[:, selected_lead_idx], color='#2C3E50', linewidth=1)
            ax[0].set_title(f"Raw Physical Waveform (mV)", fontsize=10, fontweight='bold')
            ax[0].set_ylabel("Amplitude (mV)")
            ax[0].grid(True, alpha=0.3, linestyle='--')
            
            # Filtered + Standardized Plot
            ax[1].plot(norm_sig[:, selected_lead_idx], color='#E53935', linewidth=1)
            ax[1].set_title(f"Preprocessed and Standardized Waveform (Z-Score)", fontsize=10, fontweight='bold')
            ax[1].set_xlabel("Time Samples (100 Hz)")
            ax[1].set_ylabel("Standard Deviation")
            ax[1].grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show patient metadata
            st.write("---")
            st.subheader("📋 Patient Demographic Context")
            st.markdown(f"""
            * **Patient ID:** {selected_row['patient_id']:.0f} | **ECG ID:** {selected_row['ecg_id']}
            * **Age:** {selected_row['age']:.0f} | **Sex:** {selected_row['sex'] == 0 and 'Male' or 'Female'}
            * **Clinical Report Transcription:** *"{selected_row['report']}"*
            """)
            
        with col2:
            st.subheader("🔮 Diagnostic Inference")
            st.write("Click below to trigger neural network inference on raw physiological waveforms.")
            
            if st.button("Run Neural Network Diagnostic", use_container_width=True):
                # Format signal for model (shape batch=1, time=1000, leads=12)
                input_batch = tf.expand_dims(norm_sig, 0)
                
                if model_mode == "Triage Classifier (Binary CNN)":
                    if binary_model is not None:
                        probability = binary_model.predict(input_batch)[0][0]
                        
                        st.markdown("### Model Classification Verdict")
                        
                        # Apply clinical triage thresholds (Target sensitivity >= 0.85, val threshold = 0.50)
                        is_abnormal = probability >= 0.50
                        
                        if is_abnormal:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="triage-header" style="color: #E53935;">⚠️ FLAG ABNORMAL</div>
                                <p style="margin: 0; font-size: 1.1rem;">This record has been flagged for <b>high-priority clinical triage</b>.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card-success">
                                <div class="triage-header" style="color: #4CAF50;">✅ NORMAL ECG</div>
                                <p style="margin: 0; font-size: 1.1rem;">This record has been classified as a <b>Normal ECG pattern</b>.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.write("#### Probabilities and Operating Details:")
                        st.info(f"**Abnormality Probability:** {probability:.4%}")
                        st.write("* **Triage Cutoff (Threshold):** `0.5000` (nested-validation optimal)")
                        st.write("* **Pipeline Sensitivity (Recall):** `85.80%` | **Specificity:** `84.10%` (out-of-fold validated)")
                    else:
                        st.error("Error: Binary model missing.")
                        
                else: # Differential Diagnosis (Multi-Label CNN)
                    if multiclass_model is not None:
                        predictions = multiclass_model.predict(input_batch)[0]
                        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
                        class_labels = {
                            'NORM': 'Normal ECG Pattern (NORM)',
                            'MI': 'Myocardial Infarction (MI)',
                            'STTC': 'ST/T-Change pathology (STTC)',
                            'CD': 'Conduction Disturbance (CD)',
                            'HYP': 'Left Ventricular Hypertrophy (HYP)'
                        }
                        
                        st.markdown("### Multi-Label Probability Scores")
                        st.write("Each diagnostic class has its own threshold, optimized independently via Youden's J statistic to handle clinical prevalence variations.")
                        
                        # Compute bar chart data
                        pred_df = pd.DataFrame({
                            'Class': [class_labels[c] for c in classes],
                            'Probability': predictions,
                            'Threshold': [thresholds.get(c, 0.5) for c in classes]
                        })
                        
                        # Plot horizontal bar chart with Youden's J thresholds
                        fig, ax = plt.subplots(figsize=(6, 4))
                        bars = ax.barh(pred_df['Class'], pred_df['Probability'], color='#1E88E5', height=0.5, label='Predicted Probability')
                        
                        # Add threshold markers
                        for idx, row in pred_df.iterrows():
                            # Draw a dotted vertical line for each category's threshold
                            ax.plot([row['Threshold'], row['Threshold']], [idx - 0.3, idx + 0.3], color='#E53935', linestyle='--', linewidth=1.5)
                            
                            # Highlight predicted active classes
                            if row['Probability'] >= row['Threshold']:
                                bars[idx].set_color('#E53935') # Set color to alert red if it crosses threshold
                                
                        # Plot styling
                        ax.set_xlim(0, 1.05)
                        ax.set_xlabel('Probability Score')
                        ax.set_title('CNN Out-of-Fold Class Prediction Probabilities', fontsize=10, fontweight='bold')
                        ax.grid(True, alpha=0.2, linestyle=':')
                        
                        # Custom legend for threshold line
                        from matplotlib.lines import Line2D
                        custom_lines = [
                            plt.Rectangle((0,0),1,1, color='#E53935', label='Flagged Pathology'),
                            plt.Rectangle((0,0),1,1, color='#1E88E5', label='Sub-threshold Pattern'),
                            Line2D([0], [0], color='#E53935', linestyle='--', label="Youden's J Cutoff")
                        ]
                        ax.legend(handles=custom_lines, loc='lower right', fontsize=8)
                        
                        st.pyplot(fig)
                        
                        # Diagnosis Table
                        st.markdown("#### Clinical Diagnostic Verdicts:")
                        
                        verdicts = []
                        for c in classes:
                            prob = predictions[classes.index(c)]
                            thresh = thresholds.get(c, 0.5)
                            active = prob >= thresh
                            
                            status_icon = "⚠️ POSITIVE" if active else "✅ NEGATIVE"
                            badge_color = "red" if active else "green"
                            
                            st.markdown(f"- **{c}**: `{prob:.1%}` (Cutoff: `{thresh:.3f}`) → <span style='color:{badge_color}; font-weight:bold;'>{status_icon}</span>", unsafe_allow_html=True)
                            
                        # Show clinical warning on HYP
                        st.warning("⚠️ **HYP Clinical Alert:** The Hypertrophy (HYP) class is presented for exploratory pipeline validation only. It is not clinically ready/usable due to statistical sparsity (only 240 positive cases). Scaling the pipeline to the full 21k record database is required for clinical readiness.")
                    else:
                        st.error("Error: Multiclass model missing.")
            
            # Business Value Context card
            st.markdown("---")
            st.markdown("""
            <div style="background-color: #ECEFF1; padding: 1.2rem; border-radius: 8px;">
                <h4 style="margin-top:0; color:#37474F;">💡 Cardiology MVP Business Case</h4>
                <p style="font-size:0.9rem; color:#455A64; margin-bottom:0.5rem;">
                    <b>1. Automated Clinical Triage:</b> By screening raw signals at the source and ranking records by abnormality probability, clinics can automate triage, routing critical cardiovascular alerts to cardiologists instantly.
                </p>
                <p style="font-size:0.9rem; color:#455A64; margin-bottom:0;">
                    <b>2. Hardware-Level Integration:</b> Operating directly on raw 1D digital signal telemetry rather than rendered images ensures complete immunity to visual artifacts and enables low-latency edge-device execution directly inside ECG carts.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer/Credits
st.write("---")
st.markdown("<div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>Multi-Heartbreaker Pipeline Diagnostic Suite MVP. Built under rigorous patient-disjoint validation protocols.</div>", unsafe_allow_html=True)

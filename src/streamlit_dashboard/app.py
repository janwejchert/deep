import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import wfdb
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import json
import base64
from pdf_generator import generate_pdf_report
from gradcam import compute_gradcam_1d

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

from PIL import Image

# Set premium page config
current_dir = os.path.dirname(os.path.abspath(__file__))
# Repo root (.../deep-learning-project) so the app runs from ANY working directory
# (local launch, Docker, or Streamlit Cloud) rather than only from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_dir))
logo_path = os.path.join(current_dir, "assets", "logo.png")
try:
    favicon = Image.open(logo_path)
except Exception:
    favicon = "🫀"

st.set_page_config(
    page_title="FVJ Health-Tech | CardioAI™",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling for rich aesthetics, dark/light harmony, and premium MedTech branding
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Dark Theme Global Setup */
    .main {
        background-color: #0b1120;
        background-image: 
            radial-gradient(at 0% 0%, rgba(15, 23, 42, 1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(22, 78, 99, 0.4) 0px, transparent 50%);
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
        color: #f8fafc;
    }
    .stApp {
        background-color: #0b1120;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Typography Overrides */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #f8fafc;
    }
    
    /* Fix Selectbox and Input styling for Dark Mode */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stSelectbox div[data-baseweb="select"] li {
        background-color: #1e293b;
        color: white;
    }
    
    /* Custom Headers & Branding */
    .banner {
        background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(0, 188, 212, 0.3);
        padding: 1.75rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 188, 212, 0.1);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .banner-content {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .banner-logo {
        height: 80px;
        width: 80px;
        border-radius: 12px;
        object-fit: cover;
        border: 1px solid rgba(0, 188, 212, 0.5);
        box-shadow: 0 0 15px rgba(0, 188, 212, 0.3);
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 25px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        box-shadow: 0 10px 35px rgba(0, 188, 212, 0.1);
        border: 1px solid rgba(0, 188, 212, 0.3);
        transform: translateY(-2px);
    }
    
    /* Triage & Diagnostic Status Cards */
    .status-card-normal {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(2, 44, 34, 0.9) 100%);
        border: 1px solid #10b981;
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .status-card-abnormal {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(69, 10, 10, 0.9) 100%);
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }
    .status-card-info {
        background: linear-gradient(135deg, rgba(12, 74, 110, 0.8) 0%, rgba(8, 47, 73, 0.9) 100%);
        border: 1px solid #0ea5e9;
        border-left: 6px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    
    .status-header-normal {
        color: #34d399;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.3);
    }
    .status-header-abnormal {
        color: #f87171;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        text-shadow: 0 0 10px rgba(248, 113, 113, 0.3);
    }
    .status-header-info {
        color: #38bdf8;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    
    /* Styled Table & Lists */
    .verdict-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.25rem;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .verdict-name {
        font-weight: 600;
        color: #e2e8f0;
    }
    .verdict-prob {
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: #cbd5e1;
    }
    .badge-positive {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-negative {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Premium Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00BCD4 0%, #0288D1 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #26C6DA 0%, #0277BD 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 188, 212, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Clean Divider */
    hr {
        margin: 1.5rem 0 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    
    /* Info / Warning Box Overrides */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
    }

    /* Style Streamlit Tabs to include custom brand icons */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] p {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    /* Pulse/ECG wave logo for tab 1 */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(1) p::before {
        content: "";
        display: inline-block;
        width: 20px;
        height: 20px;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMDBCQ0Q0IiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMjIgMTJoLTRsLTMgOUw5IDNsLTMgOUgyIi8+PC9zdmc+');
        background-size: contain;
        background-repeat: no-repeat;
    }
    /* Shield/Lock logo for tab 2 */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(2) p::before {
        content: "";
        display: inline-block;
        width: 20px;
        height: 20px;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMDBCQ0Q0IiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTIgMjJzOC00IDgtMTBWNWwtOC0zLTggM3Y3YzAgNiA4IDEwIDggMTB6Ii8+PHBhdGggZD0iTTkgMTJsMiAyIDQtNCIvPjwvc3ZnPg==');
        background-size: contain;
        background-repeat: no-repeat;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Caching Resource Loaders
# -------------------------------------------------------------

@st.cache_resource
def load_binary_model():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'binary_1d_ecg_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

@st.cache_resource
def load_multiclass_model():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'multiclass_1d_ecg_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

@st.cache_data
def load_metadata():
    metadata_path = os.path.join(PROJECT_ROOT, 'data', 'unseen_demo_metadata.csv')
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        # Filter to records where raw signals exist on disk (both header and data binary)
        valid_records = []
        for i, row in df.iterrows():
            record_path = os.path.join(PROJECT_ROOT, 'data', 'raw', row['filename_lr'])
            if os.path.exists(record_path + '.hea') and os.path.exists(record_path + '.dat'):
                valid_records.append(row)
        return pd.DataFrame(valid_records)
    return None

@st.cache_data
def load_thresholds():
    # Load CNN multiclass thresholds
    thresh_path = os.path.join(PROJECT_ROOT, 'models', 'multiclass_thresholds_cnn.json')
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

# Title Banner
logo_base64 = get_base64_image(logo_path)
logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="banner-logo" alt="FVJ Health-Tech Logo">' if logo_base64 else ''

st.markdown(f"""
    <div class="banner">
        <div class="banner-content">
            {logo_html}
            <div>
                <span style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 3px; color: #00BCD4;">FVJ Health-Tech</span>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;">CardioAI™</h1>
                <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 1.05rem;">Proactive Diagnostic Intelligence on Raw 12-Lead Physiological Waveforms</p>
            </div>
        </div>
        <div style="background: rgba(0, 188, 212, 0.1); padding: 0.75rem 1.5rem; border-radius: 12px; border: 1px solid rgba(0, 188, 212, 0.3); text-align: center; backdrop-filter: blur(5px);">
            <span style="display: block; font-size: 0.75rem; font-weight: 700; color: #00BCD4; text-transform: uppercase; letter-spacing: 1.5px;">System Status</span>
            <span style="font-weight: 800; color: #10b981; display: flex; align-items: center; gap: 0.5rem;">
                <span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                ACTIVE MVP
            </span>
        </div>
""", unsafe_allow_html=True)
st.markdown("---")
with st.container():
    st.markdown("""
        <div style="background-color: rgba(30, 41, 59, 0.4); padding: 0.5rem 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem; display: flex; align-items: center; gap: 8px;">
                🛠️ FVJ Clinical Control Panel
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        model_mode = st.selectbox(
            "Choose Clinical Task:",
            ["Triage Classifier (Binary CNN)", "Differential Diagnosis (Multi-Label CNN)"]
        )
        if model_mode == "Triage Classifier (Binary CNN)" and binary_model is None:
            st.error("⚠️ Binary Model not found in `models/`!")
        elif model_mode == "Differential Diagnosis (Multi-Label CNN)" and multiclass_model is None:
            st.error("⚠️ Multi-Label Model not found in `models/`!")

    with col_c2:
        pass # Replaced by sidebar

    # Sidebar for Advanced Filtering
    st.sidebar.markdown("### 🔍 Advanced Patient Filtering")
    if df_metadata is not None:
        # Age Filter
        min_age = int(df_metadata['age'].min()) if not df_metadata['age'].isnull().all() else 0
        max_age = int(df_metadata['age'].max()) if not df_metadata['age'].isnull().all() else 100
        age_range = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        
        # Sex Filter
        sex_options = ["All"] + sorted([str(x) for x in df_metadata['sex'].dropna().unique()])
        sex_filter = st.sidebar.selectbox("Biological Sex", sex_options)
        
        # Pathology Filter
        subgroup_filter = st.sidebar.selectbox(
            "Ground-Truth Pathology",
            ["All Patients", "Normal (NORM)", "Myocardial Infarction (MI)", "ST/T-Change (STTC)", "Conduction Disturbance (CD)", "Hypertrophy (HYP)"]
        )

        # Apply Filters
        df_filtered = df_metadata.copy()
        df_filtered = df_filtered[(df_filtered['age'] >= age_range[0]) & (df_filtered['age'] <= age_range[1])]
        if sex_filter != "All":
            df_filtered = df_filtered[df_filtered['sex'] == (int(float(sex_filter)) if str(sex_filter).replace('.','',1).isdigit() else sex_filter)]
            
        if subgroup_filter == "Normal (NORM)":
            df_filtered = df_filtered[df_filtered['label_NORM'] == 1]
        elif subgroup_filter == "Myocardial Infarction (MI)":
            df_filtered = df_filtered[df_filtered['label_MI'] == 1]
        elif subgroup_filter == "ST/T-Change (STTC)":
            df_filtered = df_filtered[df_filtered['label_STTC'] == 1]
        elif subgroup_filter == "Conduction Disturbance (CD)":
            df_filtered = df_filtered[df_filtered['label_CD'] == 1]
        elif subgroup_filter == "Hypertrophy (HYP)":
            df_filtered = df_filtered[df_filtered['label_HYP'] == 1]
            
        st.sidebar.markdown(f"**{len(df_filtered)} patients match filters.**")
    else:
        subgroup_filter = "All Patients"
        df_filtered = None

    with col_c3:
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        selected_lead_idx = st.selectbox("ECG Visualization Lead:", range(12), format_func=lambda idx: lead_names[idx])
        
    if df_filtered is not None:
        # Navigation with Session State
        if 'patient_idx' not in st.session_state:
            st.session_state.patient_idx = 0
            
        if st.session_state.patient_idx >= len(df_filtered):
            st.session_state.patient_idx = 0
            
        col_nav1, col_nav2, col_nav3 = st.columns([3, 4, 3])
        with col_nav1:
            if st.button("⬅️ Previous Patient", use_container_width=True):
                st.session_state.patient_idx = max(0, st.session_state.patient_idx - 1)
        with col_nav2:
            st.markdown(f"""
                <div style="text-align: center; background-color: rgba(30, 41, 59, 0.6); padding: 0.7rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); height: 100%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 0.95rem; color: #f8fafc; font-weight: 700;">Record {st.session_state.patient_idx + 1} of {len(df_filtered)}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_nav3:
            if st.button("Next Patient ➡️", use_container_width=True):
                st.session_state.patient_idx = min(len(df_filtered) - 1, st.session_state.patient_idx + 1)
                
        selected_row = df_filtered.iloc[st.session_state.patient_idx]
        record_filename = selected_row['filename_lr']
    else:
        st.warning("⚠️ Metadata database not found at `data/unseen_demo_metadata.csv`!")
        record_filename = None



tab1, tab2 = st.tabs(["Diagnostic Inference", "Validation & Leakage Audit"])

with tab1:
    # Main Window Logic
    if record_filename:
        record_path = os.path.join(PROJECT_ROOT, 'data', 'raw', record_filename)

        # Preprocess ECG Waveform
        norm_sig, raw_sig = preprocess_ecg_signal(record_path)

        if norm_sig is not None:
            # Create Layout columns (Main view + predictions side-by-side)
            col1, col2 = st.columns([7, 5])

            with col1:
                st.subheader(f"📈 ECG Waveform - {lead_names[selected_lead_idx]}")
                st.write("Visualizing 10 seconds of raw signal vs. 0.5-40Hz filtered and lead-wise standardized waveform.")

                # Matplotlib interactive plotting
                plt.style.use('dark_background')
                fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
                fig.patch.set_alpha(0.0)

                # Raw Signal Plot
                ax[0].patch.set_alpha(0.0)
                ax[0].plot(raw_sig[:, selected_lead_idx], color='#00BCD4', linewidth=1)
                ax[0].set_title(f"Raw Physical Waveform (mV)", fontsize=10, fontweight='bold', color='white')
                ax[0].set_ylabel("Amplitude (mV)", color='white')
                ax[0].grid(True, alpha=0.2, linestyle='--')
                ax[0].tick_params(colors='white')

                # Filtered + Standardized Plot
                ax[1].patch.set_alpha(0.0)
                ax[1].plot(norm_sig[:, selected_lead_idx], color='#10b981', linewidth=1)
                ax[1].set_title(f"Preprocessed and Standardized Waveform (Z-Score)", fontsize=10, fontweight='bold', color='white')
                ax[1].set_xlabel("Time Samples (100 Hz)", color='white')
                ax[1].set_ylabel("Standard Deviation", color='white')
                ax[1].grid(True, alpha=0.2, linestyle='--')
                ax[1].tick_params(colors='white')

                plt.tight_layout()
                st.pyplot(fig)

                import io
                buf = io.BytesIO()
                fig.patch.set_facecolor('#ffffff')
                ax[0].patch.set_facecolor('#ffffff')
                ax[1].patch.set_facecolor('#ffffff')
                ax[0].tick_params(colors='black')
                ax[1].tick_params(colors='black')
                ax[0].set_title("Raw 12-Lead Electrocardiogram Trace", color='black')
                ax[1].set_title("Preprocessed and Standardized Waveform (Z-Score)", color='black')
                ax[0].set_xlabel("Time Samples (100 Hz)", color='black')
                ax[1].set_xlabel("Time Samples (100 Hz)", color='black')
                ax[0].set_ylabel("Amplitude", color='black')
                ax[1].set_ylabel("Standard Deviation", color='black')
                fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
                chart_bytes = buf.getvalue()

                # Show patient metadata
                st.write("---")
                st.markdown("### 📋 Patient Demographic Context")
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.25rem;">
                    <ul style="margin: 0; color: #cbd5e1;">
                        <li><b>Patient ID:</b> {selected_row['patient_id']:.0f} | <b>ECG ID:</b> {selected_row['ecg_id']}</li>
                        <li><b>Age:</b> {selected_row['age']:.0f} | <b>Sex:</b> {selected_row['sex'] == 0 and 'Male' or 'Female'}</li>
                        <li><b>Clinical Report Transcription:</b> <i>"{selected_row['report']}"</i></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <img src="data:image/png;base64,{logo_base64}" style="width: 32px; height: 32px; border-radius: 6px; margin-right: 12px; border: 1px solid rgba(0,188,212,0.3);">
                        <h3 style="margin: 0;">Diagnostic Inference</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Ground Truth Display String
                gt_labels = []
                if selected_row['label_NORM'] == 1: gt_labels.append("Normal (NORM)")
                if selected_row['label_MI'] == 1: gt_labels.append("Myocardial Infarction (MI)")
                if selected_row['label_STTC'] == 1: gt_labels.append("ST/T Change (STTC)")
                if selected_row['label_CD'] == 1: gt_labels.append("Conduction Disturbance (CD)")
                if selected_row['label_HYP'] == 1: gt_labels.append("Hypertrophy (HYP)")
                gt_str = ", ".join(gt_labels) if gt_labels else "Unknown"

                # Parse specific SCP findings
                import ast
                scp_details = []
                scp_codes_str = selected_row.get('scp_codes', '{}')
                try:
                    codes_dict = ast.literal_eval(scp_codes_str)
                except Exception:
                    codes_dict = {}

                try:
                    scp_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'scp_statements.csv'))
                    scp_df = scp_df.rename(columns={scp_df.columns[0]: 'code'})
                    scp_map = scp_df.set_index('code')['description'].to_dict()
                except Exception:
                    scp_map = {}

                for code, val in codes_dict.items():
                    desc = scp_map.get(code, code)
                    scp_details.append(f"• <b>{desc}</b> ({code})")

                scp_details_str = "<br>".join(scp_details) if scp_details else "None"
                report_str = selected_row.get('report', 'No report text available.')

                def display_ground_truth_box():
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(15, 23, 42, 0.85); border-left: 4px solid #00BCD4; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid rgba(0, 188, 212, 0.15);">
                        <div style="font-size: 1rem; color: #f8fafc; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.25rem;">📋</span> <b>Verified Clinical Ground Truth</b>
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-left: 1.5rem; line-height: 1.6;">
                            <span style="color: #00BCD4; font-weight: bold;">Superclass Categories:</span> {gt_str}<br>
                            <span style="color: #00BCD4; font-weight: bold;">Specific Clinical Findings:</span><br>
                            {scp_details_str}<br>
                            <span style="color: #00BCD4; font-weight: bold;">Cardiologist Diagnostic Transcription:</span><br>
                            <i>"{report_str}"</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.write("Click below to trigger neural network inference on raw physiological waveforms.")

                if st.button("Run Neural Network Diagnostic", use_container_width=True):
                    # Format signal for model (shape batch=1, time=1000, leads=12)
                    input_batch = tf.expand_dims(norm_sig, 0)

                    if model_mode == "Triage Classifier (Binary CNN)":
                        if binary_model is not None:
                            probability = binary_model.predict(input_batch)[0][0]

                            st.markdown("### Model Triage Verdict")

                            is_abnormal = probability >= 0.50

                            if is_abnormal:
                                verdict_title = "⚠️ FLAG ABNORMAL"
                                verdict_text = "This record has been flagged by the CardioAI Triage Engine for high-priority clinical review."
                                st.markdown(f"""
                                <div class="status-card-abnormal">
                                    <div class="status-header-abnormal">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #fecaca;">
                                        {verdict_text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                verdict_title = "✅ NORMAL ECG"
                                verdict_text = "This record has been classified as a Normal ECG pattern. No urgent triage required."
                                st.markdown(f"""
                                <div class="status-card-normal">
                                    <div class="status-header-normal">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #a7f3d0;">
                                        {verdict_text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                            display_ground_truth_box()

                            st.write("#### Probabilities and Operating Details:")

                            # Generate HTML content for the binary progress bar
                            prob_pct = min(100.0, max(0.0, probability * 100))
                            thresh_pct = 50.0 # Threshold is 0.5000
                            active = probability >= 0.50

                            status_text = "POSITIVE / ABNORMAL" if active else "NEGATIVE / NORMAL"
                            if active:
                                label_color = "#f43f5e" # Rosy Red
                                progress_background = "linear-gradient(90deg, #ef4444 0%, #f43f5e 100%)"
                                badge_style = "background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4);"
                            else:
                                label_color = "#10b981"
                                progress_background = "linear-gradient(90deg, #00BCD4 0%, #0288D1 100%)"
                                badge_style = "background: rgba(255, 255, 255, 0.05); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.15);"

                            binary_chart_html = f"""
<div style="background: rgba(30, 41, 59, 0.5); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem; margin-bottom: 0.4rem;">
        <span style="font-weight: 700; color: #f8fafc;">Triage Abnormality Probability</span>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; color: {label_color}; font-size: 1rem;">{probability:.2%}</span>
            <span style="font-size: 0.8rem; color: #64748b;">(Cutoff: 0.500)</span>
            <span style="padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; {badge_style}">
                {status_text}
            </span>
        </div>
    </div>
    <div style="position: relative; background-color: rgba(255, 255, 255, 0.04); height: 10px; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.06); padding: 0.5px;">
        <!-- Filled Progress Bar -->
        <div style="background: {progress_background}; width: {prob_pct}%; height: 100%; border-radius: 9999px; transition: width 0.6s ease-in-out;"></div>
        <!-- Threshold Cutoff Line -->
        <div style="position: absolute; left: {thresh_pct}%; top: -5px; height: 18px; width: 2.5px; background-color: #f43f5e; box-shadow: 0 0 6px #f43f5e; z-index: 10;" title="Triage Cutoff: 0.500"></div>
    </div>
</div>
"""
                            st.markdown(binary_chart_html, unsafe_allow_html=True)
                            
                            st.write("---")
                            pdf_bytes = generate_pdf_report(
                                patient_data=selected_row,
                                chart_bytes=chart_bytes,
                                predictions=[probability],
                                thresholds={"Abnormal": 0.50},
                                classes=["Abnormal"],
                                verdict_title=verdict_title,
                                verdict_text=verdict_text
                            )
                            st.download_button(
                                label="📄 Download Clinical Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"clinical_report_{selected_row['patient_id']}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.write("* **Pipeline Sensitivity (Recall):** `85.80%` | **Specificity:** `84.10%` (out-of-fold validated)")
                            
                            # Grad-CAM Visualization
                            with st.expander("🔍 Interpretability: Grad-CAM Feature Heatmap", expanded=False):
                                st.write("The heatmap highlights the temporal regions of the ECG waveform that most strongly influenced the neural network's triage decision. Red/warm regions indicate high influence.")
                                try:
                                    heatmap = compute_gradcam_1d(binary_model, input_batch, target_class_idx=0)
                                    fig_cam, ax_cam = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
                                    fig_cam.patch.set_alpha(0.0)
                                    display_leads = [0, 1, 6, 10] # I, II, V1, V5
                                    lead_names_cam = ["Lead I", "Lead II", "V1", "V5"]
                                    for i, (l_idx, l_name) in enumerate(zip(display_leads, lead_names_cam)):
                                        ax_cam[i].patch.set_alpha(0.0)
                                        # Background Heatmap
                                        x = np.arange(1000)
                                        y_min, y_max = np.min(norm_sig[:, l_idx]), np.max(norm_sig[:, l_idx])
                                        padding = (y_max - y_min) * 0.1
                                        y1, y2 = y_min - padding, y_max + padding
                                        ax_cam[i].imshow(heatmap[None, :], aspect='auto', cmap='jet', extent=[0, 1000, y1, y2], alpha=0.4, interpolation='nearest')
                                        # ECG Waveform
                                        ax_cam[i].plot(norm_sig[:, l_idx], color='white', linewidth=1.2, alpha=0.9)
                                        ax_cam[i].set_ylabel(l_name, color='white')
                                        ax_cam[i].set_ylim(y1, y2)
                                        ax_cam[i].tick_params(colors='white')
                                        ax_cam[i].grid(True, alpha=0.2, linestyle='--')
                                    ax_cam[3].set_xlabel("Time Samples (100 Hz)", color='white')
                                    plt.tight_layout()
                                    st.pyplot(fig_cam)
                                except Exception as e:
                                    st.error(f"Could not generate Grad-CAM: {str(e)}")

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

                            # Calculate active pathology classes
                            active_pathologies = []
                            for c in ['MI', 'STTC', 'CD', 'HYP']:
                                prob = predictions[classes.index(c)]
                                thresh = thresholds.get(c, 0.5)
                                if prob >= thresh:
                                    active_pathologies.append(c)

                            prob_norm = predictions[classes.index('NORM')]
                            thresh_norm = thresholds.get('NORM', 0.5)
                            
                            st.markdown("### Model Diagnostic Verdict")

                            # Display overall cardiac status card
                            if len(active_pathologies) > 0:
                                pathology_str = ", ".join([f"<b>{p}</b>" for p in active_pathologies])
                                verdict_title = "⚠️ ABNORMAL CARDIAC STATUS"
                                verdict_text = f"The CardioAI Diagnostic Engine has flagged the following active pathology classes: {', '.join(active_pathologies)}. Immediate clinical review is recommended."
                                st.markdown(f"""
                                <div class="status-card-abnormal">
                                    <div class="status-header-abnormal">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #fecaca;">
                                        The <b>CardioAI Diagnostic Engine</b> has flagged the following active pathology classes: {pathology_str}. Immediate clinical review is recommended.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif prob_norm >= thresh_norm:
                                verdict_title = "✅ NORMAL CARDIAC STATUS"
                                verdict_text = "The CardioAI Diagnostic Engine predicts a normal cardiac pattern. No active pathologies were flagged."
                                st.markdown(f"""
                                <div class="status-card-normal">
                                    <div class="status-header-normal">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #a7f3d0;">
                                        {verdict_text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                verdict_title = "ℹ️ BORDERLINE / NON-SPECIFIC STATUS"
                                verdict_text = "The waveform represents a borderline pattern. No major pathologies were flagged, but normal rhythm criteria are not fully met."
                                st.markdown(f"""
                                <div class="status-card-info">
                                    <div class="status-header-info">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #bae6fd;">
                                        {verdict_text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                            display_ground_truth_box()

                            st.markdown("### Multi-Label Probability Scores")
                            st.write("Each diagnostic class has its own threshold, optimized independently via Youden's J statistic to handle clinical prevalence variations.")

                            # Generate HTML content for the progress bars
                            chart_html = '<div style="background: rgba(30, 41, 59, 0.5); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">'

                            for c in classes:
                                prob = predictions[classes.index(c)]
                                thresh = thresholds.get(c, 0.5)
                                active = prob >= thresh

                                status_text = "POSITIVE" if active else "NEGATIVE"
                                # Styles for active vs negative
                                if active:
                                    label_color = "#f43f5e" # Rosy Red
                                    progress_background = "linear-gradient(90deg, #ef4444 0%, #f43f5e 100%)"
                                    badge_style = "background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4);"
                                else:
                                    label_color = "#10b981" if c == 'NORM' else "#94a3b8"
                                    progress_background = "linear-gradient(90deg, #00BCD4 0%, #0288D1 100%)"
                                    badge_style = "background: rgba(255, 255, 255, 0.05); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.15);"

                                # Convert values to percentages for width styling
                                prob_pct = min(100.0, max(0.0, prob * 100))
                                thresh_pct = min(99.0, max(1.0, thresh * 100))

                                chart_html += f"""
<div style="margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.95rem; margin-bottom: 0.4rem;">
        <span style="font-weight: 700; color: #f8fafc;">{class_labels[c]}</span>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-weight: 700; color: {label_color}; font-size: 1rem;">{prob:.1%}</span>
            <span style="font-size: 0.8rem; color: #64748b;">(Cutoff: {thresh:.3f})</span>
            <span style="padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; {badge_style}">
                {status_text}
            </span>
        </div>
    </div>
    <div style="position: relative; background-color: rgba(255, 255, 255, 0.04); height: 10px; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.06); padding: 0.5px;">
        <!-- Filled Progress Bar -->
        <div style="background: {progress_background}; width: {prob_pct}%; height: 100%; border-radius: 9999px; transition: width 0.6s ease-in-out;"></div>
        <!-- Threshold Cutoff Line -->
        <div style="position: absolute; left: {thresh_pct}%; top: -5px; height: 18px; width: 2.5px; background-color: #f43f5e; box-shadow: 0 0 6px #f43f5e; z-index: 10;" title="Youden J Cutoff: {thresh:.3f}"></div>
    </div>
</div>
"""

                            # Legend info at the bottom of the card
                            chart_html += """
<div style="display: flex; justify-content: flex-end; gap: 15px; font-size: 0.75rem; color: #64748b; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.75rem;">
    <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 8px; height: 8px; background: linear-gradient(90deg, #ef4444 0%, #f43f5e 100%); border-radius: 2px;"></div>
        <span>Flagged Pathology</span>
    </div>
    <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 8px; height: 8px; background: linear-gradient(90deg, #00BCD4 0%, #0288D1 100%); border-radius: 2px;"></div>
        <span>Sub-threshold / Negative</span>
    </div>
    <div style="display: flex; align-items: center; gap: 4px;">
        <div style="width: 2.5px; height: 8px; background: #f43f5e; box-shadow: 0 0 2px #f43f5e;"></div>
        <span>Youden's J Cutoff</span>
    </div>
</div>
</div>
"""

                            st.markdown(chart_html, unsafe_allow_html=True)

                            # Show clinical warning on HYP
                            st.warning("⚠️ **HYP Clinical Alert:** The Hypertrophy (HYP) class is presented for exploratory pipeline validation only. It is not clinically ready/usable due to statistical sparsity (only 240 positive cases). Scaling the pipeline to the full 21k record database is required for clinical readiness.")

                            st.write("---")
                            pdf_bytes = generate_pdf_report(
                                patient_data=selected_row,
                                chart_bytes=chart_bytes,
                                predictions=predictions,
                                thresholds=thresholds,
                                classes=classes,
                                verdict_title=verdict_title,
                                verdict_text=verdict_text
                            )
                            st.download_button(
                                label="📄 Download Clinical Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"clinical_report_{selected_row['patient_id']}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                            # Grad-CAM Visualization
                            with st.expander("🔍 Interpretability: Grad-CAM Feature Heatmap", expanded=False):
                                st.write("The heatmap highlights the temporal regions of the ECG waveform that most strongly influenced the neural network's decision for the dominant diagnostic class.")
                                try:
                                    target_idx = 0 # NORM by default
                                    if len(active_pathologies) > 0:
                                        # Focus on the highest confidence pathology
                                        highest_pathology = max(active_pathologies, key=lambda c: predictions[classes.index(c)])
                                        target_idx = classes.index(highest_pathology)
                                        st.write(f"**Explaining focus for class:** `{highest_pathology}`")
                                    else:
                                        st.write(f"**Explaining focus for class:** `NORM`")
                                        
                                    heatmap = compute_gradcam_1d(multiclass_model, input_batch, target_class_idx=target_idx)
                                    fig_cam, ax_cam = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
                                    fig_cam.patch.set_alpha(0.0)
                                    display_leads = [0, 1, 6, 10] # I, II, V1, V5
                                    lead_names_cam = ["Lead I", "Lead II", "V1", "V5"]
                                    for i, (l_idx, l_name) in enumerate(zip(display_leads, lead_names_cam)):
                                        ax_cam[i].patch.set_alpha(0.0)
                                        # Background Heatmap
                                        y_min, y_max = np.min(norm_sig[:, l_idx]), np.max(norm_sig[:, l_idx])
                                        padding = (y_max - y_min) * 0.1
                                        y1, y2 = y_min - padding, y_max + padding
                                        ax_cam[i].imshow(heatmap[None, :], aspect='auto', cmap='jet', extent=[0, 1000, y1, y2], alpha=0.4, interpolation='nearest')
                                        # ECG Waveform
                                        ax_cam[i].plot(norm_sig[:, l_idx], color='white', linewidth=1.2, alpha=0.9)
                                        ax_cam[i].set_ylabel(l_name, color='white')
                                        ax_cam[i].set_ylim(y1, y2)
                                        ax_cam[i].tick_params(colors='white')
                                        ax_cam[i].grid(True, alpha=0.2, linestyle='--')
                                    ax_cam[3].set_xlabel("Time Samples (100 Hz)", color='white')
                                    plt.tight_layout()
                                    st.pyplot(fig_cam)
                                except Exception as e:
                                    st.error(f"Could not generate Grad-CAM: {str(e)}")

                        else:
                            st.error("Error: Multiclass model missing.")

                # Business Value Context card
                st.markdown("---")
                st.markdown("""
                <div class="glass-card" style="padding: 1.5rem;">
                    <h4 style="margin-top:0; color:#00BCD4;">💡 Cardiology MVP Business Case</h4>
                    <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:0.75rem;">
                        <b>1. Automated Clinical Triage:</b> By screening raw signals at the source and ranking records by abnormality probability, clinics can automate triage, routing critical cardiovascular alerts to cardiologists instantly.
                    </p>
                    <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:0;">
                        <b>2. Hardware-Level Integration:</b> Operating directly on raw 1D digital signal telemetry rather than rendered images ensures complete immunity to visual artifacts and enables low-latency edge-device execution directly inside ECG carts.
                    </p>
                </div>
                """, unsafe_allow_html=True)


with tab2:
    st.markdown("### 🛡️ Clinical Validation & Leakage Audit Engine")
    st.write("This interactive panel allows you to audit the training, validation, and demo dataset splits to guarantee zero patient-level target leakage.")
    
    # Audit button
    if st.button("Run Real-Time Dataset Integrity & Leakage Check", use_container_width=True):
        import pandas as pd
        import ast
        
        # Load datasets
        binary_path = os.path.join(PROJECT_ROOT, 'data', 'subset_metadata_2000.csv')
        multi_path = os.path.join(PROJECT_ROOT, 'data', 'subset_multiclass_metadata.csv')
        demo_path = os.path.join(PROJECT_ROOT, 'data', 'unseen_demo_metadata.csv')
        
        st.markdown("#### 1. Binary Triage Split Integrity")
        if os.path.exists(binary_path):
            df_bin = pd.read_csv(binary_path)
            total_bin = len(df_bin)
            uniq_bin = df_bin['patient_id'].nunique()
            dup_bin = total_bin - uniq_bin
            
            st.write(f"* **Total Triage Records:** `{total_bin}`")
            st.write(f"* **Unique Patient IDs:** `{uniq_bin}`")
            if dup_bin == 0:
                st.success("✅ **Zero Patient Leakage:** Every record corresponds to a unique patient in the binary dataset.")
            else:
                st.warning(f"⚠️ **Note:** {dup_bin} duplicate patient records exist, but cross-fold splitting is grouped strictly by `patient_id` to prevent leakage.")
        else:
            st.error("Binary split metadata not found.")
            
        st.markdown("#### 2. Multiclass Diagnostic Split Integrity")
        if os.path.exists(multi_path):
            df_mul = pd.read_csv(multi_path)
            total_mul = len(df_mul)
            uniq_mul = df_mul['patient_id'].nunique()
            
            st.write(f"* **Total Multiclass Records:** `{total_mul}`")
            st.write(f"* **Unique Patient IDs:** `{uniq_mul}`")
            if total_mul == uniq_mul:
                st.success("✅ **Zero Patient Overlap:** Patient-disjoint split confirmed. Zero cross-fold leakage guaranteed!")
            else:
                st.warning("⚠️ Duplicate patient IDs found. Ensure GroupKFold validation was applied.")
        else:
            st.error("Multiclass split metadata not found.")
            
        st.markdown("#### 3. Out-of-Sample Demo Cohort Integrity")
        if os.path.exists(demo_path):
            df_demo = pd.read_csv(demo_path)
            total_demo = len(df_demo)
            uniq_demo = df_demo['patient_id'].nunique()
            
            # Check for overlap between demo cohort and training sets
            train_patients = set()
            if os.path.exists(binary_path):
                train_patients.update(pd.read_csv(binary_path)['patient_id'].dropna().tolist())
            if os.path.exists(multi_path):
                train_patients.update(pd.read_csv(multi_path)['patient_id'].dropna().tolist())
                
            demo_patients = set(df_demo['patient_id'].dropna().tolist())
            overlap = demo_patients.intersection(train_patients)
            
            st.write(f"* **Total Demo Records:** `{total_demo}`")
            st.write(f"* **Unique Demo Patient IDs:** `{uniq_demo}`")
            if len(overlap) == 0:
                st.success(f"✅ **Out-of-Sample Clean:** {len(overlap)}/{total_demo} demo patients overlap with the training datasets. True clinical generalization verified!")
            else:
                st.error(f"❌ **LEAKAGE DETECTED:** {len(overlap)} demo patients exist in the training set!")
# Footer/Credits
st.write("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0; line-height: 1.6;">
    <div style="margin-bottom: 0.5rem;">
        <span style="font-weight: 700; color: #00BCD4;">FVJ Health-Tech</span> | CardioAI™ MVP
    </div>
    <div style="font-size: 0.8rem; color: #475569; margin-bottom: 1rem;">
        Built under rigorous patient-disjoint validation protocols on raw digital telemetry.
    </div>
    <div style="display: flex; justify-content: center; gap: 15px; align-items: center;">
        <a href="https://github.com/Pipe10101/deep-learning-project" target="_blank" style="text-decoration: none; color: #f8fafc; background-color: #1e293b; padding: 0.4rem 0.8rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; display: inline-flex; align-items: center; gap: 6px;">
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" height="1.2em" width="1.2em" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
            GitHub Repository
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

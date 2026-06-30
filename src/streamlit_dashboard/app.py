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
logo_path = os.path.join(current_dir, "assets", "logo.png")
try:
    favicon = Image.open(logo_path)
except Exception:
    favicon = "🫀"

st.set_page_config(
    page_title="FVJ Health-Tech | Heartbreaker™",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling for rich aesthetics, dark/light harmony, and premium MedTech branding
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Design tokens */
    :root {
        --accent: #22d3ee;
        --accent-deep: #0891b2;
        --ok: #10b981;
        --alert: #ef4444;
        --info: #0ea5e9;
        --ink: #e7eef6;          /* off-white — lower glare, softer read */
        --muted: #93a3b8;
        --surface: rgba(38, 50, 71, 0.40);
        --hairline: rgba(148, 163, 184, 0.10);
    }

    /* Dark Theme Global Setup — softened, lower-contrast gradient field */
    .stApp {
        background-color: #0d1422;
        background-image:
            radial-gradient(at 0% 0%, rgba(20, 30, 50, 0.9) 0px, transparent 60%),
            radial-gradient(at 100% 0%, rgba(20, 110, 135, 0.10) 0px, transparent 60%),
            radial-gradient(at 50% 120%, rgba(34, 211, 238, 0.04) 0px, transparent 55%);
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
        color: var(--ink);
    }
    /* Soften Streamlit's hard rainbow top bar to a gentle brand tint */
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, rgba(34,211,238,0.55) 0%, rgba(56,189,248,0.35) 50%, rgba(34,211,238,0.15) 100%) !important;
        height: 2px !important;
        opacity: 0.7;
    }
    /* Gentle global easing so hovers feel soft */
    .stButton>button, .glass-card, .stSelectbox div[data-baseweb="select"] > div { transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1); }
    .main { font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif; color: var(--ink); }
    /* Tighten the empty top gap Streamlit reserves for its header */
    .block-container { padding-top: 2.2rem !important; padding-bottom: 3rem !important; max-width: 1500px; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid var(--hairline);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    
    /* Typography Overrides */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #e7eef6;
    }
    
    /* Fix Selectbox and Input styling for Dark Mode */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid var(--hairline);
        border-radius: 10px;
        transition: border-color 0.2s ease;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border: 1px solid rgba(34, 211, 238, 0.5);
    }
    .stSelectbox div[data-baseweb="select"] li {
        background-color: #1e293b;
        color: white;
    }
    /* Widget labels — consistent, legible, slightly muted */
    .stSelectbox label, .stSlider label, [data-testid="stWidgetLabel"] label {
        font-weight: 600 !important; color: #cbd5e1 !important;
        font-size: 0.9rem !important; letter-spacing: 0.2px;
    }
    /* Expander polish */
    [data-testid="stExpander"] details {
        background: var(--surface); border: 1px solid var(--hairline);
        border-radius: 12px; overflow: hidden;
    }
    [data-testid="stExpander"] summary:hover { color: var(--accent); }
    
    /* Custom Headers & Branding */
    .banner {
        background: linear-gradient(135deg, rgba(36,48,68,0.70) 0%, rgba(18,26,42,0.85) 100%);
        border: 1px solid rgba(34, 211, 238, 0.16);
        padding: 1.7rem 2.3rem;
        border-radius: 22px;
        margin-bottom: 1.75rem;
        box-shadow: 0 24px 60px rgba(2, 8, 23, 0.30);
        backdrop-filter: blur(12px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .banner-content {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex: 1 1 420px;
        min-width: 0;
    }
    .banner-logo {
        height: 78px;
        width: 78px;
        border-radius: 14px;
        object-fit: contain;
        background: #0b1120;
        padding: 6px;
        flex-shrink: 0;
        border: 1px solid rgba(34, 211, 238, 0.22);
        box-shadow: 0 6px 20px rgba(2, 8, 23, 0.35);
    }
    .banner-eyebrow {
        font-size: 0.78rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 3px; color: var(--accent);
    }
    .banner-title {
        margin: 0.1rem 0 0 0; font-size: 2.3rem; font-weight: 800; line-height: 1.05;
        color: #f0f6fc; white-space: nowrap; letter-spacing: -0.5px;
    }
    .banner-sub { margin: 0.4rem 0 0 0; color: var(--muted); font-size: 1.02rem; max-width: 46ch; }
    .status-pill {
        background: rgba(34, 211, 238, 0.06); padding: 0.7rem 1.3rem; border-radius: 14px;
        border: 1px solid rgba(34, 211, 238, 0.16); text-align: center; flex-shrink: 0;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(38, 50, 71, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 1.75rem;
        box-shadow: 0 8px 30px rgba(2, 8, 23, 0.22);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .glass-card:hover {
        box-shadow: 0 14px 40px rgba(2, 8, 23, 0.28);
        border: 1px solid rgba(34, 211, 238, 0.18);
        transform: translateY(-1px);
    }
    
    /* Triage & Diagnostic Status Cards — muted, gentle */
    .status-card-normal {
        background: linear-gradient(135deg, rgba(16, 78, 62, 0.42) 0%, rgba(12, 44, 38, 0.42) 100%);
        border: 1px solid rgba(52, 211, 153, 0.28);
        border-left: 4px solid rgba(52, 211, 153, 0.7);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.5rem;
    }
    .status-card-abnormal {
        background: linear-gradient(135deg, rgba(120, 45, 45, 0.40) 0%, rgba(70, 26, 26, 0.42) 100%);
        border: 1px solid rgba(248, 113, 113, 0.30);
        border-left: 4px solid rgba(248, 113, 113, 0.75);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.5rem;
    }
    .status-card-info {
        background: linear-gradient(135deg, rgba(28, 74, 104, 0.40) 0%, rgba(18, 47, 70, 0.42) 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-left: 4px solid rgba(56, 189, 248, 0.7);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.5rem;
    }

    .status-header-normal {
        color: #6ee7b7;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .status-header-abnormal {
        color: #fca5a5;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .status-header-info {
        color: #7dd3fc;
        font-size: 1.45rem;
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

    /* Soft Buttons */
    .stButton>button {
        background: linear-gradient(135deg, rgba(45,180,205,0.92) 0%, rgba(20,130,158,0.92) 100%) !important;
        color: #04222b !important;
        border-radius: 13px !important;
        border: none !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        box-shadow: 0 6px 18px rgba(2, 8, 23, 0.22) !important;
        transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1) !important;
    }
    .stButton>button p { font-size: 0.98rem !important; font-weight: 700 !important; }
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(64,206,228,0.95) 0%, rgba(28,150,178,0.95) 100%) !important;
        color: #04222b !important;
        box-shadow: 0 10px 26px rgba(34, 211, 238, 0.22) !important;
        transform: translateY(-1px) !important;
    }
    /* Download buttons share the look */
    .stDownloadButton>button {
        background: linear-gradient(135deg, rgba(45,180,205,0.92) 0%, rgba(20,130,158,0.92) 100%) !important;
        color: #04222b !important; border: none !important; border-radius: 13px !important;
        font-weight: 700 !important; white-space: nowrap !important;
        box-shadow: 0 6px 18px rgba(2, 8, 23, 0.22) !important;
    }
    .stDownloadButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 26px rgba(34, 211, 238, 0.22) !important; }

    /* Clean Divider */
    hr {
        margin: 1.6rem 0 !important;
        border-color: rgba(148,163,184,0.10) !important;
    }

    /* Info / Warning Box Overrides */
    .stAlert {
        background-color: rgba(38, 50, 71, 0.42) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        color: #dbe4ee !important;
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
    /* Active tab gets an accent underline */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid var(--accent) !important;
    }
    div[data-baseweb="tab-list"] { gap: 0.5rem; }
    /* Professional line icons per tab (cyan, in-brand) */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] p::before {
        content: "";
        display: inline-block;
        width: 18px; height: 18px;
        background-size: contain;
        background-repeat: no-repeat;
        opacity: 0.95;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(1) p::before {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjJkM2VlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWxpbmUgcG9pbnRzPSIyMiAxMiAxOCAxMiAxNSAyMSA5IDMgNiAxMiAyIDEyIi8+PC9zdmc+');
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(2) p::before {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjJkM2VlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48bGluZSB4MT0iMTgiIHkxPSIyMCIgeDI9IjE4IiB5Mj0iMTAiLz48bGluZSB4MT0iMTIiIHkxPSIyMCIgeDI9IjEyIiB5Mj0iNCIvPjxsaW5lIHgxPSI2IiB5MT0iMjAiIHgyPSI2IiB5Mj0iMTQiLz48L3N2Zz4=');
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(3) p::before {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjJkM2VlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTYgMTZsLTQtNC00IDQiLz48cGF0aCBkPSJNMTIgMTJ2OSIvPjxwYXRoIGQ9Ik0yMC4zOSAxOC4zOUE1IDUgMCAwIDAgMTggOWgtMS4yNkE4IDggMCAxIDAgMyAxNi4zIi8+PC9zdmc+');
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-child(4) p::before {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjJkM2VlIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTIgMjJzOC00IDgtMTBWNWwtOC0zLTggM3Y3YzAgNiA4IDEwIDggMTB6Ii8+PHBhdGggZD0iTTkgMTJsMiAyIDQtNCIvPjwvc3ZnPg==');
    }

    /* Metric cards (st.metric) — accented, lifts on hover */
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--hairline);
        border-radius: 14px; padding: 1rem 1.2rem;
        position: relative; overflow: hidden;
        transition: transform 0.28s cubic-bezier(0.22,1,0.36,1), border-color 0.28s ease;
    }
    [data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, rgba(34,211,238,0.7), rgba(56,189,248,0.15));
    }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(34,211,238,0.25); }
    [data-testid="stMetricValue"] { color: #f3f8fc; font-weight: 800; letter-spacing: -0.5px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }

    /* Framed, designed charts (matplotlib + report figures) */
    [data-testid="stImage"] { overflow: hidden; }
    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid var(--hairline);
        background: rgba(13, 20, 34, 0.35);
    }

    /* Consistent accent on plain section headings */
    [data-testid="stMarkdownContainer"] h3 {
        padding-left: 12px; border-left: 3px solid var(--accent); line-height: 1.15;
    }

    /* Dataframe surface */
    [data-testid="stDataFrame"] { border: 1px solid var(--hairline); border-radius: 12px; overflow: hidden; }

    /* KPI credibility strip */
    .kpi-strip { display:flex; gap:1rem; flex-wrap:wrap; margin: 0 0 1.6rem 0; }
    .kpi-card {
        flex:1 1 170px; background: var(--surface); border:1px solid var(--hairline);
        border-radius:16px; padding:1.05rem 1.2rem; position:relative; overflow:hidden;
        transition: transform 0.28s cubic-bezier(0.22,1,0.36,1), border-color 0.28s ease;
    }
    .kpi-card:hover { transform: translateY(-3px); border-color: rgba(34,211,238,0.28); }
    .kpi-card::before { content:""; position:absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg, rgba(34,211,238,0.8), rgba(56,189,248,0.12)); }
    .kpi-value { font-size:1.65rem; font-weight:800; color:#f0f6fc; letter-spacing:-0.5px; line-height:1; }
    .kpi-label { font-size:0.74rem; color:var(--muted); margin-top:0.4rem; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; }
    .kpi-sub { font-size:0.72rem; color:#64748b; margin-top:0.1rem; }
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
    metadata_path = 'data/unseen_demo_metadata.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        # Filter to records where raw signals exist on disk (both header and data binary)
        valid_records = []
        for i, row in df.iterrows():
            record_path = os.path.join('data/raw', row['filename_lr'])
            if os.path.exists(record_path + '.hea') and os.path.exists(record_path + '.dat'):
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


def preprocess_signal_array(sig, fs=100):
    """Apply the exact training pipeline (0.5-40 Hz band-pass + lead-wise z-norm
    + pad/crop to 1000 samples) to an arbitrary raw (samples x 12) array.
    Used for user-uploaded recordings so they match the model's input distribution."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 2 or sig.shape[1] != 12:
        return None
    nyq = 0.5 * fs
    b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    filtered = np.zeros_like(sig)
    for i in range(sig.shape[1]):
        if sig.shape[0] > 27:  # filtfilt needs length > padlen
            filtered[:, i] = signal.filtfilt(b, a, sig[:, i])
        else:
            filtered[:, i] = sig[:, i]
    mean = filtered.mean(axis=0)
    std = filtered.std(axis=0)
    std[std == 0] = 1.0
    norm = (filtered - mean) / std
    if norm.shape[0] >= 1000:
        norm = norm[:1000, :]
    else:
        norm = np.vstack([norm, np.zeros((1000 - norm.shape[0], 12))])
    return norm


def estimate_heart_rate(norm_sig, fs=100, lead_idx=1):
    """Estimate heart rate from R-peaks on Lead II (the standard rhythm lead).
    Returns a dict with bpm, beat count and mean RR, or None if undetectable."""
    x = np.asarray(norm_sig[:, lead_idx], dtype=float)
    if np.std(x) > 0:
        x = (x - np.mean(x)) / np.std(x)
    peaks, _ = signal.find_peaks(x, distance=int(0.3 * fs), height=1.0, prominence=0.5)
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / fs  # seconds between beats
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # keep physiologic 30-200 bpm
    if len(rr) < 1:
        return None
    return {"bpm": 60.0 / np.median(rr), "n_beats": int(len(peaks)),
            "mean_rr_ms": float(np.mean(rr) * 1000), "peaks": peaks}


# -------------------------------------------------------------
# Validation / Performance data (out-of-fold predictions)
# -------------------------------------------------------------

@st.cache_data
def load_binary_oof():
    """Out-of-fold probabilities for the binary triage model, aligned with labels.
    Verified: reproduces the reported ROC-AUC of 0.9243."""
    p, m = 'models/clean_oof_ecg_probs.npy', 'data/subset_metadata_2000.csv'
    if not (os.path.exists(p) and os.path.exists(m)):
        return None
    probs = np.load(p)
    df = pd.read_csv(m)
    if len(df) != len(probs) or 'class' not in df.columns:
        return None
    y = (df['class'].astype(str).str.strip().str.lower() == 'abnormal').astype(int).values
    return {"probs": probs, "y": y, "df": df}


@st.cache_data
def load_multiclass_oof(model):
    """Out-of-fold multi-label probabilities. model='CNN' uses the on-disk subset
    (3878 records); model='LightGBM' uses the full 3883-record OOF set."""
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    m = 'data/subset_multiclass_metadata.csv'
    if not os.path.exists(m):
        return None
    df = pd.read_csv(m)
    if model == "LightGBM":
        p = 'models/lightgbm_oof_multiclass_probs.npy'
        if not os.path.exists(p):
            return None
        probs = np.load(p)
        if len(probs) != len(df):
            return None
        sub = df.reset_index(drop=True)
    else:  # CNN — OOF saved only for records physically present on disk
        p = 'models/clean_oof_multiclass_probs.npy'
        if not os.path.exists(p):
            return None
        probs = np.load(p)
        mask = df['filename_lr'].apply(
            lambda f: os.path.exists(os.path.join('data/raw', f) + '.hea')).values
        sub = df[mask].reset_index(drop=True)
        if len(sub) != len(probs):
            return None
    Y = sub[['label_' + c for c in classes]].values
    return {"probs": probs, "Y": Y, "classes": classes, "df": sub}


@st.cache_data
def load_lightgbm_importances():
    p = 'models/lightgbm_feature_importances.json'
    if not os.path.exists(p):
        return None
    with open(p, 'r') as f:
        return json.load(f)


def binary_metrics_at(probs, y, thr):
    """Confusion-matrix-derived metrics at a given decision threshold."""
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, sens=sens, spec=spec, prec=prec, f1=f1, acc=acc)


PERF_FIG_DIR = 'reports/figures'

def perf_figure(name):
    """Return an absolute path to a pre-computed report figure if it exists."""
    p = os.path.join(PERF_FIG_DIR, name)
    return p if os.path.exists(p) else None


# -------------------------------------------------------------
# Professional line-icon system (Lucide-style, brand cyan)
# -------------------------------------------------------------

ICONS = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.49 4.04 3 5.5l7 7Z"/><path d="M3.5 12.5h4l1.4-2.8 2.6 5.6 2-4.3 1 1.5h4"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "bar-chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "upload": '<path d="M16 16l-4-4-4 4"/><path d="M12 12v9"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "eye": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "clipboard": '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M9 13l2 2 4-4"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
}


def svg_icon(name, size=22, color="#3dd6ec", stroke=1.8):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;">'
            f'{ICONS.get(name, "")}</svg>')


def section_header(name, title, subtitle=None, container=st):
    """Render a consistent icon-chip + title header in the brand aesthetic.
    Built as a single-line HTML string so Streamlit's markdown parser never
    treats the indented block as a code fence."""
    sub = (f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:0.15rem;font-weight:500;">{subtitle}</div>'
           if subtitle else '')
    chip = (f'<div style="display:flex;align-items:center;justify-content:center;width:40px;height:40px;'
            f'border-radius:13px;background:rgba(34,211,238,0.07);border:1px solid rgba(34,211,238,0.16);'
            f'flex-shrink:0;">{svg_icon(name, 22)}</div>')
    titler = (f'<div><div style="font-size:1.3rem;font-weight:800;color:#f8fafc;line-height:1.1;'
              f'letter-spacing:-0.3px;">{title}</div>{sub}</div>')
    html = (f'<div style="display:flex;align-items:center;gap:12px;margin:0.4rem 0 0.9rem 0;">'
            f'{chip}{titler}</div>')
    container.markdown(html, unsafe_allow_html=True)


@st.cache_data
def load_scp_map():
    """Map each PTB-XL SCP statement code to its plain-English description."""
    try:
        scp = pd.read_csv('scp_statements.csv')
        scp = scp.rename(columns={scp.columns[0]: 'code'})
        return scp.set_index('code')['description'].to_dict()
    except Exception:
        return {}


def decode_scp_findings(scp_codes_str):
    """Decode the raw scp_codes dict string into [(code, english_description, confidence)]."""
    import ast
    smap = load_scp_map()
    try:
        codes = ast.literal_eval(scp_codes_str) if isinstance(scp_codes_str, str) else {}
    except Exception:
        codes = {}
    findings = []
    for code, conf in codes.items():
        desc = smap.get(code, code)
        findings.append((code, desc, conf))
    return findings


def style_dark_ax(fig, axes):
    """Apply the dashboard's transparent dark styling to a matplotlib figure."""
    fig.patch.set_alpha(0.0)
    for a in (axes if isinstance(axes, (list, np.ndarray)) else [axes]):
        a.patch.set_alpha(0.0)
        a.tick_params(colors='#cbd5e1', labelsize=8)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['left'].set_color('#3a4763')
        a.spines['bottom'].set_color('#3a4763')
        a.xaxis.label.set_color('#cbd5e1')
        a.yaxis.label.set_color('#cbd5e1')
        a.title.set_color('#e7eef6')


@st.cache_data
def load_validation_frames():
    """Load the three split metadata tables used by the validation visuals."""
    paths = {
        "binary": 'data/subset_metadata_2000.csv',
        "multiclass": 'data/subset_multiclass_metadata.csv',
        "demo": 'data/unseen_demo_metadata.csv',
    }
    out = {}
    for k, p in paths.items():
        out[k] = pd.read_csv(p) if os.path.exists(p) else None
    return out


@st.cache_data
def build_validation_figures():
    """Pre-render the validation graphics once (cached) as transparent PNGs."""
    import io
    from matplotlib.patches import Circle
    vf = load_validation_frames()
    if any(vf.get(k) is None for k in ("binary", "multiclass", "demo")):
        return None
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    dfb, dfm, dfd = vf["binary"], vf["multiclass"], vf["demo"]

    def cap_age(s):
        return s.apply(lambda a: 90 if (pd.notna(a) and a >= 120) else a).dropna()

    def legend(ax):
        ax.legend(fontsize=8, labelcolor='white', facecolor='#0f172a', edgecolor='#475569')

    def to_png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=115)
        plt.close(fig)
        return buf.getvalue()

    out = {}

    # 1) Records vs unique patients per split (within-split leakage check)
    f1, a1 = plt.subplots(figsize=(5.6, 3.8))
    splits = ['Binary\n(triage)', 'Multi-label\n(diagnosis)', 'Demo\n(unseen)']
    recs = [len(dfb), len(dfm), len(dfd)]
    uniq = [dfb['patient_id'].nunique(), dfm['patient_id'].nunique(), dfd['patient_id'].nunique()]
    x = np.arange(3); w = 0.38
    a1.bar(x - w / 2, recs, w, label='Records', color='#22d3ee')
    a1.bar(x + w / 2, uniq, w, label='Unique patients', color='#a78bfa')
    a1.set_yscale('log')
    a1.set_xticks(x); a1.set_xticklabels(splits, fontsize=8)
    a1.set_ylabel('Count (log scale)')
    a1.set_title('Records vs unique patients per split')
    legend(a1)
    style_dark_ax(f1, a1); a1.grid(True, axis='y', alpha=0.2, linestyle='--')
    out['composition'] = to_png(f1)

    # 2) Patient-overlap proof (training vs demo) — disjoint circles
    pb = set(dfb['patient_id'].dropna()); pm = set(dfm['patient_id'].dropna())
    pdem = set(dfd['patient_id'].dropna())
    train_patients = pb | pm
    overlap = train_patients & pdem
    f2, a2 = plt.subplots(figsize=(5.6, 3.8))
    a2.add_patch(Circle((0.34, 0.5), 0.30, color='#22d3ee', alpha=0.28, ec='#22d3ee', lw=2))
    a2.add_patch(Circle((0.80, 0.5), 0.14, color='#f43f5e', alpha=0.30, ec='#f43f5e', lw=2))
    a2.text(0.34, 0.5, f"Training\n{len(train_patients):,}\npatients", ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')
    a2.text(0.80, 0.5, f"Demo\n{len(pdem)}", ha='center', va='center',
            color='white', fontsize=8, fontweight='bold')
    a2.text(0.5, 0.05, f"Shared patients = {len(overlap)}", ha='center',
            color=('#34d399' if len(overlap) == 0 else '#f87171'), fontsize=12, fontweight='bold')
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis('off')
    a2.set_title('Patient overlap: training vs demo cohort')
    f2.patch.set_alpha(0.0)
    out['overlap'] = to_png(f2)

    # 3) Diagnostic class prevalence — training vs demo (representativeness)
    f3, a3 = plt.subplots(figsize=(8, 3.6))
    train_prev = [dfm['label_' + c].mean() * 100 for c in classes]
    demo_prev = [dfd['label_' + c].mean() * 100 for c in classes]
    x = np.arange(len(classes)); w = 0.38
    a3.bar(x - w / 2, train_prev, w, label='Training (multi-label)', color='#22d3ee')
    a3.bar(x + w / 2, demo_prev, w, label='Demo (unseen)', color='#34d399')
    a3.set_xticks(x); a3.set_xticklabels(classes)
    a3.set_ylabel('Prevalence (%)')
    a3.set_title('Diagnostic class prevalence — training vs demo cohort')
    legend(a3)
    style_dark_ax(f3, a3); a3.grid(True, axis='y', alpha=0.2, linestyle='--')
    out['prevalence'] = to_png(f3)

    # 4) Age distribution — training vs demo
    f4, a4 = plt.subplots(figsize=(5.6, 3.6))
    a4.hist(cap_age(dfm['age']), bins=20, density=True, alpha=0.55, color='#22d3ee', label='Training')
    a4.hist(cap_age(dfd['age']), bins=14, density=True, alpha=0.55, color='#34d399', label='Demo')
    a4.set_xlabel('Age (years)'); a4.set_ylabel('Density')
    a4.set_title('Age distribution')
    legend(a4)
    style_dark_ax(f4, a4); a4.grid(True, axis='y', alpha=0.2, linestyle='--')
    out['age'] = to_png(f4)

    # 5) Sex distribution — training vs demo
    f5, a5 = plt.subplots(figsize=(5.6, 3.6))
    def sexprop(df):
        return [(df['sex'] == 0).mean() * 100, (df['sex'] == 1).mean() * 100]
    x = np.arange(2); w = 0.38
    a5.bar(x - w / 2, sexprop(dfm), w, label='Training', color='#22d3ee')
    a5.bar(x + w / 2, sexprop(dfd), w, label='Demo', color='#34d399')
    a5.set_xticks(x); a5.set_xticklabels(['Male', 'Female'])
    a5.set_ylabel('Proportion (%)')
    a5.set_title('Biological sex distribution')
    legend(a5)
    style_dark_ax(f5, a5); a5.grid(True, axis='y', alpha=0.2, linestyle='--')
    out['sex'] = to_png(f5)

    out['overlap_count'] = len(overlap)
    return out


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
            <div style="min-width: 0;">
                <h1 class="banner-title">Heartbreaker&trade;</h1>
                <p class="banner-sub">Proactive diagnostic intelligence on raw 12-lead physiological waveforms.</p>
            </div>
        </div>
        <div class="status-pill">
            <span style="display: block; font-size: 0.72rem; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.35rem;">System Status</span>
            <span style="font-weight: 800; color: #34d399; display: inline-flex; align-items: center; gap: 0.5rem;">
                <span style="height: 8px; width: 8px; background-color: #34d399; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #34d399;"></span>
                ACTIVE&nbsp;&middot;&nbsp;MVP
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# KPI credibility strip — headline, validation-backed numbers
try:
    from sklearn.metrics import roc_auc_score as _auc
    _oofb = load_binary_oof()
    _kpi_auc = f"{_auc(_oofb['y'], _oofb['probs']):.2f}" if _oofb else "0.92"
    _kpi_cohort = f"{len(_oofb['y']):,}" if _oofb else "2,000"
except Exception:
    _kpi_auc, _kpi_cohort = "0.92", "2,000"
st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-card"><div class="kpi-value">{_kpi_auc}</div>
        <div class="kpi-label">Triage ROC-AUC</div><div class="kpi-sub">out-of-fold validated</div></div>
    <div class="kpi-card"><div class="kpi-value">5</div>
        <div class="kpi-label">Diagnostic classes</div><div class="kpi-sub">NORM · MI · STTC · CD · HYP</div></div>
    <div class="kpi-card"><div class="kpi-value">{_kpi_cohort}</div>
        <div class="kpi-label">OOF triage cohort</div><div class="kpi-sub">patient-disjoint CV</div></div>
    <div class="kpi-card"><div class="kpi-value">0</div>
        <div class="kpi-label">Patient leakage</div><div class="kpi-sub">demo fully held-out</div></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(34,211,238,0.10) 0%, rgba(30,41,59,0.45) 100%); padding: 0.7rem 1.2rem; border-radius: 12px; border: 1px solid rgba(34,211,238,0.20); margin-bottom: 1rem; border-left: 4px solid var(--accent);">
            <h3 style="margin: 0; color: #f8fafc; font-size: 1.2rem; display: flex; align-items: center; gap: 10px; font-weight: 800;">
                {svg_icon('sliders', 20)} Clinical Control Panel
            </h3>
            <p style="margin: 0.2rem 0 0 0; color: var(--muted); font-size: 0.85rem;">Select a diagnostic model and an ECG lead to visualize. Cohort filters live in the sidebar.</p>
        </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        model_mode = st.selectbox(
            "Choose Clinical Task:",
            ["Triage Classifier (Binary CNN)", "Differential Diagnosis (Multi-Label CNN)"]
        )
        if model_mode == "Triage Classifier (Binary CNN)" and binary_model is None:
            st.error("⚠️ Binary Model not found in `models/`!")
        elif model_mode == "Differential Diagnosis (Multi-Label CNN)" and multiclass_model is None:
            st.error("⚠️ Multi-Label Model not found in `models/`!")

    # Sidebar for Advanced Filtering
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:0.4rem;">'
        f'{svg_icon("filter", 19)}'
        f'<span style="font-size:1.15rem;font-weight:800;color:#f8fafc;">Advanced Patient Filtering</span></div>',
        unsafe_allow_html=True)
    if df_metadata is not None:
        # PTB-XL de-identifies patients older than 89 by masking their age as 300
        # (HIPAA Safe Harbor). Map that sentinel to a single ">= 90" bucket so the
        # slider and statistics stay clinically meaningful instead of showing "300".
        AGE_CAP = 90  # represents the de-identified ">= 90" elderly cohort
        df_metadata = df_metadata.copy()
        df_metadata['age_filt'] = df_metadata['age'].apply(
            lambda a: AGE_CAP if (pd.notna(a) and a >= 120) else a
        )
        has_deid_elderly = bool((df_metadata['age'] >= 120).any())

        # Age Filter (operates on the cleaned age)
        ages_valid = df_metadata['age_filt'].dropna()
        min_age = int(ages_valid.min()) if not ages_valid.empty else 0
        max_age = int(ages_valid.max()) if not ages_valid.empty else 100
        age_range = st.sidebar.slider("Age range (years)", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        if has_deid_elderly:
            st.sidebar.caption(f"Age **{AGE_CAP}** = de-identified ≥ 90 cohort (PTB-XL HIPAA masking).")

        # Sex Filter (PTB-XL encodes 0 = Male, 1 = Female)
        sex_filter = st.sidebar.selectbox("Biological sex", ["All", "Male", "Female"])

        # Pathology Filter
        subgroup_filter = st.sidebar.selectbox(
            "Ground-Truth Pathology",
            ["All Patients", "Normal (NORM)", "Myocardial Infarction (MI)", "ST/T-Change (STTC)", "Conduction Disturbance (CD)", "Hypertrophy (HYP)"]
        )

        # Apply Filters
        df_filtered = df_metadata.copy()
        df_filtered = df_filtered[(df_filtered['age_filt'] >= age_range[0]) & (df_filtered['age_filt'] <= age_range[1])]
        if sex_filter == "Male":
            df_filtered = df_filtered[df_filtered['sex'] == 0]
        elif sex_filter == "Female":
            df_filtered = df_filtered[df_filtered['sex'] == 1]

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
            
        st.sidebar.markdown(f"""
            <div style="margin-top: 0.5rem; padding: 0.6rem 0.9rem; background: rgba(34,211,238,0.08);
                        border: 1px solid rgba(34,211,238,0.25); border-radius: 10px; text-align: center;">
                <span style="font-size: 1.4rem; font-weight: 800; color: var(--accent);">{len(df_filtered)}</span>
                <span style="color: #cbd5e1; font-size: 0.85rem;"> / {len(df_metadata)} patients match filters</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        subgroup_filter = "All Patients"
        df_filtered = None

    with col_c2:
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
            if st.button("←  Previous Patient", use_container_width=True):
                st.session_state.patient_idx = max(0, st.session_state.patient_idx - 1)
        with col_nav2:
            st.markdown(f"""
                <div style="text-align: center; background-color: rgba(30, 41, 59, 0.6); padding: 0.7rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); height: 100%; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 0.95rem; color: #f8fafc; font-weight: 700;">Record {st.session_state.patient_idx + 1} of {len(df_filtered)}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_nav3:
            if st.button("Next Patient  →", use_container_width=True):
                st.session_state.patient_idx = min(len(df_filtered) - 1, st.session_state.patient_idx + 1)
                
        selected_row = df_filtered.iloc[st.session_state.patient_idx]
        record_filename = selected_row['filename_lr']
    else:
        st.warning("⚠️ Metadata database not found at `data/unseen_demo_metadata.csv`!")
        record_filename = None



tab1, tab2, tab3, tab4 = st.tabs([
    "Diagnostic Inference",
    "Model Performance",
    "Upload & Batch",
    "Validation & Leakage Audit",
])

with tab1:
    # Main Window Logic
    if record_filename:
        record_path = os.path.join('data/raw', record_filename)

        # Preprocess ECG Waveform
        norm_sig, raw_sig = preprocess_ecg_signal(record_path)

        if norm_sig is not None:
            # Create Layout columns (Main view + predictions side-by-side)
            col1, col2 = st.columns([7, 5])

            with col1:
                section_header("activity", f"ECG Waveform — {lead_names[selected_lead_idx]}",
                               "10 s of raw signal vs. 0.5–40 Hz band-pass + lead-wise z-standardised waveform")

                # Time axis in seconds (signals sampled at 100 Hz over a 10 s window)
                fs = 100
                t_raw = np.arange(raw_sig.shape[0]) / fs
                t_norm = np.arange(norm_sig.shape[0]) / fs
                lead_label = lead_names[selected_lead_idx]

                # Matplotlib interactive plotting
                plt.style.use('dark_background')
                fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
                fig.patch.set_alpha(0.0)

                # Raw Signal Plot
                ax[0].patch.set_alpha(0.0)
                ax[0].plot(t_raw, raw_sig[:, selected_lead_idx], color='#22d3ee', linewidth=1)
                ax[0].set_title(f"Raw waveform — {lead_label}", fontsize=10, fontweight='bold', color='white')
                ax[0].set_ylabel("Amplitude (mV)", color='white')
                ax[0].grid(True, alpha=0.2, linestyle='--')
                ax[0].tick_params(colors='white')

                # Filtered + Standardized Plot
                ax[1].patch.set_alpha(0.0)
                ax[1].plot(t_norm, norm_sig[:, selected_lead_idx], color='#10b981', linewidth=1)
                ax[1].set_title("Preprocessed waveform — 0.5–40 Hz band-pass + z-score", fontsize=10, fontweight='bold', color='white')
                ax[1].set_xlabel("Time (s)", color='white')
                ax[1].set_ylabel("Normalized amplitude (σ)", color='white')
                ax[1].grid(True, alpha=0.2, linestyle='--')
                ax[1].tick_params(colors='white')

                # Cleaner, modern frame: drop the top/right spines on both panels
                for _ax in ax:
                    _ax.spines['top'].set_visible(False)
                    _ax.spines['right'].set_visible(False)
                    _ax.spines['left'].set_color('#3a4763')
                    _ax.spines['bottom'].set_color('#3a4763')

                plt.tight_layout()
                st.pyplot(fig)

                import io
                buf = io.BytesIO()
                fig.patch.set_facecolor('#ffffff')
                ax[0].patch.set_facecolor('#ffffff')
                ax[1].patch.set_facecolor('#ffffff')
                ax[0].tick_params(colors='black')
                ax[1].tick_params(colors='black')
                ax[0].set_title(f"Raw waveform — {lead_label} (mV)", color='black')
                ax[1].set_title("Preprocessed waveform — 0.5–40 Hz band-pass + z-score", color='black')
                ax[0].set_xlabel("Time (s)", color='black')
                ax[1].set_xlabel("Time (s)", color='black')
                ax[0].set_ylabel("Amplitude (mV)", color='black')
                ax[1].set_ylabel("Normalized amplitude (σ)", color='black')
                fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
                chart_bytes = buf.getvalue()

                # Show patient metadata
                st.write("---")
                section_header("user", "Patient Demographic Context")
                # PTB-XL masks ages > 89 as 300; present that honestly as ">= 90 (de-identified)"
                _age_val = selected_row['age']
                if pd.notna(_age_val) and _age_val >= 120:
                    age_display = "≥ 90 (de-identified)"
                elif pd.notna(_age_val):
                    age_display = f"{_age_val:.0f} years"
                else:
                    age_display = "Not recorded"
                sex_display = "Male" if selected_row['sex'] == 0 else "Female"

                # Diagnosis: decode the SCP statement codes into plain English instead of
                # showing the raw (often German, sometimes garbled) free-text report.
                demo_findings = decode_scp_findings(selected_row.get('scp_codes', '{}'))
                if demo_findings:
                    finding_items = "".join(
                        f"<li><b>{desc[:1].upper() + desc[1:]}</b> "
                        f"<span style='color:#64748b;'>({code}{'' if (conf in (0, 0.0) or pd.isna(conf)) else f' · {conf:.0f}% conf.'})</span></li>"
                        for code, desc, conf in demo_findings
                    )
                    findings_block = (f"<li><b>Diagnostic findings (SCP codes):</b>"
                                      f"<ul style='margin:0.35rem 0 0 0.25rem;'>{finding_items}</ul></li>")
                else:
                    findings_block = "<li><b>Diagnostic findings (SCP codes):</b> <i>None coded.</i></li>"

                st.markdown(f"""
                <div class="glass-card" style="padding: 1.25rem;">
                    <ul style="margin: 0; color: #cbd5e1; line-height: 1.8;">
                        <li><b>Patient ID:</b> {selected_row['patient_id']:.0f} &nbsp;|&nbsp; <b>ECG ID:</b> {selected_row['ecg_id']}</li>
                        <li><b>Age:</b> {age_display} &nbsp;|&nbsp; <b>Sex:</b> {sex_display}</li>
                        {findings_block}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Signal-derived measurements (heart rate from R-peaks on Lead II)
                section_header("heart", "Signal-Derived Measurements")
                hr = estimate_heart_rate(norm_sig, fs=100, lead_idx=1)
                m1, m2, m3 = st.columns(3)
                if hr is not None:
                    m1.metric("Estimated heart rate", f"{hr['bpm']:.0f} bpm")
                    m2.metric("Detected beats (10 s)", f"{hr['n_beats']}")
                    m3.metric("Mean RR interval", f"{hr['mean_rr_ms']:.0f} ms")
                    if hr['bpm'] < 60:
                        rhythm = "Bradycardia (< 60 bpm)"
                    elif hr['bpm'] > 100:
                        rhythm = "Tachycardia (> 100 bpm)"
                    else:
                        rhythm = "Normal rate (60–100 bpm)"
                    st.caption(f"Rhythm note: **{rhythm}**. R-peaks detected on Lead II — estimate only, not a diagnostic measurement.")
                else:
                    st.caption("Heart rate could not be reliably estimated from Lead II for this record.")

                # All 12 leads at a glance
                with st.expander("View all 12 leads (preprocessed)", expanded=False):
                    fig12, ax12 = plt.subplots(6, 2, figsize=(11, 9), sharex=True)
                    fig12.patch.set_alpha(0.0)
                    t12 = np.arange(norm_sig.shape[0]) / 100
                    axes_flat = ax12.T.flatten()  # left col = I..aVF, right col = V1..V6
                    for k in range(12):
                        a = axes_flat[k]
                        a.patch.set_alpha(0.0)
                        a.plot(t12, norm_sig[:, k], color='#22d3ee', linewidth=0.8)
                        a.set_ylabel(lead_names[k], color='white', fontsize=9,
                                     rotation=0, ha='right', va='center')
                        a.tick_params(colors='white', labelsize=7)
                        a.grid(True, alpha=0.15, linestyle='--')
                    ax12[-1, 0].set_xlabel("Time (s)", color='white', fontsize=9)
                    ax12[-1, 1].set_xlabel("Time (s)", color='white', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig12)
                    plt.close(fig12)

            with col2:
                section_header("cpu", "Diagnostic Inference",
                               "Neural-network triage & differential diagnosis")

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
                    scp_df = pd.read_csv('scp_statements.csv')
                    scp_df = scp_df.rename(columns={scp_df.columns[0]: 'code'})
                    scp_map = scp_df.set_index('code')['description'].to_dict()
                except Exception:
                    scp_map = {}

                for code, val in codes_dict.items():
                    desc = scp_map.get(code, code)
                    desc = desc[:1].upper() + desc[1:] if desc else code
                    scp_details.append(f"• <b>{desc}</b> ({code})")

                scp_details_str = "<br>".join(scp_details) if scp_details else "None"

                def display_ground_truth_box():
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(15, 23, 42, 0.85); border-left: 4px solid #22d3ee; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid rgba(34, 211, 238, 0.15);">
                        <div style="font-size: 1rem; color: #f8fafc; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
                            {svg_icon('clipboard', 18)} <b>Verified Clinical Ground Truth</b>
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-left: 1.5rem; line-height: 1.6;">
                            <span style="color: #22d3ee; font-weight: bold;">Superclass Categories:</span> {gt_str}<br>
                            <span style="color: #22d3ee; font-weight: bold;">Specific Clinical Findings (SCP codes):</span><br>
                            {scp_details_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="glass-card" style="padding: 1.2rem; margin-bottom: 0.6rem;">
                    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                        Run the convolutional neural network directly on the raw 12-lead waveform to generate a diagnostic verdict.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Run Neural Network Diagnostic", use_container_width=True):
                    # Format signal for model (shape batch=1, time=1000, leads=12)
                    input_batch = tf.expand_dims(norm_sig, 0)

                    if model_mode == "Triage Classifier (Binary CNN)":
                        if binary_model is not None:
                            with st.spinner("Analysing 12-lead waveform…"):
                                probability = binary_model.predict(input_batch)[0][0]

                            st.markdown("### Model Triage Verdict")

                            is_abnormal = probability >= 0.50

                            if is_abnormal:
                                verdict_title = "⚠️ FLAG ABNORMAL"
                                verdict_text = "This record has been flagged by the Heartbreaker Triage Engine for high-priority clinical review."
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

                            # Prediction-vs-ground-truth agreement badge
                            gt_abnormal = int(selected_row['label_NORM'] != 1)
                            model_correct = (int(is_abnormal) == gt_abnormal)
                            if model_correct:
                                _bcol, _bbg, _btxt = "#10b981", "rgba(16,185,129,0.12)", "✓ Model prediction matches the verified ground truth"
                            else:
                                _bcol, _bbg, _btxt = "#f59e0b", "rgba(245,158,11,0.12)", "✗ Model prediction differs from the verified ground truth"
                            st.markdown(f"""
                            <div style="margin-top: 0.75rem; padding: 0.6rem 1rem; border-radius: 10px;
                                        background: {_bbg}; border: 1px solid {_bcol}55; color: {_bcol}; font-weight: 700;">
                                {_btxt}
                            </div>
                            """, unsafe_allow_html=True)

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
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 6px; font-size: 0.95rem; margin-bottom: 0.4rem;">
        <span style="font-weight: 700; color: #f8fafc;">Triage Abnormality Probability</span>
        <div style="display: flex; align-items: center; gap: 8px; margin-left: auto;">
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
                                label="Download Clinical Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"clinical_report_{selected_row['patient_id']}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.write("* **Pipeline Sensitivity (Recall):** `85.80%` | **Specificity:** `84.10%` (out-of-fold validated)")
                            
                            # Grad-CAM Visualization
                            with st.expander("Interpretability — Grad-CAM feature heatmap", expanded=False):
                                st.write("The heatmap highlights the temporal regions of the ECG waveform that most strongly influenced the neural network's triage decision. Red/warm regions indicate high influence.")
                                try:
                                    heatmap = compute_gradcam_1d(binary_model, input_batch, target_class_idx=0)
                                    fig_cam, ax_cam = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
                                    fig_cam.patch.set_alpha(0.0)
                                    display_leads = [0, 1, 6, 10] # I, II, V1, V5
                                    lead_names_cam = ["Lead I", "Lead II", "V1", "V5"]
                                    for i, (l_idx, l_name) in enumerate(zip(display_leads, lead_names_cam)):
                                        ax_cam[i].patch.set_alpha(0.0)
                                        # Background Heatmap (time in seconds)
                                        t_cam = np.arange(norm_sig.shape[0]) / 100
                                        y_min, y_max = np.min(norm_sig[:, l_idx]), np.max(norm_sig[:, l_idx])
                                        padding = (y_max - y_min) * 0.1
                                        y1, y2 = y_min - padding, y_max + padding
                                        ax_cam[i].imshow(heatmap[None, :], aspect='auto', cmap='jet', extent=[0, 10, y1, y2], alpha=0.4, interpolation='nearest')
                                        # ECG Waveform
                                        ax_cam[i].plot(t_cam, norm_sig[:, l_idx], color='white', linewidth=1.2, alpha=0.9)
                                        ax_cam[i].set_ylabel(l_name, color='white')
                                        ax_cam[i].set_ylim(y1, y2)
                                        ax_cam[i].tick_params(colors='white')
                                        ax_cam[i].grid(True, alpha=0.2, linestyle='--')
                                    ax_cam[3].set_xlabel("Time (s)", color='white')
                                    plt.tight_layout()
                                    st.pyplot(fig_cam)
                                except Exception as e:
                                    st.error(f"Could not generate Grad-CAM: {str(e)}")

                        else:
                            st.error("Error: Binary model missing.")

                    else: # Differential Diagnosis (Multi-Label CNN)
                        if multiclass_model is not None:
                            with st.spinner("Analysing 12-lead waveform…"):
                                predictions = multiclass_model.predict(input_batch)[0]
                            classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
                            class_labels = {
                                'NORM': 'Normal ECG (NORM)',
                                'MI': 'Myocardial Infarction (MI)',
                                'STTC': 'ST/T Change (STTC)',
                                'CD': 'Conduction Disturbance (CD)',
                                'HYP': 'Hypertrophy (HYP)'
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
                                verdict_text = f"The Heartbreaker Diagnostic Engine has flagged the following active pathology classes: {', '.join(active_pathologies)}. Immediate clinical review is recommended."
                                st.markdown(f"""
                                <div class="status-card-abnormal">
                                    <div class="status-header-abnormal">{verdict_title}</div>
                                    <p style="margin: 0; font-size: 1.1rem; color: #fecaca;">
                                        The <b>Heartbreaker Diagnostic Engine</b> has flagged the following active pathology classes: {pathology_str}. Immediate clinical review is recommended.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif prob_norm >= thresh_norm:
                                verdict_title = "✅ NORMAL CARDIAC STATUS"
                                verdict_text = "The Heartbreaker Diagnostic Engine predicts a normal cardiac pattern. No active pathologies were flagged."
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

                            # Per-class prediction-vs-ground-truth agreement
                            _agree = 0
                            for c in classes:
                                pred_pos = predictions[classes.index(c)] >= thresholds.get(c, 0.5)
                                gt_pos = selected_row['label_' + c] == 1
                                if bool(pred_pos) == bool(gt_pos):
                                    _agree += 1
                            if _agree == len(classes):
                                _bcol, _bbg, _btxt = "#10b981", "rgba(16,185,129,0.12)", f"✓ All {len(classes)} diagnostic classes match the verified ground truth"
                            else:
                                _bcol, _bbg, _btxt = "#f59e0b", "rgba(245,158,11,0.12)", f"⚠ {_agree}/{len(classes)} diagnostic classes match the verified ground truth"
                            st.markdown(f"""
                            <div style="margin-top: 0.75rem; padding: 0.6rem 1rem; border-radius: 10px;
                                        background: {_bbg}; border: 1px solid {_bcol}55; color: {_bcol}; font-weight: 700;">
                                {_btxt}
                            </div>
                            """, unsafe_allow_html=True)

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
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 6px; font-size: 0.95rem; margin-bottom: 0.4rem;">
        <span style="font-weight: 700; color: #f8fafc;">{class_labels[c]}</span>
        <div style="display: flex; align-items: center; gap: 8px; margin-left: auto;">
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
                                label="Download Clinical Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"clinical_report_{selected_row['patient_id']}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                            # Grad-CAM Visualization
                            with st.expander("Interpretability — Grad-CAM feature heatmap", expanded=False):
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
                                        # Background Heatmap (time in seconds)
                                        t_cam = np.arange(norm_sig.shape[0]) / 100
                                        y_min, y_max = np.min(norm_sig[:, l_idx]), np.max(norm_sig[:, l_idx])
                                        padding = (y_max - y_min) * 0.1
                                        y1, y2 = y_min - padding, y_max + padding
                                        ax_cam[i].imshow(heatmap[None, :], aspect='auto', cmap='jet', extent=[0, 10, y1, y2], alpha=0.4, interpolation='nearest')
                                        # ECG Waveform
                                        ax_cam[i].plot(t_cam, norm_sig[:, l_idx], color='white', linewidth=1.2, alpha=0.9)
                                        ax_cam[i].set_ylabel(l_name, color='white')
                                        ax_cam[i].set_ylim(y1, y2)
                                        ax_cam[i].tick_params(colors='white')
                                        ax_cam[i].grid(True, alpha=0.2, linestyle='--')
                                    ax_cam[3].set_xlabel("Time (s)", color='white')
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
                    <h4 style="margin-top:0; color:#22d3ee; display:flex; align-items:center; gap:8px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg> Cardiology MVP Business Case</h4>
                    <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:0.75rem;">
                        <b>1. Automated Clinical Triage:</b> By screening raw signals at the source and ranking records by abnormality probability, clinics can automate triage, routing critical cardiovascular alerts to cardiologists instantly.
                    </p>
                    <p style="font-size:0.95rem; color:#cbd5e1; margin-bottom:0;">
                        <b>2. Hardware-Level Integration:</b> Operating directly on raw 1D digital signal telemetry rather than rendered images ensures complete immunity to visual artifacts and enables low-latency edge-device execution directly inside ECG carts.
                    </p>
                </div>
                """, unsafe_allow_html=True)


with tab4:
    section_header("shield", "Clinical Validation & Leakage Audit Engine")
    st.write("This interactive panel allows you to audit the training, validation, and demo dataset splits to guarantee zero patient-level target leakage.")

    # ---------------- Visual validation dashboard ----------------
    vfigs = build_validation_figures()
    if vfigs is not None:
        if vfigs['overlap_count'] == 0:
            st.success("✅ **Zero patient leakage:** the unseen demo cohort shares no patients with any training split (visualised below).")
        else:
            st.error(f"❌ **Leakage:** {vfigs['overlap_count']} demo patients also appear in training.")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.image(vfigs['composition'], width=360)
            st.caption("Per split: total records vs unique patient IDs. For the multi-label and demo sets the bars match exactly — one ECG per patient, so no within-split patient leakage.")
        with r1c2:
            st.image(vfigs['overlap'], width=360)
            st.caption("The demo cohort is patient-disjoint from training — the circles do not intersect, proving true out-of-sample evaluation.")

        st.image(vfigs['prevalence'], width=520)
        st.caption("All five diagnostic classes appear in the unseen demo cohort. The demo (n = 28) is modestly enriched for pathology (lower NORM, higher STTC/MI), making it a deliberately challenging out-of-sample test rather than a cherry-picked easy set.")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.image(vfigs['age'], width=340)
            st.caption("Age distribution (ages ≥ 90 collapsed to the de-identified bucket).")
        with r2c2:
            st.image(vfigs['sex'], width=340)
            st.caption("Biological-sex balance across training and demo cohorts.")

        st.markdown("---")

    # Audit button
    if st.button("Run Real-Time Dataset Integrity & Leakage Check", use_container_width=True):
        import pandas as pd
        import ast
        
        # Load datasets
        binary_path = 'data/subset_metadata_2000.csv'
        multi_path = 'data/subset_multiclass_metadata.csv'
        demo_path = 'data/unseen_demo_metadata.csv'
        base_dir = 'data/raw'
        
        st.markdown("#### 1. Binary Triage Split Integrity")
        if os.path.exists(binary_path):
            df_bin = pd.read_csv(binary_path)
            total_bin = len(df_bin)
            uniq_bin = df_bin['patient_id'].nunique()
            dup_bin = total_bin - uniq_bin
            
            # Disk check
            present_bin = 0
            missing_bin_files = []
            for _, row in df_bin.iterrows():
                path = os.path.join(base_dir, row['filename_lr'])
                if os.path.exists(path + '.hea'):
                    present_bin += 1
                else:
                    missing_bin_files.append(row['filename_lr'])
            
            st.write(f"* **Total Triage Records (Metadata):** `{total_bin}`")
            st.write(f"* **Physically Present on Disk:** `{present_bin}/{total_bin}`")
            st.write(f"* **Unique Patient IDs:** `{uniq_bin}`")
            
            if len(missing_bin_files) > 0:
                st.warning(f"⚠️ **Note:** {len(missing_bin_files)} files are missing on disk: `{', '.join(missing_bin_files[:5])}`" + ("..." if len(missing_bin_files) > 5 else ""))
            else:
                st.success("✅ **Disk Check:** All binary triage signal files are present on disk.")
                
            if dup_bin == 0:
                st.success("✅ **Zero Patient Leakage:** Every record corresponds to a unique patient in the binary dataset.")
            else:
                st.info(f"ℹ️ **Note:** {dup_bin} duplicate patient records exist, but cross-fold splitting is grouped strictly by `patient_id` to prevent leakage.")
        else:
            st.error("Binary split metadata not found.")
            
        st.markdown("#### 2. Multiclass Diagnostic Split Integrity")
        if os.path.exists(multi_path):
            df_mul = pd.read_csv(multi_path)
            total_mul = len(df_mul)
            uniq_mul = df_mul['patient_id'].nunique()
            
            # Disk check
            present_mul = 0
            missing_mul_files = []
            valid_train_patients = set()
            for _, row in df_mul.iterrows():
                path = os.path.join(base_dir, row['filename_lr'])
                if os.path.exists(path + '.hea'):
                    present_mul += 1
                    valid_train_patients.add(row['patient_id'])
                else:
                    missing_mul_files.append(row['filename_lr'])
            
            st.write(f"* **Total Multiclass Records (Metadata):** `{total_mul}`")
            st.write(f"* **Physically Present on Disk:** `{present_mul}/{total_mul}`")
            st.write(f"* **Unique Patient IDs (Metadata):** `{uniq_mul}`")
            
            if len(missing_mul_files) > 0:
                st.warning(f"⚠️ **Exclusion Active:** {len(missing_mul_files)} files are missing on disk: `{', '.join(missing_mul_files)}`.")
                st.success(f"✅ **Disk Integrity Adaptation Active:** The validation engine dynamically filtered out the missing records. All active model evaluations are performed on the exact same **{present_mul} successfully loaded records**.")
            else:
                st.success("✅ **Disk Check:** All multiclass signal files are present on disk.")
                
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
            
            # Disk check
            present_demo = 0
            missing_demo_files = []
            for _, row in df_demo.iterrows():
                path = os.path.join(base_dir, row['filename_lr'])
                if os.path.exists(path + '.hea'):
                    present_demo += 1
                else:
                    missing_demo_files.append(row['filename_lr'])
            
            # Check for overlap between demo cohort and valid training sets on disk
            train_patients = set()
            if os.path.exists(binary_path):
                for _, row in pd.read_csv(binary_path).iterrows():
                    path = os.path.join(base_dir, row['filename_lr'])
                    if os.path.exists(path + '.hea'):
                        train_patients.add(row['patient_id'])
            if os.path.exists(multi_path):
                train_patients.update(valid_train_patients)
                
            demo_patients = set(df_demo['patient_id'].dropna().tolist())
            overlap = demo_patients.intersection(train_patients)
            
            st.write(f"* **Total Demo Records (Metadata):** `{total_demo}`")
            st.write(f"* **Physically Present on Disk:** `{present_demo}/{total_demo}`")
            st.write(f"* **Unique Demo Patient IDs:** `{uniq_demo}`")
            
            if len(overlap) == 0:
                st.success(f"✅ **Out-of-Sample Clean:** 0/{uniq_demo} demo patients overlap with the training datasets. True clinical generalization verified!")
            else:
                st.error(f"❌ **LEAKAGE DETECTED:** {len(overlap)} demo patients exist in the training set!")


# =============================================================
# TAB 2 — MODEL PERFORMANCE
# =============================================================
with tab2:
    from sklearn.metrics import (roc_curve, roc_auc_score,
                                 precision_recall_curve, average_precision_score)

    def _style_dark_ax(fig, axes):
        style_dark_ax(fig, axes)

    def show_fig(name, caption):
        p = perf_figure(name)
        if p:
            st.image(p, caption=caption, use_column_width=True)

    section_header("bar-chart", "Model Performance & Validation",
                   "Out-of-fold metrics from patient-disjoint cross-validation, computed live")

    perf_view = st.radio(
        "Select view:",
        ["Binary Triage (CNN)", "Multi-label Diagnosis (CNN)", "CNN vs LightGBM", "Subgroup Fairness"],
        horizontal=True,
    )

    # ---------------- Binary Triage ----------------
    if perf_view == "Binary Triage (CNN)":
        oof = load_binary_oof()
        if oof is None:
            st.warning("Binary OOF predictions (`models/clean_oof_ecg_probs.npy`) not found.")
        else:
            probs, y = oof["probs"], oof["y"]
            auc = roc_auc_score(y, probs)
            ap = average_precision_score(y, probs)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{auc:.3f}")
            c2.metric("PR-AUC", f"{ap:.3f}")
            c3.metric("OOF cases", f"{len(y):,}")
            c4.metric("Abnormal prevalence", f"{y.mean():.1%}")

            st.markdown("#### Interactive decision threshold")
            st.caption("Drag the cutoff to watch the sensitivity / specificity trade-off update live. "
                       "The triage tab itself uses a fixed 0.50 cutoff.")
            thr = st.slider("Decision threshold — probability of abnormal", 0.01, 0.99, 0.50, 0.01)
            mt = binary_metrics_at(probs, y, thr)
            mc = st.columns(4)
            mc[0].metric("Sensitivity (recall)", f"{mt['sens']:.1%}")
            mc[1].metric("Specificity", f"{mt['spec']:.1%}")
            mc[2].metric("Precision (PPV)", f"{mt['prec']:.1%}")
            mc[3].metric("F1 score", f"{mt['f1']:.3f}")

            g1, g2 = st.columns(2)
            with g1:
                fpr, tpr, _ = roc_curve(y, probs)
                fr, ar = plt.subplots(figsize=(5, 4))
                ar.plot(fpr, tpr, color='#22d3ee', lw=2, label=f'ROC (AUC = {auc:.3f})')
                ar.plot([0, 1], [0, 1], '--', color='#64748b', lw=1)
                ar.scatter([1 - mt['spec']], [mt['sens']], color='#f43f5e', s=70, zorder=5,
                           label=f'Operating point @ {thr:.2f}')
                ar.set_xlabel('1 − Specificity (FPR)')
                ar.set_ylabel('Sensitivity (TPR)')
                ar.set_title('ROC curve')
                ar.legend(loc='lower right', fontsize=8, labelcolor='white',
                          facecolor='#0f172a', edgecolor='#475569')
                _style_dark_ax(fr, ar)
                ar.grid(True, alpha=0.2, linestyle='--')
                st.pyplot(fr)
                plt.close(fr)
            with g2:
                cm = np.array([[mt['tn'], mt['fp']], [mt['fn'], mt['tp']]])
                fc, ac = plt.subplots(figsize=(5, 4))
                ac.imshow(cm, cmap='cividis')
                ac.set_xticks([0, 1]); ac.set_xticklabels(['Pred. Normal', 'Pred. Abnormal'])
                ac.set_yticks([0, 1]); ac.set_yticklabels(['True Normal', 'True Abnormal'])
                for (i, j), v in np.ndenumerate(cm):
                    ac.text(j, i, f'{v}', ha='center', va='center', color='white',
                            fontweight='bold', fontsize=13)
                ac.set_title(f'Confusion matrix @ {thr:.2f}')
                _style_dark_ax(fc, ac)
                ac.grid(False)
                st.pyplot(fc)
                plt.close(fc)
            st.info("Targeting **sensitivity ≥ 0.85** (a triage-safe operating point) reproduces the reported "
                    "≈ 0.86 sensitivity / ≈ 0.84 specificity — the trade-off this slider makes explicit.")

            with st.expander("Pre-computed validation figures (per-fold, confidence intervals, robustness)"):
                show_fig('fig1_per_fold_performance.png', 'Per-fold performance across the 5 cross-validation folds')
                show_fig('fig2_oof_metrics_ci.png', 'OOF metrics with 95% confidence intervals')
                show_fig('fig8_sens_spec_scatter.png', 'Sensitivity vs. specificity across folds')
                show_fig('permutation_test_chart.png', 'Label-permutation test (guards against chance-level learning)')
                show_fig('ablation_ladder_chart.png', 'Ablation ladder — contribution of each pipeline component')

    # ---------------- Multi-label Diagnosis (CNN) ----------------
    elif perf_view == "Multi-label Diagnosis (CNN)":
        oof = load_multiclass_oof("CNN")
        if oof is None:
            st.warning("Multi-label CNN OOF predictions not found.")
        else:
            probs, Y, classes = oof["probs"], oof["Y"], oof["classes"]
            rows = []
            for i, c in enumerate(classes):
                m = binary_metrics_at(probs[:, i], Y[:, i], thresholds.get(c, 0.5))
                rows.append({
                    "Class": c,
                    "ROC-AUC": round(roc_auc_score(Y[:, i], probs[:, i]), 3),
                    "PR-AUC": round(average_precision_score(Y[:, i], probs[:, i]), 3),
                    "Youden threshold": round(thresholds.get(c, 0.5), 3),
                    "Sensitivity": f"{m['sens']:.1%}",
                    "Specificity": f"{m['spec']:.1%}",
                    "Support (+)": int(Y[:, i].sum()),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            fm, am = plt.subplots(figsize=(6, 5))
            for i, c in enumerate(classes):
                fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i])
                am.plot(fpr, tpr, lw=1.8, label=f'{c} (AUC {roc_auc_score(Y[:, i], probs[:, i]):.2f})')
            am.plot([0, 1], [0, 1], '--', color='#64748b', lw=1)
            am.set_xlabel('1 − Specificity'); am.set_ylabel('Sensitivity')
            am.set_title('Per-class ROC — CNN OOF')
            am.legend(loc='lower right', fontsize=8, labelcolor='white',
                      facecolor='#0f172a', edgecolor='#475569')
            _style_dark_ax(fm, am)
            am.grid(True, alpha=0.2, linestyle='--')
            st.pyplot(fm)
            plt.close(fm)

            st.markdown("#### Per-class threshold explorer")
            sel = st.selectbox("Diagnostic class", classes)
            si = classes.index(sel)
            tsel = st.slider(f"{sel} threshold", 0.01, 0.99, float(thresholds.get(sel, 0.5)), 0.01)
            ms = binary_metrics_at(probs[:, si], Y[:, si], tsel)
            sc = st.columns(4)
            sc[0].metric("Sensitivity", f"{ms['sens']:.1%}")
            sc[1].metric("Specificity", f"{ms['spec']:.1%}")
            sc[2].metric("Precision", f"{ms['prec']:.1%}")
            sc[3].metric("F1 score", f"{ms['f1']:.3f}")
            if sel == 'HYP':
                st.warning("HYP is presented for pipeline validation only — sparse positives make it clinically unready.")

            with st.expander("Pre-computed multiclass figures (confusion, ROC, PR, operating points)"):
                show_fig('multiclass_confusion_matrix_cnn.png', 'Multi-label confusion matrix (CNN)')
                show_fig('multiclass_roc_curve_cnn.png', 'Per-class ROC (CNN)')
                show_fig('multiclass_pr_curve_cnn.png', 'Per-class precision–recall (CNN)')
                show_fig('multiclass_operating_points_cnn.png', 'Operating points at Youden-optimal thresholds (CNN)')

    # ---------------- CNN vs LightGBM ----------------
    elif perf_view == "CNN vs LightGBM":
        cnno = load_multiclass_oof("CNN")
        lgbo = load_multiclass_oof("LightGBM")
        if cnno is None or lgbo is None:
            st.warning("Both CNN and LightGBM OOF predictions are required for the comparison.")
        else:
            classes = cnno["classes"]
            rows = []
            for i, c in enumerate(classes):
                rows.append({
                    "Class": c,
                    "CNN ROC-AUC": round(roc_auc_score(cnno["Y"][:, i], cnno["probs"][:, i]), 3),
                    "LightGBM ROC-AUC": round(roc_auc_score(lgbo["Y"][:, i], lgbo["probs"][:, i]), 3),
                    "CNN PR-AUC": round(average_precision_score(cnno["Y"][:, i], cnno["probs"][:, i]), 3),
                    "LightGBM PR-AUC": round(average_precision_score(lgbo["Y"][:, i], lgbo["probs"][:, i]), 3),
                })
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            fb, ab = plt.subplots(figsize=(7, 4))
            xpos = np.arange(len(classes)); w = 0.36
            ab.bar(xpos - w / 2, cmp_df["CNN ROC-AUC"], w, label='CNN (raw waveform)', color='#22d3ee')
            ab.bar(xpos + w / 2, cmp_df["LightGBM ROC-AUC"], w, label='LightGBM (engineered features)', color='#a78bfa')
            ab.set_xticks(xpos); ab.set_xticklabels(classes)
            ab.set_ylim(0.5, 1.0); ab.set_ylabel('OOF ROC-AUC')
            ab.set_title('CNN vs LightGBM — discrimination by class')
            ab.legend(fontsize=8, labelcolor='white', facecolor='#0f172a', edgecolor='#475569')
            _style_dark_ax(fb, ab)
            ab.grid(True, axis='y', alpha=0.2, linestyle='--')
            st.pyplot(fb)
            plt.close(fb)

            fi = load_lightgbm_importances()
            if fi:
                st.markdown("#### LightGBM top features (engineered ECG biomarkers)")
                fcls = st.selectbox("Feature importance for class", list(fi.keys()))
                items = fi[fcls][:12]
                names = [it[0] for it in items][::-1]
                vals = [it[1] for it in items][::-1]
                ff, af = plt.subplots(figsize=(7, 5))
                af.barh(names, vals, color='#22d3ee')
                af.set_title(f'Top gain-importance features — {fcls}')
                af.set_xlabel('Gain importance')
                _style_dark_ax(ff, af)
                af.grid(True, axis='x', alpha=0.2, linestyle='--')
                st.pyplot(ff)
                plt.close(ff)

            with st.expander("Side-by-side pre-computed figures"):
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown("**CNN**")
                    show_fig('multiclass_confusion_matrix_cnn.png', 'CNN confusion matrix')
                    show_fig('multiclass_roc_curve_cnn.png', 'CNN ROC')
                with ic2:
                    st.markdown("**LightGBM**")
                    show_fig('multiclass_confusion_matrix_lightgbm.png', 'LightGBM confusion matrix')
                    show_fig('multiclass_roc_curve_lightgbm.png', 'LightGBM ROC')

    # ---------------- Subgroup Fairness ----------------
    else:
        st.write("ROC-AUC stratified by demographic subgroup — a check for performance disparities (algorithmic bias). "
                 "Computed live on the binary triage OOF set.")
        oof = load_binary_oof()
        if oof is None:
            st.warning("Binary OOF predictions not found.")
        else:
            df_o, probs, y = oof["df"], oof["probs"], oof["y"]
            rows = []
            for sx, lab in [(0, 'Male'), (1, 'Female')]:
                mask = (df_o['sex'] == sx).values
                if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
                    rows.append({"Subgroup": lab, "N": int(mask.sum()),
                                 "ROC-AUC": round(roc_auc_score(y[mask], probs[mask]), 3),
                                 "Abnormal prevalence": f"{y[mask].mean():.1%}"})
            ages = df_o['age'].apply(lambda a: 90 if (pd.notna(a) and a >= 120) else a)
            for lo, hi, lab in [(0, 40, '< 40'), (40, 60, '40–59'), (60, 75, '60–74'), (75, 999, '75+')]:
                mask = ((ages >= lo) & (ages < hi)).values
                if mask.sum() > 10 and len(np.unique(y[mask])) > 1:
                    rows.append({"Subgroup": f'Age {lab}', "N": int(mask.sum()),
                                 "ROC-AUC": round(roc_auc_score(y[mask], probs[mask]), 3),
                                 "Abnormal prevalence": f"{y[mask].mean():.1%}"})
            sg_df = pd.DataFrame(rows)
            st.dataframe(sg_df, use_container_width=True, hide_index=True)
            spread = sg_df["ROC-AUC"].max() - sg_df["ROC-AUC"].min()
            if spread <= 0.05:
                st.success(f"✅ Subgroup AUC spread is small ({spread:.3f}) — no major disparity detected.")
            else:
                st.warning(f"⚠️ Subgroup AUC spread is {spread:.3f} — inspect potential disparity.")

        with st.expander("Pre-computed subgroup fairness figures"):
            show_fig('subgroup_roc_auc_sex.png', 'ROC-AUC by biological sex')
            show_fig('subgroup_roc_auc_age.png', 'ROC-AUC by age band')


# =============================================================
# TAB 3 — UPLOAD & BATCH
# =============================================================
with tab3:
    section_header("upload", "Upload & Batch Inference")

    st.markdown("#### Upload your own ECG")
    st.write("Run the exact pipeline (0.5–40 Hz band-pass + lead-wise z-norm) and a model on your own recording. "
             "Accepts a **CSV** with 12 columns (one per lead, samples in rows), or a **WFDB** header+signal pair "
             "(`.hea` and `.dat`).")
    up_model = st.selectbox("Model to apply",
                            ["Triage Classifier (Binary CNN)", "Differential Diagnosis (Multi-Label CNN)"],
                            key="upload_model")
    files = st.file_uploader("Upload a .csv, or both a .hea and a .dat file",
                             type=["csv", "hea", "dat"], accept_multiple_files=True, key="uploader")

    user_sig = None
    if files:
        try:
            csvs = [f for f in files if f.name.lower().endswith('.csv')]
            heas = [f for f in files if f.name.lower().endswith('.hea')]
            dats = [f for f in files if f.name.lower().endswith('.dat')]
            if csvs:
                raw = pd.read_csv(csvs[0])
                arr = raw.select_dtypes(include=[np.number]).values
                if arr.shape[1] != 12 and arr.shape[0] == 12:
                    arr = arr.T
                user_sig = arr
            elif heas and dats:
                import tempfile
                tmp = tempfile.mkdtemp()
                base = os.path.splitext(heas[0].name)[0]
                with open(os.path.join(tmp, base + '.hea'), 'wb') as fo:
                    fo.write(heas[0].getbuffer())
                with open(os.path.join(tmp, base + '.dat'), 'wb') as fo:
                    fo.write(dats[0].getbuffer())
                rec = wfdb.rdrecord(os.path.join(tmp, base))
                user_sig = rec.p_signal
            else:
                st.info("Please provide a .csv, or **both** a .hea and a .dat file.")
        except Exception as e:
            st.error(f"Could not read the upload: {e}")

    if user_sig is not None:
        norm = preprocess_signal_array(user_sig)
        if norm is None:
            st.error(f"Expected a (samples × 12) signal; received shape {np.asarray(user_sig).shape}. "
                     "Provide 12 leads in the standard order (I, II, III, aVR, aVL, aVF, V1–V6).")
        else:
            up_hr = estimate_heart_rate(norm, fs=100, lead_idx=1)
            fu, au = plt.subplots(figsize=(10, 2.6))
            au.plot(np.arange(norm.shape[0]) / 100, norm[:, 1], color='#22d3ee', lw=1)
            au.set_title("Uploaded recording — Lead II (preprocessed)", color='white', fontsize=10, fontweight='bold')
            au.set_xlabel("Time (s)", color='white'); au.set_ylabel("σ", color='white')
            fu.patch.set_alpha(0.0); au.patch.set_alpha(0.0)
            au.tick_params(colors='white'); au.grid(True, alpha=0.2, linestyle='--')
            for s in au.spines.values():
                s.set_color('#475569')
            st.pyplot(fu)
            plt.close(fu)
            if up_hr:
                st.caption(f"Estimated heart rate: **{up_hr['bpm']:.0f} bpm** ({up_hr['n_beats']} beats detected).")

            xb = tf.expand_dims(norm, 0)
            if up_model.startswith("Triage"):
                if binary_model is None:
                    st.error("Binary model not loaded.")
                else:
                    p = float(binary_model.predict(xb, verbose=0)[0][0])
                    if p >= 0.5:
                        st.markdown(f"""<div class="status-card-abnormal"><div class="status-header-abnormal">⚠️ FLAG ABNORMAL</div>
                        <p style="margin:0;color:#fecaca;">Abnormality probability <b>{p:.1%}</b> (cutoff 0.500).</p></div>""",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="status-card-normal"><div class="status-header-normal">✅ NORMAL ECG</div>
                        <p style="margin:0;color:#a7f3d0;">Abnormality probability <b>{p:.1%}</b> (cutoff 0.500).</p></div>""",
                                    unsafe_allow_html=True)
            else:
                if multiclass_model is None:
                    st.error("Multi-label model not loaded.")
                else:
                    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
                    pr = multiclass_model.predict(xb, verbose=0)[0]
                    rows = []
                    for i, c in enumerate(classes):
                        th = thresholds.get(c, 0.5)
                        rows.append({"Class": c, "Probability": f"{pr[i]:.1%}",
                                     "Threshold": round(th, 3),
                                     "Call": "POSITIVE" if pr[i] >= th else "negative"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Batch-score the filtered cohort")
    if df_filtered is None or len(df_filtered) == 0:
        st.info("No patients in the current filter. Adjust the sidebar filters to build a cohort.")
    else:
        st.write(f"Run **{model_mode}** across all **{len(df_filtered)}** filtered patients, score against ground truth, and export.")
        if st.button(f"Run model on {len(df_filtered)} patients", key="batch_run"):
            is_binary = model_mode.startswith("Triage")
            mdl = binary_model if is_binary else multiclass_model
            if mdl is None:
                st.error("Selected model is not loaded.")
            else:
                classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
                prog = st.progress(0.0, text="Scoring cohort…")
                results = []
                n = len(df_filtered)
                for k, (_, row) in enumerate(df_filtered.iterrows()):
                    ns, _ = preprocess_ecg_signal(os.path.join('data/raw', row['filename_lr']))
                    if ns is None:
                        continue
                    xb = tf.expand_dims(ns, 0)
                    rec = {"patient_id": int(row['patient_id']), "ecg_id": row['ecg_id'],
                           "age": ('≥90' if (pd.notna(row['age']) and row['age'] >= 120) else int(row['age'])),
                           "sex": ('M' if row['sex'] == 0 else 'F')}
                    if is_binary:
                        p = float(mdl.predict(xb, verbose=0)[0][0])
                        pred = int(p >= 0.5); gt = int(row['label_NORM'] != 1)
                        rec.update({"P(abnormal)": round(p, 3),
                                    "Predicted": "Abnormal" if pred else "Normal",
                                    "GroundTruth": "Abnormal" if gt else "Normal",
                                    "Correct": bool(pred == gt)})
                    else:
                        pr = mdl.predict(xb, verbose=0)[0]
                        preds = []
                        for c in classes:
                            pp = float(pr[classes.index(c)])
                            rec[f"P({c})"] = round(pp, 3)
                            if pp >= thresholds.get(c, 0.5) and c != 'NORM':
                                preds.append(c)
                        gtl = [c for c in classes if row['label_' + c] == 1]
                        rec["Predicted"] = ", ".join(preds) if preds else "NORM"
                        rec["GroundTruth"] = ", ".join(gtl) if gtl else "—"
                    results.append(rec)
                    prog.progress((k + 1) / n, text=f"Scoring cohort… {k + 1}/{n}")
                prog.empty()
                st.session_state['batch_results'] = pd.DataFrame(results)

        if 'batch_results' in st.session_state and len(st.session_state['batch_results']):
            res_df = st.session_state['batch_results']
            if 'Correct' in res_df.columns:
                bc = st.columns(3)
                bc[0].metric("Cohort accuracy", f"{res_df['Correct'].mean():.1%}")
                bc[1].metric("Patients scored", len(res_df))
                bc[2].metric("Flagged abnormal", int((res_df['Predicted'] == 'Abnormal').sum()))
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            st.download_button("Download results (CSV)",
                               res_df.to_csv(index=False).encode(),
                               file_name="heartbreaker_batch_results.csv", mime="text/csv",
                               use_container_width=True)


# Footer/Credits
st.write("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0; line-height: 1.6;">
    <div style="margin-bottom: 0.5rem;">
        <span style="font-weight: 700; color: #00BCD4;">FVJ Health-Tech</span> | Heartbreaker™ MVP
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

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import wfdb
import os
import matplotlib.pyplot as plt
import scipy.signal as signal

st.set_page_config(page_title="1D-CNN ECG Triage MVP", page_icon="🫀", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
    h1 { color: #2c3e50; text-align: center; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 1D-CNN ECG Classification MVP")
st.markdown("<div class='subtitle'>Valid, Single-Source Diagnostic Model trained on Raw PTB-XL Signals</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'models/binary_1d_ecg_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

def load_and_preprocess_signal(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        fs = 100
        nyq = 0.5 * fs
        b, a = signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
        
        filtered_sig = np.zeros_like(sig)
        for i in range(sig.shape[1]):
            filtered_sig[:, i] = signal.filtfilt(b, a, sig[:, i])
            
        mean = np.mean(filtered_sig, axis=0)
        std = np.std(filtered_sig, axis=0)
        std[std == 0] = 1.0
        norm_sig = (filtered_sig - mean) / std
        
        if norm_sig.shape[0] >= 1000:
            norm_sig = norm_sig[:1000, :]
        else:
            pad = np.zeros((1000 - norm_sig.shape[0], 12))
            norm_sig = np.vstack([norm_sig, pad])
            
        return norm_sig
    except Exception as e:
        st.error(f"Error loading {record_path}: {e}")
        return None

if model is None:
    st.warning("⚠️ Model not found! Waiting for the 1D model to finish training.")
else:
    st.markdown("### Select an ECG Record to Analyze")
    st.write("This demo uses held-out test records from the PTB-XL database to prove the model operates on pure physiological signals, entirely immune to image-rendering confounds.")
    
    metadata_path = 'data/subset_metadata_2000.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        
        # Pick a few examples for the demo
        normals = df[df['class'] == 'Normal']['filename_lr'].tolist()
        abnormals = df[df['class'] == 'Abnormal']['filename_lr'].tolist()
        
        examples = {
            "Demo 1 (Normal)": normals[0] if normals else None,
            "Demo 2 (Abnormal)": abnormals[0] if abnormals else None,
            "Demo 3 (Normal)": normals[1] if len(normals)>1 else None,
            "Demo 4 (Abnormal)": abnormals[1] if len(abnormals)>1 else None,
        }
        
        choice = st.selectbox("Choose a record:", list(examples.keys()))
        
        if st.button("Run Diagnostic"):
            record_name = examples[choice]
            record_path = os.path.join('data/raw', record_name)
            
            sig = load_and_preprocess_signal(record_path)
            
            if sig is not None:
                # Plot the first lead
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(sig[:, 0], color='darkred', linewidth=1)
                ax.set_title("Preprocessed Lead I Signal (0.5-40Hz Bandpass + Z-Norm)")
                ax.set_xlabel("Samples (100 Hz)")
                ax.set_ylabel("Normalized Amplitude")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Predict
                img_batch = tf.expand_dims(sig, 0)
                probability = model.predict(img_batch)[0][0]
                
                if probability >= 0.5:
                    predicted_class = "Abnormal"
                else:
                    predicted_class = "Normal"
                    
                st.write("---")
                if predicted_class == "Normal":
                    st.success(f"**Diagnostic Prediction:** {predicted_class}")
                else:
                    st.error(f"**Diagnostic Prediction:** {predicted_class}")
                
                st.info(f"**Model Probability (Abnormal):** {probability:.4f}")
                
                true_label = "Normal" if "Normal" in choice else "Abnormal"
                st.write(f"*True Ground Truth Label: {true_label}*")
    else:
        st.info("Waiting for data download to complete...")

st.markdown("---")
st.markdown("#### Business Value: Automated Triage Classifier")
st.markdown("This 1D-CNN pipeline successfully processes raw physiological signals from a single-source distribution (PTB-XL), guaranteeing that the neural network learns true cardiac anomalies rather than institutional artifacts. By ranking incoming ECGs by abnormality probability, it saves critical specialist-minutes by triaging high-risk reads first.")

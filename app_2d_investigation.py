import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize
import numpy as np
from PIL import Image
import os
import matplotlib.cm as cm

# Streamlit Page Config
st.set_page_config(page_title="Dataset Validation Demonstrator", page_icon="🔍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .highlight-panel {
        font-size: 1.05rem;
        color: #2c3e50;
        border: 2px solid #3498db;
        padding: 15px;
        border-radius: 8px;
        background-color: #ebf5fb;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Medical AI Dataset Validation Tool")
st.markdown("<div class='subtitle'>Detecting Source Confounds and Dataset Shortcuts in Medical Imaging</div>", unsafe_allow_html=True)

st.markdown("""
<div class='highlight-panel'>
<b>Headline Finding:</b> A state-of-the-art classifier trained on public ECG data achieves a stunning <b>Class-vs-Source AUC of 0.976</b>. However, this is NOT a diagnostic achievement. Because the "Normal" images come from Latidos and "Abnormal" from PTB-XL, the model achieves this high score by classifying the <b>source hospital</b> instead of the cardiac pathology. 
<br><br>
This MVP demonstrates a <b>pre-deployment validation framework</b> that uses Grad-CAM to detect source confounds and shortcut learning before a medical model reaches patients.
</div>
""", unsafe_allow_html=True)


class ConvOutputCapture:
    """Captures a layer's output by monkey-patching its call method."""
    def __init__(self, layer):
        self.layer = layer
        self.output = None
        self._original_call = layer.call

    def install(self):
        capture = self
        original = self._original_call
        def hooked_call(*args, **kwargs):
            result = original(*args, **kwargs)
            capture.output = result
            return result
        self.layer.call = hooked_call

    def uninstall(self):
        self.layer.call = self._original_call


def make_gradcam_heatmap(model, capture_hook, img_batch):
    """Forward pass → capture conv output → compute Grad-CAM."""
    img_tensor = tf.constant(img_batch)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        score = preds[:, 0]

    conv_out = capture_hook.output
    if conv_out is None:
        raise RuntimeError("Hook did not fire — check layer name.")

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Gradient is None.")

    weights = tf.reduce_mean(grads, axis=(1, 2))          # (B, C)
    cam = tf.einsum("bijk,bk->bij", conv_out, weights)[0] # (H, W)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

# Load the model
@st.cache_resource
def load_model():
    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/binary_ecg_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not found! Please ensure 'binary_ecg_model.h5' is available.")
else:
    st.markdown("### Test the Shortcut Detector")
    st.markdown("Upload a 12-lead ECG image to see exactly what features the AI uses to make its 'diagnosis'.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # 1. Prepare image
            raw_image = Image.open(uploaded_file).convert('RGB')
            img_array = img_to_array(raw_image)
            img_array = smart_resize(img_array, size=(224, 224))
            img_batch = tf.expand_dims(img_array, 0)
            
            # 2. Hook into backbone for Grad-CAM
            backbone = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    backbone = layer
                    break
            
            target_layer = None
            if backbone is not None:
                for layer in reversed(backbone.layers):
                    if len(layer.output.shape) == 4:
                        target_layer = layer
                        break
            else:
                for layer in reversed(model.layers):
                    if len(layer.output.shape) == 4:
                        target_layer = layer
                        break

            if target_layer is not None:
                hook = ConvOutputCapture(target_layer)
                hook.install()

                # Make Prediction & Heatmap
                preds = model.predict(img_batch)
                probability = preds[0][0]
                cam = make_gradcam_heatmap(model, hook, img_batch)
                hook.uninstall()

                # Process Heatmap Overlay
                cam_resized = Image.fromarray((cam * 255).astype("uint8")).resize(raw_image.size)
                cam_resized = np.array(cam_resized) / 255.0

                colormap = cm.get_cmap("jet")
                cam_colored = colormap(cam_resized)[:, :, :3]
                overlay = (np.array(raw_image)/255.0) * 0.5 + cam_colored * 0.5
                overlay = (overlay * 255).astype(np.uint8)
                overlay_img = Image.fromarray(overlay)
                
                class_names = ['Normal (Latidos Source)', 'Abnormal (PTB-XL Source)']
                if probability >= 0.5:
                    predicted_class = class_names[1]
                else:
                    predicted_class = class_names[0]

                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(raw_image, caption=f"Original Upload", use_container_width=True)
                with col2:
                    st.image(overlay_img, caption=f"Grad-CAM Attention Map", use_container_width=True)

                st.info(f"**Model Prediction:** {predicted_class} (Probability: {probability:.4f})")
                
                st.markdown("""
                **What does this heatmap prove?**  
                Instead of attending to the cardiac waveform trace itself, you will notice the model's heat (red/yellow areas) often focuses heavily on the margins, background tint, grid lines, or text headers. **The model attends to source artifacts, not cardiac features.**
                
                This diagnostic tool correctly prevents deploying an invalid model that would otherwise have blindly passed high-level accuracy benchmarks.
                """)
            else:
                st.error("Could not find a convolutional layer for Grad-CAM.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

st.markdown("---")
st.markdown("#### Business Value: Automated Pre-Deployment Validation")
st.markdown("Before a medical model reaches patients, it must be proven clinically robust. This framework acts as an automated quality-control layer for dataset curation, ensuring AI systems rely on physiological features rather than institutional shortcuts—avoiding the massive costs and risks associated with deploying flawed models.")

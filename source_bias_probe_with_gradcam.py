import os, collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Directories containing the split data – adjust if your layout changes
SPLITS = {
    "train": "dataset_split/train",
    "val": "dataset_split/val",
    "test": "dataset_split/test",
}

# Path to the trained EfficientNetB0 model (produced by train_binary_ecg_model.py)
MODEL_PATH = "binary_ecg_model.h5"
# Name of the last convolutional layer in EfficientNetB0 – used for Grad‑CAM
# Dynamically find last Conv2D layer name
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
last_conv = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv = layer.name
if last_conv is None:
    raise ValueError('No Conv2D layer found in model')
LAST_CONV_LAYER = last_conv  # use this for Grad‑CAMtNetB0's default last conv name

# Output directories for diagnostics
GRADCAM_OUT = "gradcam_outputs"
os.makedirs(GRADCAM_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# CHECK 1 – Image dimensions / format statistics per class
# ---------------------------------------------------------------------------
stats = collections.defaultdict(list)
for split, root in SPLITS.items():
    for cls in ("Normal", "Abnormal"):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            try:
                im = Image.open(os.path.join(d, fn))
                stats[cls].append((im.size[0], im.size[1], im.format, im.mode))
            except Exception:
                pass

print("--- CHECK 1: Dimension / format stats ---")
for cls in ("Normal", "Abnormal"):
    s = stats[cls]
    if not s:
        print(f"{cls}: no images found")
        continue
    ws = [x[0] for x in s]
    hs = [x[1] for x in s]
    fmts = collections.Counter(x[2] for x in s)
    print(f"{cls}: n={len(s)} | width {min(ws)}-{max(ws)} (med {int(np.median(ws))}) "
          f"| height {min(hs)}-{max(hs)} (med {int(np.median(hs))}) | formats {dict(fmts)}")

# ---------------------------------------------------------------------------
# CHECK 2 – Metadata‑only classifier (dimensions only)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = [], []
for cls, label in (("Normal", 0), ("Abnormal", 1)):
    for (w, h, fmt, mode) in stats[cls]:
        X.append([w, h, w / h])
        y.append(label)
X = np.array(X)
y = np.array(y)
if len(X) > 0:
    auc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="roc_auc").mean()
    print("\n--- CHECK 2: Metadata‑only classifier ---")
    print(f"Metadata‑only (dimensions) AUC: {auc:.3f}")
    print("  >0.65 => source shortcut confirmed. The CNN's 0.96 is mostly this.")
    print("  ~0.50 => dimensions are clean; shortcut must be checked via Grad‑CAM.")
else:
    print("No data to compute metadata‑only AUC.")

# ---------------------------------------------------------------------------
# CHECK 3 – Grad‑CAM overlay on a few test images
# ---------------------------------------------------------------------------

def get_img_array(img_path, size):
    """Load image, resize to model input size and convert to a 4‑D array."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    array = np.array(img) / 255.0
    return np.expand_dims(array, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate a Grad‑CAM heatmap for `img_array`.
    Returns a 2‑D (height, width) heatmap normalized to [0, 1]."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap onto the original image and return a PIL image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)

print("\n--- CHECK 3: Grad‑CAM on sample test images ---")
# Load model (ensure the path is correct)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Choose a few test images (one per class if possible)
sample_paths = []
for cls in ("Normal", "Abnormal"):
    cls_dir = os.path.join(SPLITS["test"], cls)
    if os.path.isdir(cls_dir):
        files = os.listdir(cls_dir)
        if files:
            sample_paths.append(os.path.join(cls_dir, files[0]))
# Fallback: just grab any two images from the test split
if len(sample_paths) < 2:
    for root, _, files in os.walk(SPLITS["test"]):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_paths.append(os.path.join(root, f))
                if len(sample_paths) >= 2:
                    break
        if len(sample_paths) >= 2:
            break

input_size = (224, 224)
for img_path in sample_paths:
    arr = get_img_array(img_path, input_size)
    heatmap = make_gradcam_heatmap(arr, model, LAST_CONV_LAYER)
    overlay = superimpose_heatmap(img_path, heatmap)
    out_name = os.path.join(GRADCAM_OUT, os.path.basename(img_path))
    overlay.save(out_name)
    print(f"Grad‑CAM saved: {out_name}")

print("\nAll checks completed. Review the printed statistics and the images in "
      f"'{GRADCAM_OUT}' for visual evidence of source bias.")

"""Grad-CAM visualisation for the binary ECG classifier.

Uses a Keras 3-compatible approach: instead of building a multi-output
Model (which fails across nested sub-Model boundaries), we register a
lightweight hook on the target conv layer to capture its output during
the normal forward pass.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)
OUT_DIR  = "eval_outputs/gradcam"


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
        raise RuntimeError("Gradient is None — the hook output is not "
                           "on the tape. This shouldn't happen.")

    weights = tf.reduce_mean(grads, axis=(1, 2))          # (B, C)
    cam = tf.einsum("bijk,bk->bij", conv_out, weights)[0] # (H, W)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_and_save(img_path, cam, pred, true_cls, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    raw = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    cam_up = np.array(
        Image.fromarray((cam * 255).astype("uint8")).resize(IMG_SIZE)
    ) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].imshow(raw)
    axes[0].set_title(f"Original  [{true_cls}]")
    axes[0].axis("off")

    axes[1].imshow(raw)
    axes[1].imshow(cam_up, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Grad-CAM  P(Abn)={pred:.3f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load --------------------------------------------------------------
    model = tf.keras.models.load_model(args.model, compile=False)

    # ---- find backbone & last conv -----------------------------------------
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    assert backbone is not None, "No nested backbone found."

    target_layer = None
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            target_layer = layer
            break
    assert target_layer is not None, "No 4D layer in backbone."
    print(f"Target conv layer: {target_layer.name}  "
          f"shape={target_layer.output.shape}")

    # ---- install hook -------------------------------------------------------
    hook = ConvOutputCapture(target_layer)
    hook.install()

    # ---- generate overlays --------------------------------------------------
    from PIL import Image
    for cls in ("Normal", "Abnormal"):
        d = os.path.join(args.images, cls)
        if not os.path.isdir(d):
            print(f"  Skipping {cls}/ (not found)")
            continue
        files = sorted(
            f for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:args.n]
        for fn in files:
            img_path = os.path.join(d, fn)
            arr = np.array(
                Image.open(img_path).convert("RGB").resize(IMG_SIZE),
                dtype="float32",
            )
            batch = preprocess_input(arr[None, ...])
            pred  = float(model.predict(batch, verbose=0).ravel()[0])

            # Grad-CAM (separate call so hook + tape align)
            cam = make_gradcam_heatmap(model, hook, batch)

            out = os.path.join(OUT_DIR,
                               f"{cls}_{os.path.splitext(fn)[0]}.png")
            overlay_and_save(img_path, cam, pred, cls, out)
            print(f"  {cls}/{fn}:  P(Abn)={pred:.3f}  →  {out}")

    hook.uninstall()
    print(f"\nDone — overlays in {OUT_DIR}/")
    print("Look at where the heat lands:")
    print("  • On waveform traces → model may be reading real ECG morphology")
    print("  • On grid/margins/text → second shortcut confirmed")


if __name__ == "__main__":
    main()

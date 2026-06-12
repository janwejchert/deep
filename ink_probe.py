import argparse, os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data_clean", help="root with {split}/{class}/")
args = ap.parse_args()

X, y = [], []
counts = {"Normal": 0, "Abnormal": 0}
for cls, lab in (("Normal", 0), ("Abnormal", 1)):
    for split in ("train", "val", "test"):
        d = os.path.join(args.root, split, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(EXT):      # skip _sample_sheet.png etc.
                continue
            try:
                a = np.array(Image.open(os.path.join(d, fn)).convert("L"))
                X.append([(a < 128).mean()])      # fraction of black pixels
                y.append(lab)
                counts[cls] += 1
            except Exception as e:
                print(f"  skip {fn}: {e}")

print(f"loaded: {counts}")
if counts["Normal"] == 0 or counts["Abnormal"] == 0:
    raise SystemExit(f"One class has 0 images. Check --root path: {args.root}")

X, y = np.array(X), np.array(y)
auc = cross_val_score(LogisticRegression(max_iter=1000), X, y,
                      cv=5, scoring="roc_auc").mean()
print(f"\nink-density-only AUC: {auc:.3f}")
print("  >0.65  -> binarization created/sharpened a trace-density shortcut; 0.98 is mostly that")
print("  ~0.50  -> density is clean; the 0.98 is more likely real (still caveated by 26 normals)")

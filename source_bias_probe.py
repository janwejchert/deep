import os, collections
import numpy as np
from PIL import Image

# Adjust paths to match project structure
SPLITS = {
    "train": "data_clean/train",
    "val": "data_clean/val",
    "test": "data_clean/test"
}

# CHECK 1: do image dimensions/format differ by CLASS?
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

# CHECK 2: metadata-only classifier
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
    print(f"\nMetadata-only (dimensions) AUC: {auc:.3f}")
    print("  >0.65 => source shortcut confirmed. The CNN's 0.96 is mostly this.")
    print("  ~0.50 => dimensions are clean; shortcut must be checked via Grad-CAM.")
else:
    print("No data to compute metadata-only AUC.")

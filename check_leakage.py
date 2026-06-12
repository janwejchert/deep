"""
check_leakage.py
================
Detects probable patient-level leakage in an image-level ECG split, since the
raw data lacks patient IDs. Strategy: near-duplicate detection via perceptual
hashing. Images that are near-duplicates are very likely the same patient (same
recording re-exported, adjacent strips, or augmentation artifacts) and MUST NOT
straddle train/val/test.

Outputs:
  1. Count of near-duplicate clusters whose members fall in >1 split (leakage).
  2. A leakage severity estimate (fraction of test images with a near-duplicate
     in train).
  3. A suggested patient-disjoint regrouping (cluster-level assignment) you can
     use to rebuild the split and re-run evaluate_model.py.
"""

import os
from collections import defaultdict

from PIL import Image
import imagehash

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SPLIT_DIRS = {
    "train": "/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/train",
    "val":   "/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/val",
    "test":  "/Users/felipedeleon/Desktop/Deep Ler,Project/data_clean/test",
}
HASH_SIZE = 16          # higher = more sensitive to fine detail (8 is coarse)
HAMMING_THRESHOLD = 6   # <= this Hamming distance counts as a near-duplicate.
                        # 0 = exact dup; 5-8 = near-dup (same patient likely).
OUT_DIR = "/Users/felipedeleon/Desktop/Deep Ler,Project/docs"
# ----------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images():
    records = []  # (path, split, class, phash)
    for split, root in SPLIT_DIRS.items():
        for cls in ("Abnormal", "Normal"):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(IMG_EXT):
                    continue
                p = os.path.join(d, fn)
                try:
                    h = imagehash.phash(Image.open(p).convert("L"),
                                        hash_size=HASH_SIZE)
                except Exception as e:
                    print(f"  skip {p}: {e}")
                    continue
                records.append(dict(path=p, split=split, cls=cls, phash=h))
    return records


def cluster_near_duplicates(records):
    """Union-Find over images within HAMMING_THRESHOLD of each other."""
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # O(n^2) pairwise -- fine for ~1000 images.
    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["phash"] - records[j]["phash"] <= HAMMING_THRESHOLD:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return [idxs for idxs in clusters.values()]


def main():
    print("=" * 70)
    print("PATIENT-LEAKAGE CHECK (near-duplicate detection)")
    print("=" * 70)
    records = collect_images()
    print(f"Indexed {len(records)} images across "
          f"{len(SPLIT_DIRS)} splits.")

    clusters = cluster_near_duplicates(records)
    multi = [c for c in clusters if len(c) > 1]
    print(f"Near-duplicate clusters (size>1): {len(multi)}")

    # Leakage: clusters spanning more than one split
    leaky = []
    for c in multi:
        splits = {records[i]["split"] for i in c}
        if len(splits) > 1:
            leaky.append((c, splits))

    print(f"\nClusters straddling >1 split (LEAKAGE): {len(leaky)}")
    for c, splits in leaky[:20]:
        print(f"  size {len(c)} across {sorted(splits)}:")
        for i in c:
            print(f"     [{records[i]['split']:<5}] "
                  f"{records[i]['cls']:<8} {records[i]['path']}")
    if len(leaky) > 20:
        print(f"  ... and {len(leaky) - 20} more")

    # Severity: fraction of TEST images that have a near-dup in TRAIN
    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]
    cluster_of = {}
    for cid, c in enumerate(clusters):
        for i in c:
            cluster_of[i] = cid
    test_with_train_dup = 0
    for i in test_idx:
        mates = clusters[cluster_of[i]]
        if any(records[j]["split"] == "train" for j in mates if j != i):
            test_with_train_dup += 1
    frac = test_with_train_dup / max(len(test_idx), 1)
    print(f"\nTEST images with a near-duplicate in TRAIN: "
          f"{test_with_train_dup}/{len(test_idx)} ({frac:.1%})")
    if frac > 0.05:
        print("  >> Material leakage. The AUC is likely inflated. "
              "Regroup and re-evaluate before any clinical-range claim.")
    elif len(leaky) == 0:
        print("  >> No cross-split near-duplicates found at this threshold. "
              "The split is clean w.r.t. duplicate-based leakage "
              "(does NOT rule out same-patient-different-image leakage, but "
              "is strong evidence).")

    # Suggested patient-disjoint regrouping
    print(f"\nWriting suggested patient-disjoint split to "
          f"{OUT_DIR}/regroup_suggestion.csv ...")
    import csv, random
    random.seed(42)
    cluster_ids = list(range(len(clusters)))
    random.shuffle(cluster_ids)
    n_clu = len(cluster_ids)
    bounds = (int(0.70 * n_clu), int(0.85 * n_clu))
    assign = {}
    for rank, cid in enumerate(cluster_ids):
        split = ("train" if rank < bounds[0]
                 else "val" if rank < bounds[1] else "test")
        for i in clusters[cid]:
            assign[i] = split
    with open(os.path.join(OUT_DIR, "regroup_suggestion.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "old_split", "new_split", "cluster_id"])
        for i, r in enumerate(records):
            w.writerow([r["path"], r["cls"], r["split"],
                        assign[i], cluster_of[i]])
    print("Done. Rebuild folders from new_split, then re-run "
          "evaluate_model.py for the patient-disjoint AUC.")


if __name__ == "__main__":
    main()

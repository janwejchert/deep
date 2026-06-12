import os
import csv
import random
from collections import defaultdict
from PIL import Image
import imagehash
import shutil

DATA_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary'
SPLIT_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split'
OUT_CSV = '/Users/felipedeleon/Desktop/Deep Ler,Project/docs/split_assignment.csv'

HASH_SIZE = 16
HAMMING_THRESHOLD = 6

def main():
    records = []
    for cls in ("Normal", "Abnormal"):
        d = os.path.join(DATA_DIR, cls)
        if not os.path.exists(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(d, fn)
            try:
                with Image.open(p) as img:
                    h = imagehash.phash(img.convert("L"), hash_size=HASH_SIZE)
                records.append({'path': p, 'cls': cls, 'phash': h})
            except Exception as e:
                print(f"Failed to hash {p}: {e}")

    print(f"Indexed {len(records)} images from {DATA_DIR}.")

    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["phash"] - records[j]["phash"] <= HAMMING_THRESHOLD:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    
    cluster_list = list(clusters.values())
    print(f"Found {len(cluster_list)} unique clusters.")

    # Assign clusters to train/val/test
    random.seed(42)
    cluster_ids = list(range(len(cluster_list)))
    random.shuffle(cluster_ids)
    
    n_clu = len(cluster_ids)
    bounds = (int(0.70 * n_clu), int(0.85 * n_clu))
    assign = {}
    for rank, cid in enumerate(cluster_ids):
        split = "train" if rank < bounds[0] else "val" if rank < bounds[1] else "test"
        for i in cluster_list[cid]:
            assign[i] = split

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "split"])
        for i, r in enumerate(records):
            w.writerow([r["path"], r["cls"], assign[i]])

    print("Rebuilding split directories...")
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    
    for split in ('train', 'val', 'test'):
        for cls in ('Normal', 'Abnormal'):
            os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

    moved = 0
    for i, r in enumerate(records):
        src = r['path']
        cls = r['cls']
        fname = os.path.basename(src)
        split = assign[i]
        dst = os.path.join(SPLIT_DIR, split, cls, fname)
        shutil.copy2(src, dst)
        moved += 1

    print(f"Moved {moved} files to {SPLIT_DIR}.")

    for split in ('train', 'val', 'test'):
        for cls in ('Normal', 'Abnormal'):
            d = os.path.join(SPLIT_DIR, split, cls)
            n_files = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/{cls}: {n_files}")

if __name__ == "__main__":
    main()

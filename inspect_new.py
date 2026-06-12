import os
import numpy as np

base_dir = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted"

def walk_dir(path, max_depth=3, depth=0):
    if depth > max_depth:
        return
    indent = "  " * depth
    try:
        items = os.listdir(path)
    except Exception as e:
        print(f"{indent}Error listing {path}: {e}")
        return
        
    for item in sorted(items):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"{indent}[DIR] {item}")
            walk_dir(item_path, max_depth, depth + 1)
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"{indent}[FILE] {item} ({size_mb:.2f} MB)")

print("=== Directory Structure of dataset_new_extracted ===")
walk_dir(base_dir)

# Try loading npz files if they exist
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.npz'):
            p = os.path.join(root, f)
            print(f"\nFound npz file: {p}")
            try:
                data = np.load(p, allow_pickle=True)
                keys = list(data.keys())
                print(f"  Keys (total {len(keys)}): {keys[:10]} ...")
                for k in keys[:3]:
                    val = data[k]
                    print(f"  Key {k} shape: {val.shape if hasattr(val, 'shape') else type(val)}")
            except Exception as e:
                print(f"  Error loading {f}: {e}")

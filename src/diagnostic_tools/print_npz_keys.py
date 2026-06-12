import numpy as np

p = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted/final_data/data/leads.npz"
data = np.load(p, allow_pickle=True)
keys = sorted(list(data.keys()))
print(f"Total keys: {len(keys)}")
print("First 50 keys:")
for k in keys[:50]:
    print(" ", k)

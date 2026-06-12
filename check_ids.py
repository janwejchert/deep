import numpy as np

p = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted/final_data/data/leads.npz"
data = np.load(p, allow_pickle=True)
keys = list(data.keys())

ids = set()
for k in keys:
    # Example key: high_freq_noise_low_LPAE_15529_hr
    parts = k.split('_')
    for part in parts:
        if part.isdigit():
            ids.add(part)

print(f"Total keys: {len(keys)}")
print(f"Total unique IDs: {len(ids)}")
print(f"Unique IDs: {sorted(list(ids))}")

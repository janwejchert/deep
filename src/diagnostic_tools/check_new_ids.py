import os

scans_dir = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted/final_data/visual_data/photos_scans"
files = sorted(os.listdir(scans_dir))
image_ids = set()
for f in files:
    if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg'):
        # Name format: img_X_page_Y.jpeg
        parts = f.split('_')
        if len(parts) >= 2:
            image_ids.add(int(parts[1]))

print(f"Total files in photos_scans: {len(files)}")
print(f"Total unique image IDs: {len(image_ids)}")
print(f"Min image ID: {min(image_ids)}")
print(f"Max image ID: {max(image_ids)}")
print(f"Sorted unique image IDs: {sorted(list(image_ids))}")

import os
import shutil
import json
import uuid

visual_data_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted/final_data/visual_data'
target_abnormal = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary/Abnormal'
target_normal = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary/Normal'

# Load mapping
with open('/Users/felipedeleon/Desktop/Deep Ler,Project/img_to_label.json', 'r') as f:
    img_to_label = json.load(f)

count = 0
for root, dirs, files in os.walk(visual_data_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # filename is like img_1_page_0.jpeg
            parts = file.split('_')
            if len(parts) >= 2 and parts[0] == 'img':
                img_id = parts[1]
                label = img_to_label.get(img_id, 'Abnormal') # We know they are all Abnormal anyway
                
                # Create a unique name to avoid collisions across augmentation folders
                aug_folder = os.path.basename(root)
                new_filename = f"new_{aug_folder}_{file}"
                
                src_path = os.path.join(root, file)
                if label == 'Normal':
                    dst_path = os.path.join(target_normal, new_filename)
                else:
                    dst_path = os.path.join(target_abnormal, new_filename)
                    
                shutil.copy2(src_path, dst_path)
                count += 1

print(f"Successfully integrated {count} new images into dataset_binary.")

import os
import csv
import ast
import pandas as pd
import requests

# 1. Load metadata.csv to map image_id to PTB-XL ID
metadata_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/metadata.csv'
img_to_ptb = {}
with open(metadata_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_id = row['Image ID']
        ecg_id = row['ECG ID']
        # e.g. LPAE_11666_hr -> 11666
        ptb_id = ecg_id.split('_')[1]
        img_to_ptb[img_id] = ptb_id

print(f"Loaded {len(img_to_ptb)} unique image mappings from metadata.csv")

# 2. Download ptbxl_database.csv
ptb_url = 'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
ptb_csv_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/ptbxl_database.csv'
if not os.path.exists(ptb_csv_path):
    print("Downloading ptbxl_database.csv...")
    r = requests.get(ptb_url)
    with open(ptb_csv_path, 'wb') as f:
        f.write(r.content)

# 3. Read PTB-XL metadata to get labels
print("Parsing PTB-XL labels...")
df = pd.read_csv(ptb_csv_path)

# 4. We need to map diagnostic codes to superclasses. 
# Usually, NORM is normal. MI, STTC, CD, HYP are abnormal.
# In the user's project: Normal vs Abnormal (MI, HB, etc.)
# Let's map NORM -> Normal, anything else -> Abnormal.
# Actually, wait. We need the superclass mapping. 
# In PTB-XL, scp_codes contains the diagnostic statements.
# Let's download scp_statements.csv to map statements to superclass, or just assume if 'NORM' is present and it's the only one or primary.
scp_url = 'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv'
scp_csv_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/scp_statements.csv'
if not os.path.exists(scp_csv_path):
    r = requests.get(scp_url)
    with open(scp_csv_path, 'wb') as f:
        f.write(r.content)

scp_df = pd.read_csv(scp_csv_path)
scp_df = scp_df[scp_df.diagnostic == 1]
diag_to_class = dict(zip(scp_df.index, scp_df.diagnostic_class))

# Parse labels for our 100 ptb_ids
ptb_to_label = {}
for index, row in df.iterrows():
    ecg_id = str(row['ecg_id']).zfill(5)
    if ecg_id in img_to_ptb.values():
        # Parse scp_codes
        try:
            codes = ast.literal_eval(row['scp_codes'])
        except:
            codes = {}
        
        # Determine superclass
        superclasses = set()
        for code in codes.keys():
            if code in diag_to_class:
                superclasses.add(diag_to_class[code])
        
        if 'NORM' in superclasses and len(superclasses) == 1:
            label = 'Normal'
        elif len(superclasses) > 0 and 'NORM' not in superclasses:
            label = 'Abnormal'
        else:
            # If mixed or empty, check if 'NORM' is present
            if 'NORM' in superclasses:
                label = 'Normal' # Or maybe mixed? We will see.
            else:
                label = 'Abnormal'
        ptb_to_label[ecg_id] = label

# Check how many we mapped
img_to_label = {}
for img_id, ptb_id in img_to_ptb.items():
    if ptb_id in ptb_to_label:
        img_to_label[img_id] = ptb_to_label[ptb_id]

normals = sum(1 for l in img_to_label.values() if l == 'Normal')
abnormals = sum(1 for l in img_to_label.values() if l == 'Abnormal')
print(f"Mapped {len(img_to_label)} images: {normals} Normal, {abnormals} Abnormal")

# Write out the mapping for future reference
import json
with open('/Users/felipedeleon/Desktop/Deep Ler,Project/img_to_label.json', 'w') as f:
    json.dump(img_to_label, f, indent=2)

print("Mapping saved to img_to_label.json")

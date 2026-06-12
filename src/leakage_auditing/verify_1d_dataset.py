import pandas as pd
import ast

df = pd.read_csv('data/subset_metadata_2000.csv')

print(f"Total records: {len(df)}")
print(f"Unique ecg_ids: {df['ecg_id'].nunique()}")
print(f"Unique patient_ids: {df['patient_id'].nunique()}")
print(f"Class counts:\n{df['class'].value_counts()}")

# Check for duplicates
duplicates = df[df.duplicated(subset=['patient_id'], keep=False)]
print(f"Duplicate patient records: {len(duplicates)}")

# Verify label correctness
label_errors = 0
for idx, row in df.iterrows():
    try:
        codes = ast.literal_eval(row['scp_codes'])
    except:
        codes = {}
    
    is_normal = 'NORM' in codes
    expected_class = 'Normal' if is_normal else 'Abnormal'
    if row['class'] != expected_class:
        print(f"Label mismatch at index {idx}: id={row['ecg_id']}, codes={codes}, class={row['class']}")
        label_errors += 1

print(f"Label check completed. Mismatches found: {label_errors}")

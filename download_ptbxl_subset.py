import pandas as pd
import ast
import os
import urllib.request
from tqdm import tqdm

def get_class(scp_codes_dict):
    if 'NORM' in scp_codes_dict and scp_codes_dict['NORM'] > 50:
        is_abnormal = False
        for code, conf in scp_codes_dict.items():
            if code != 'NORM' and conf > 50:
                is_abnormal = True
        if not is_abnormal:
            return 'Normal'
            
    if 'NORM' not in scp_codes_dict:
        return 'Abnormal'
    return 'Unknown'

def main():
    print("Reading PTB-XL database metadata...")
    df = pd.read_csv('ptbxl_database.csv')
    
    df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    df['class'] = df['scp_codes_dict'].apply(get_class)
    
    # Globally drop duplicate patient_id to guarantee no patient exists in both classes
    df = df.drop_duplicates(subset=['patient_id'])
    
    normals = df[df['class'] == 'Normal']
    abnormals = df[df['class'] == 'Abnormal']
    
    # Tiny dataset for fast MVP (100 each)
    N_SAMPLES = 100
    normals = normals.sample(n=N_SAMPLES, random_state=42)
    abnormals = abnormals.sample(n=N_SAMPLES, random_state=42)
        
    subset = pd.concat([normals, abnormals]).reset_index(drop=True)
    print(f"Downloading {len(subset)} records (100Hz)...")
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    out_dir = "dataset_1d/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    subset.to_csv("dataset_1d/subset_metadata.csv", index=False)
    
    for i, row in tqdm(subset.iterrows(), total=len(subset)):
        filename_lr = row['filename_lr']
        file_path = os.path.dirname(filename_lr)
        os.makedirs(os.path.join(out_dir, file_path), exist_ok=True)
        
        for ext in ['.dat', '.hea']:
            url = f"{base_url}{filename_lr}{ext}"
            dest = os.path.join(out_dir, f"{filename_lr}{ext}")
            
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

if __name__ == '__main__':
    main()

import pandas as pd
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(url, dest):
    if not os.path.exists(dest):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            # Physionet rate limiting or 404
            return False
    return True

def main():
    print("Reading PTB-XL database metadata...")
    meta = pd.read_csv('ptbxl_database.csv')
    
    # Load the evaluation set
    eval_meta = pd.read_csv('dataset_1d/subset_metadata.csv')
    eval_patients = set(eval_meta['patient_id'].unique())
    
    # Filter out eval patients
    pretrain_meta = meta[~meta.patient_id.isin(eval_patients)].copy()
    
    # Hard assertion
    assert len(set(pretrain_meta.patient_id) & eval_patients) == 0, "LEAKAGE: eval patients present in pretraining pool"
    print(f"Pretrain pool: {len(pretrain_meta)} records, {pretrain_meta.patient_id.nunique()} patients — disjoint from eval. OK")
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    out_dir = "dataset_1d/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    pretrain_meta.to_csv("dataset_1d/pretrain_metadata.csv", index=False)
    
    download_tasks = []
    
    for i, row in pretrain_meta.iterrows():
        filename_lr = row['filename_lr']
        if pd.isna(filename_lr):
            continue
            
        file_path = os.path.dirname(filename_lr)
        os.makedirs(os.path.join(out_dir, file_path), exist_ok=True)
        
        for ext in ['.dat', '.hea']:
            url = f"{base_url}{filename_lr}{ext}"
            dest = os.path.join(out_dir, f"{filename_lr}{ext}")
            download_tasks.append((url, dest))
            
    print(f"Total files to download: {len(download_tasks)}")
    
    # Download with 20 threads to speed up
    max_workers = 20
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, dest) for url, dest in download_tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                success_count += 1
                
    print(f"Successfully downloaded {success_count} / {len(download_tasks)} files.")

if __name__ == '__main__':
    main()

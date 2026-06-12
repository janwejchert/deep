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
            return False
    return True

def main():
    meta = pd.read_csv('data/subset_metadata_2000.csv')
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    download_tasks = []
    
    for i, row in meta.iterrows():
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

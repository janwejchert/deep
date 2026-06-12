import pandas as pd
import ast
import os
import urllib.request
from tqdm import tqdm

def main():
    print("Reading PTB-XL database metadata...")
    df = pd.read_csv('ptbxl_database.csv')
    scp_statements = pd.read_csv('scp_statements.csv', index_col=0)
    
    def get_superclasses(scp_codes_str):
        try:
            codes = ast.literal_eval(scp_codes_str)
        except:
            return []
        superclasses = set()
        for code, val in codes.items():
            if val > 0:
                if code in scp_statements.index:
                    scls = scp_statements.loc[code, 'diagnostic_class']
                    if pd.notna(scls):
                        superclasses.add(scls)
        return list(superclasses)

    df['superclasses'] = df['scp_codes'].apply(get_superclasses)
    
    # Drop duplicate patients globally to ensure 100% patient-disjoint split
    df_unique = df.drop_duplicates(subset=['patient_id']).copy()
    
    # Define Normal: strictly ['NORM']
    normals = df_unique[df_unique['superclasses'].apply(lambda x: x == ['NORM'])]
    
    # Define Abnormals strictly by subclass
    mi_only = df_unique[df_unique['superclasses'].apply(lambda x: x == ['MI'])]
    sttc_only = df_unique[df_unique['superclasses'].apply(lambda x: x == ['STTC'])]
    cd_only = df_unique[df_unique['superclasses'].apply(lambda x: x == ['CD'])]
    hyp_only = df_unique[df_unique['superclasses'].apply(lambda x: x == ['HYP'])]
    
    print(f"Available Normals: {len(normals)}")
    print(f"Available MI: {len(mi_only)}")
    print(f"Available STTC: {len(sttc_only)}")
    print(f"Available CD: {len(cd_only)}")
    print(f"Available HYP: {len(hyp_only)}")
    
    # Sample
    sampled_normals = normals.sample(n=100, random_state=42).copy()
    sampled_normals['class'] = 'Normal'
    sampled_normals['subclass'] = 'NORM'
    
    sampled_mi = mi_only.sample(n=25, random_state=42).copy()
    sampled_mi['class'] = 'Abnormal'
    sampled_mi['subclass'] = 'MI'
    
    sampled_sttc = sttc_only.sample(n=25, random_state=42).copy()
    sampled_sttc['class'] = 'Abnormal'
    sampled_sttc['subclass'] = 'STTC'
    
    sampled_cd = cd_only.sample(n=25, random_state=42).copy()
    sampled_cd['class'] = 'Abnormal'
    sampled_cd['subclass'] = 'CD'
    
    sampled_hyp = hyp_only.sample(n=25, random_state=42).copy()
    sampled_hyp['class'] = 'Abnormal'
    sampled_hyp['subclass'] = 'HYP'
    
    subset = pd.concat([sampled_normals, sampled_mi, sampled_sttc, sampled_cd, sampled_hyp]).reset_index(drop=True)
    
    # Double check patient overlap
    assert subset['patient_id'].nunique() == 200, "Error: Duplicate patient IDs found in subset!"
    assert len(subset) == 200, "Error: Subset size is not 200!"
    
    # Save metadata
    os.makedirs('data', exist_ok=True)
    subset.to_csv("data/subset_metadata.csv", index=False)
    print("Saved metadata to data/subset_metadata.csv")
    
    # Download missing files
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Downloading raw signals...")
    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
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
                    
    print("All downloads/verifications complete.")

if __name__ == '__main__':
    main()

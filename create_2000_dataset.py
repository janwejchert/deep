import pandas as pd
import ast
import os

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
    
    # 1000 Normal, 1000 Abnormal
    N_SAMPLES = 1000
    
    normals = normals.sample(n=N_SAMPLES, random_state=42)
    abnormals = abnormals.sample(n=N_SAMPLES, random_state=42)
        
    subset = pd.concat([normals, abnormals]).reset_index(drop=True)
    print(f"Selected {len(subset)} records (100Hz)...")
    
    subset.to_csv("dataset_1d/subset_metadata_2000.csv", index=False)
    print("Saved to dataset_1d/subset_metadata_2000.csv")

if __name__ == '__main__':
    main()

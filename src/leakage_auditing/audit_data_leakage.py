import pandas as pd
import os

print("=== DATA INTEGRITY AUDIT ===")

# 1. Check 1D Dataset Subset
if os.path.exists("data/subset_metadata_2000.csv"):
    df_1d = pd.read_csv("data/subset_metadata_2000.csv")
    print(f"\n[1D Dataset (Subset 2000)] Total records: {len(df_1d)}")
    n_unique_patients = df_1d['patient_id'].nunique()
    print(f"Unique patients: {n_unique_patients}")
    if len(df_1d) == n_unique_patients:
        print("✅ NO PATIENT OVERLAP: Every record is a unique patient.")
    else:
        print(f"❌ LEAKAGE DETECTED: {len(df_1d) - n_unique_patients} duplicate patients found!")
        
    print("\n[PTB-XL Literature Standard Check]")
    if 'strat_fold' in df_1d.columns:
        print("The dataset contains the official 'strat_fold' column (1-10) defined by Strodthoff et al.")
        print("Fold distribution in current subset:")
        print(df_1d['strat_fold'].value_counts().sort_index())
    else:
        print("Missing 'strat_fold'.")
else:
    print("data/subset_metadata_2000.csv not found.")

# 2. Check 2D Dataset (if it exists)
if os.path.exists("metadata.csv"):
    df_2d = pd.read_csv("metadata.csv")
    print(f"\n[Historical 2D Image Dataset] Total records: {len(df_2d)}")
    if 'patient_id' in df_2d.columns:
        n_unique_patients_2d = df_2d['patient_id'].nunique()
        print(f"Unique patients: {n_unique_patients_2d}")
        if len(df_2d) == n_unique_patients_2d:
            print("✅ NO PATIENT OVERLAP.")
        else:
            print(f"❌ LEAKAGE DETECTED: {len(df_2d) - n_unique_patients_2d} duplicate patients found!")
    else:
        print("patient_id column not found in 2D metadata!")

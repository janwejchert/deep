"""
LightGBM Multi-Label Classifier for Multi-Heartbreaker Pipeline.

Trains 5 independent LightGBM binary classifiers (one-vs-rest) on clinically
engineered features extracted from 12-lead ECGs. Uses patient-disjoint 5-Fold CV.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def get_feature_columns(df):
    """Get all feature columns (exclude labels, IDs, patient_id)."""
    exclude = {'ecg_id', 'patient_id'} | {f'label_{sc}' for sc in SUPERCLASSES}
    return [c for c in df.columns if c not in exclude]


def main():
    features_path = 'data/clinical_features.csv'
    
    if not os.path.exists(features_path):
        print(f"File {features_path} not found. Run extract_clinical_features.py first.")
        return
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} records with clinical features.")
    
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} clinical features.")
    
    X = df[feature_cols].values
    
    y = np.column_stack([df[f'label_{sc}'].values for sc in SUPERCLASSES])
    
    patient_ids = df['patient_id'].values
    
    # 5-Fold CV, stratified on the majority label for reasonable splits
    primary_label = np.argmax(y, axis=1)
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros_like(y, dtype=float)
    feature_importances = {sc: np.zeros(len(feature_cols)) for sc in SUPERCLASSES}
    
    print("\n=== Starting 5-Fold CV (One-vs-Rest LightGBM) ===\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, primary_label, groups=patient_ids)):
        print(f"--- Fold {fold+1}/5 ---")
        
        # Verify patient disjointness
        train_patients = set(patient_ids[train_idx])
        test_patients = set(patient_ids[test_idx])
        overlap = train_patients & test_patients
        if overlap:
            print(f"  ⚠️ WARNING: {len(overlap)} overlapping patients!")
        else:
            print(f"  ✅ Patient-disjoint split confirmed.")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit SimpleImputer on training data only to prevent leakage
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        for c_idx, class_name in enumerate(SUPERCLASSES):
            y_c_train = y_train[:, c_idx]
            y_c_test = y_test[:, c_idx]
            
            # Compute scale_pos_weight for class imbalance
            n_pos = np.sum(y_c_train)
            n_neg = len(y_c_train) - n_pos
            scale_pos = n_neg / max(n_pos, 1)
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'scale_pos_weight': scale_pos,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }
            
            train_data = lgb.Dataset(X_train, label=y_c_train)
            val_data = lgb.Dataset(X_test, label=y_c_test, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            probs = model.predict(X_test, num_iteration=model.best_iteration)
            oof_probs[test_idx, c_idx] = probs
            
            feature_importances[class_name] += model.feature_importance(importance_type='gain')
            
            try:
                auc = roc_auc_score(y_c_test, probs)
                print(f"  {class_name} Fold AUC: {auc:.4f}")
            except ValueError:
                print(f"  {class_name} Fold AUC: N/A")
    
    # Aggregate OOF results
    print("\n=====================================")
    print("LightGBM Aggregate OOF Multi-label Results:")
    
    os.makedirs('models', exist_ok=True)
    
    results = {}
    with open('models/lightgbm_multiclass_results.txt', 'w') as f:
        f.write("LightGBM Multi-Label OOF Results (5 Diagnostic Superclasses)\n")
        f.write("=============================================================\n")
        for c_idx, class_name in enumerate(SUPERCLASSES):
            try:
                auc = roc_auc_score(y[:, c_idx], oof_probs[:, c_idx])
                pr = average_precision_score(y[:, c_idx], oof_probs[:, c_idx])
                results[class_name] = {'roc_auc': auc, 'pr_auc': pr}
                print(f"[{class_name}] OOF ROC-AUC: {auc:.4f} | PR-AUC: {pr:.4f}")
                f.write(f"[{class_name}] OOF ROC-AUC: {auc:.4f} | PR-AUC: {pr:.4f}\n")
            except ValueError:
                pass
    
    np.save("models/lightgbm_oof_multiclass_probs.npy", oof_probs)
    
    # Save feature importances
    avg_importance = {}
    for sc in SUPERCLASSES:
        imp = feature_importances[sc] / 5  # average over folds
        top_indices = np.argsort(imp)[::-1][:15]
        avg_importance[sc] = [(feature_cols[i], float(imp[i])) for i in top_indices]
    
    with open('models/lightgbm_feature_importances.json', 'w') as f:
        json.dump(avg_importance, f, indent=2)
    
    print("\nTop 5 features per class:")
    for sc in SUPERCLASSES:
        top5 = avg_importance[sc][:5]
        print(f"  {sc}: {', '.join([f'{name} ({imp:.1f})' for name, imp in top5])}")
    
    print(f"\nSaved OOF probs to models/lightgbm_oof_multiclass_probs.npy")
    print(f"Saved results to models/lightgbm_multiclass_results.txt")
    print(f"Saved feature importances to models/lightgbm_feature_importances.json")


if __name__ == '__main__':
    main()

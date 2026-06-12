"""
multimodal_data_prep.py
-----------------------
Leakage-safe metadata feature engineering for the multimodal ECG model.

Usage (standalone audit):
    python multimodal_data_prep.py

Usage (from train_multimodal_ecg_model.py):
    from multimodal_data_prep import build_metadata_matrix

All transformations that could leak information (TF-IDF, imputer medians,
scaler means) are returned as *unfitted* callables — the training script
fits them inside each CV fold on training data only.
"""

import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Leakage audit: German diagnostic terms that directly encode the label ───
# Any TF-IDF term matching these patterns is DROPPED before training.
# Add more if you discover new leaking terms during the label-correlation audit.
LABEL_LEAKING_TERMS = re.compile(
    r"infarkt|myokard|ischemi|herzinfarkt|"
    r"linksschenkel|rechtsschenkel|schenkelblock|"
    r"hypertrophie|überlastung|linkstyp|vorhofflimmern|"
    r"st.elevation|st.senkung|t.negativierung|"
    r"avblock|av.block|vorhof|bradykardi|tachykardi",
    re.IGNORECASE,
)

HEART_AXIS_CATS = ["MID", "LAD", "ALAD", "RAD", "ARAD", "AXL", "AXR", "SAG", "unknown"]


# ────────────────────────────────────────────────────────────────────────────
# 1. Static transformations (no fit required — applied identically everywhere)
# ────────────────────────────────────────────────────────────────────────────

def _static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic, label-safe transformations to raw metadata.
    Returns a new DataFrame with clean numeric/binary columns.
    No fitting required — safe to apply to full dataset before CV split.
    """
    out = pd.DataFrame(index=df.index)

    # ── Demographics ──────────────────────────────────────────────────────
    # Age: clip physiologically impossible values (max in data = 300)
    out["age"] = df["age"].clip(1, 120).astype(float)

    # Sex: already binary 0/1
    out["sex"] = df["sex"].astype(float)

    # Height & weight: track missingness explicitly — missingness pattern
    # itself could correlate with label so we include a flag.
    out["height_missing"]  = df["height"].isna().astype(float)
    out["weight_missing"]  = df["weight"].isna().astype(float)
    out["height"]  = df["height"].astype(float)   # imputed later (inside CV)
    out["weight"]  = df["weight"].astype(float)   # imputed later (inside CV)

    # BMI: only when both present
    has_bmi = (~df["height"].isna()) & (~df["weight"].isna()) & (df["height"] > 0)
    out["bmi"]         = np.nan
    out["bmi_missing"] = 1.0
    out.loc[has_bmi, "bmi"]         = (df.loc[has_bmi, "weight"] /
                                        (df.loc[has_bmi, "height"] / 100) ** 2)
    out.loc[has_bmi, "bmi_missing"] = 0.0

    # ── Heart axis: one-hot + "unknown" bucket ─────────────────────────
    axis = df["heart_axis"].fillna("unknown").str.strip()
    for cat in HEART_AXIS_CATS:
        out[f"axis_{cat}"] = (axis == cat).astype(float)

    # ── Signal quality flags ──────────────────────────────────────────
    out["validated_by_human"] = df["validated_by_human"].astype(float)
    out["has_baseline_drift"] = (~df["baseline_drift"].isna()).astype(float)
    out["has_static_noise"]   = (~df["static_noise"].isna()).astype(float)
    out["has_burst_noise"]    = (~df["burst_noise"].isna()).astype(float)
    out["has_electrode_prob"] = (~df["electrodes_problems"].isna()).astype(float)

    # ── Derived / interaction ─────────────────────────────────────────
    # Signal quality composite: 0 = all clean, higher = more issues
    out["noise_score"] = (out["has_baseline_drift"] + out["has_static_noise"] +
                           out["has_burst_noise"]    + out["has_electrode_prob"])

    return out


# ────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF text features (must be fit inside CV on training fold only)
# ────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(max_features: int = 100) -> TfidfVectorizer:
    """
    Returns an *unfitted* TfidfVectorizer configured for PTB-XL German reports.
    Call .fit_transform(X_train_reports) inside each fold.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),          # unigrams + bigrams catch "sinusrhythmus normales"
        min_df=5,                     # ignore very rare terms
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def audit_tfidf_leakage(vectorizer: TfidfVectorizer, y_train: np.ndarray,
                         X_train_tfidf: np.ndarray,
                         threshold_corr: float = 0.3) -> list:
    """
    After fitting TF-IDF on training data, checks each term for label correlation.
    Returns list of column indices to DROP from the TF-IDF matrix.

    A term is flagged if:
      (a) its name matches LABEL_LEAKING_TERMS regex, OR
      (b) its point-biserial correlation with y_train exceeds threshold_corr
    """
    terms = vectorizer.get_feature_names_out()
    drop_cols = []
    for i, term in enumerate(terms):
        if LABEL_LEAKING_TERMS.search(term):
            drop_cols.append(i)
            continue
        col = X_train_tfidf[:, i]
        if col.std() > 0:
            corr = abs(np.corrcoef(col, y_train)[0, 1])
            if corr >= threshold_corr:
                drop_cols.append(i)
    if drop_cols:
        dropped = [terms[i] for i in drop_cols]
        print(f"  [leakage audit] Dropping {len(dropped)} TF-IDF terms: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
    return drop_cols


# ────────────────────────────────────────────────────────────────────────────
# 3. Master builder — returns raw (unfitted) feature matrix
# ────────────────────────────────────────────────────────────────────────────

def build_metadata_matrix(metadata_csv: str,
                           include_text: bool = True,
                           tfidf_max_features: int = 100
                           ) -> tuple:
    """
    Loads the 2000-patient subset metadata CSV and builds the static
    feature DataFrame.  Text (TF-IDF) columns are NOT yet included —
    they must be fitted inside each CV fold.

    Returns:
        df_static  (pd.DataFrame, shape [N, n_static_features])
            All non-text features. Contains NaNs for height/weight/bmi
            where missing — impute inside CV.
        reports    (pd.Series)
            Raw German report text, aligned with df_static index.
        y          (np.ndarray, int)
            Binary labels: 0 = Normal, 1 = Abnormal
        patient_ids (np.ndarray)
            patient_id for each row (used to build CV groups).
    """
    df = pd.read_csv(metadata_csv)

    df_static   = _static_features(df)
    reports     = df["report"].fillna("")
    y           = (df["class"] != "Normal").astype(int).values
    patient_ids = df["patient_id"].values

    print(f"Metadata matrix: {df_static.shape[0]} records × {df_static.shape[1]} static features")
    print(f"  Continuous features: age, height, weight, bmi")
    print(f"  Binary flags:        sex, validated_by_human, has_* noise/drift/electrode flags")
    print(f"  One-hot:             heart_axis ({len(HEART_AXIS_CATS)} buckets)")
    print(f"  Text reports:        {(reports != '').sum()} non-empty ({include_text=})")
    print(f"  Class balance:       Normal={int((y==0).sum())}, Abnormal={int((y==1).sum())}")

    return df_static, reports, y, patient_ids


# ────────────────────────────────────────────────────────────────────────────
# 4. Fold-level preprocessing (called inside each CV fold)
# ────────────────────────────────────────────────────────────────────────────

def fit_metadata_preprocessors(X_static_train: pd.DataFrame,
                                 reports_train: pd.Series,
                                 y_train: np.ndarray,
                                 include_text: bool = True,
                                 tfidf_max_features: int = 100
                                 ) -> tuple:
    """
    Fits imputer + scaler + (optional) TF-IDF on TRAINING fold only.
    Returns fitted objects and the processed training array.

    CRITICAL: call this inside each CV fold with train data only.
    Never fit on validation or test data.
    """
    # Imputer: median on continuous columns (robust to outliers)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_static_train)

    # Scaler: StandardScaler on all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    tfidf_vec = None
    tfidf_drop_cols = []
    n_text_cols = 0

    if include_text:
        tfidf_vec = build_tfidf_vectorizer(tfidf_max_features)
        X_tfidf = tfidf_vec.fit_transform(reports_train).toarray()
        tfidf_drop_cols = audit_tfidf_leakage(tfidf_vec, y_train, X_tfidf)
        # Remove leaking columns
        keep = [i for i in range(X_tfidf.shape[1]) if i not in tfidf_drop_cols]
        X_tfidf_clean = X_tfidf[:, keep]
        n_text_cols = X_tfidf_clean.shape[1]
        X_final_train = np.hstack([X_scaled, X_tfidf_clean])
    else:
        X_final_train = X_scaled

    preprocessors = {
        "imputer":         imputer,
        "scaler":          scaler,
        "tfidf_vec":       tfidf_vec,
        "tfidf_drop_cols": tfidf_drop_cols,
        "include_text":    include_text,
    }

    print(f"  [meta-preproc] Train shape: {X_final_train.shape} "
          f"({X_scaled.shape[1]} structured + {n_text_cols} text features)")

    return X_final_train, preprocessors


def apply_metadata_preprocessors(X_static: pd.DataFrame,
                                   reports: pd.Series,
                                   preprocessors: dict) -> np.ndarray:
    """
    Applies pre-fitted imputer + scaler + TF-IDF to a new split (val or test).
    Only transforms — never fits.
    """
    X_imp    = preprocessors["imputer"].transform(X_static)
    X_scaled = preprocessors["scaler"].transform(X_imp)

    if preprocessors["include_text"] and preprocessors["tfidf_vec"] is not None:
        X_tfidf = preprocessors["tfidf_vec"].transform(reports).toarray()
        keep    = [i for i in range(X_tfidf.shape[1])
                   if i not in preprocessors["tfidf_drop_cols"]]
        X_tfidf_clean = X_tfidf[:, keep]
        return np.hstack([X_scaled, X_tfidf_clean])

    return X_scaled


# ────────────────────────────────────────────────────────────────────────────
# Standalone audit
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    META_CSV = "dataset_1d/subset_metadata_2000.csv"
    if not os.path.exists(META_CSV):
        print(f"Metadata CSV not found: {META_CSV}")
    else:
        df_static, reports, y, pids = build_metadata_matrix(META_CSV)
        print(f"\nStatic feature columns ({df_static.shape[1]}):")
        for col in df_static.columns:
            null_rate = df_static[col].isna().mean()
            print(f"  {col:<30}  null={null_rate:.1%}")

        print("\n── Quick label-correlation audit on static features ──")
        for col in df_static.columns:
            vals = df_static[col].fillna(df_static[col].median())
            if vals.std() > 0:
                corr = abs(np.corrcoef(vals, y)[0, 1])
                if corr > 0.10:
                    flag = " ⚠️  REVIEW" if corr > 0.25 else ""
                    print(f"  {col:<30}  r={corr:.3f}{flag}")

        print("\n── TF-IDF leakage audit (fit on all 2000 — for audit only, not training) ──")
        vec = build_tfidf_vectorizer(200)
        X_tfidf = vec.fit_transform(reports).toarray()
        dropped = audit_tfidf_leakage(vec, y, X_tfidf, threshold_corr=0.25)
        print(f"  Would drop {len(dropped)} / {X_tfidf.shape[1]} TF-IDF terms")

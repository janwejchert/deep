# ECG Feature Engineering — Complete Detailed Guide

> **Project Status (2026-06-12):** The 1D ResNet baseline on 2,000 patients achieves  
> OOF ROC-AUC = 0.9192, Sensitivity = 0.8480, Specificity = 0.8400.  
> Feature engineering is the recommended **next enhancement layer** to push  
> interpretability, sub-pathology discrimination, and robustness beyond the deep baseline.

---

## 0. Why Feature Engineering — Even With a Working Deep Model

Your 1D ResNet is now performant, but it is a **black box**. Hand-engineered ECG features add three distinct values:

| Value | What it gives you |
|---|---|
| **Interpretability** | Feature importances map directly to cardiology (QRS duration, ST deviation, pathological Q waves) — an examiner or clinician can audit why the model predicted Abnormal |
| **Sub-pathology discrimination** | "Abnormal" collapses MI + STTC + CD + HYP. Each pathology has a well-known ECG fingerprint. Targeted features give the model the exact signal each one produces; the ResNet must learn this from waveforms alone |
| **Ensemble diversity** | A gradient-boosted tree (LightGBM) on features makes uncorrelated errors vs the ResNet. Averaging probabilities from both typically beats either model alone |
| **Low-data robustness** | If you ever test smaller subsets, a LightGBM on 50–150 features is still highly competitive at N=200; the ResNet falls apart |

**Recommended path:**
1. Extract features (this guide + `feature_engineering.py` script)
2. Train a LightGBM on the feature matrix under the same patient-disjoint CV
3. Average LightGBM and ResNet probabilities (stacking ensemble)
4. Report each model separately AND the ensemble

---

## 1. The Signal Processing Foundation

Before extracting any features, the raw 12-lead signal must be preprocessed **identically** to the ResNet pipeline:

### 1.1 Preprocessing Steps

```python
import wfdb
import numpy as np
import scipy.signal as scipy_signal

def load_and_preprocess(record_path, fs=100, target_length=1000):
    """
    Loads a PTB-XL record and applies the full preprocessing pipeline.
    Returns: (1000, 12) numpy array, or None on failure.
    
    Steps:
      1. Load raw WFDB signal (12-lead, 100 Hz)
      2. Bandpass filter: 0.5–40 Hz (removes baseline wander + powerline noise)
      3. Per-lead Z-normalise (removes amplitude scale differences between patients)
      4. Window to exactly 10 seconds (pad or truncate)
    """
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal.astype(np.float32)          # shape: (T, 12)
        
        # 1. Bandpass: 4th-order Butterworth, zero-phase (filtfilt)
        nyq = 0.5 * fs
        b, a = scipy_signal.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
        filtered = np.zeros_like(sig)
        for ch in range(sig.shape[1]):
            filtered[:, ch] = scipy_signal.filtfilt(b, a, sig[:, ch])
        
        # 2. Per-lead Z-normalise
        mean = filtered.mean(axis=0)
        std  = filtered.std(axis=0)
        std[std == 0] = 1.0
        normalised = (filtered - mean) / std
        
        # 3. Window to target_length samples (10 s @ 100 Hz)
        if normalised.shape[0] >= target_length:
            return normalised[:target_length, :]
        else:
            pad = np.zeros((target_length - normalised.shape[0], 12))
            return np.vstack([normalised, pad])
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        return None
```

### 1.2 Why Each Step Matters

| Step | Problem it solves |
|---|---|
| **Bandpass 0.5–40 Hz** | Removes respiratory baseline drift (< 0.5 Hz) and powerline noise (50/60 Hz). Without this, ST-level features are contaminated by baseline wander |
| **Per-lead Z-normalisation** | Removes patient-to-patient amplitude variation (electrode contact quality, body habitus). The ResNet needs this; so does feature extraction for ST and amplitude features |
| **10-second window** | Ensures all records have the same length for consistent beat counting and HRV calculation |

---

## 2. Fiducial Point Detection — The Extraction Foundation

Every ECG feature is derived from locating waveform landmarks (P, Q, R, S, T) per heartbeat.
**Use NeuroKit2** — it implements Pan-Tompkins R-peak detection plus wavelet-based delineation:

```python
import neurokit2 as nk

def delineate_lead(signal_1d, fs=100):
    """
    Detects R-peaks and delineates P, Q, S, T waves for one lead.
    
    Returns:
      rpeaks   (array): sample indices of R-peaks
      waves    (dict):  keys like ECG_P_Onsets, ECG_R_Onsets, ECG_T_Peaks, etc.
      quality  (float): 0–1 signal quality score from NeuroKit2
    
    Robustness: returns ([], {}, 0.0) on any failure — caller should
    record a detection_failed flag and impute from training median.
    """
    try:
        _, info = nk.ecg_process(signal_1d, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            return [], {}, 0.0
        _, waves = nk.ecg_delineate(
            signal_1d, rpeaks, sampling_rate=fs, method="dwt"
        )
        quality = info.get("ECG_Quality", np.ones(len(signal_1d))).mean()
        return rpeaks, waves, float(quality)
    except Exception:
        return [], {}, 0.0
```

### PTB-XL Lead Order (12-lead mapping)

```python
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
# signal[:, 0] = Lead I
# signal[:, 1] = Lead II
# ...
# signal[:, 11] = Lead V6
```

---

## 3. Interval & Timing Features (The Diagnostic Backbone)

These are the most clinically validated features. Compute per-beat, then aggregate (mean, std, min, max).

| Feature | Definition | Normal range | Clinical signal |
|---|---|---|---|
| **RR interval** | Time between consecutive R-peaks | 0.6–1.0 s | Heart rate, rhythm regularity |
| **Heart rate** | 60 / mean(RR) | 60–100 bpm | Brady/tachycardia |
| **PR interval** | P-onset → QRS-onset | 120–200 ms | AV conduction delay (CD) |
| **QRS duration** | QRS-onset → QRS-offset | 60–100 ms | Bundle branch block (CD) |
| **QT interval** | QRS-onset → T-offset | 350–440 ms | Repolarisation |
| **QTc (Bazett)** | QT / √(RR) | < 440 ms | Rate-corrected QT |
| **QTc (Fridericia)** | QT / RR^(1/3) | < 450 ms | More accurate at high HR |
| **ST duration** | J-point → T-onset | — | Ischaemia duration |

```python
def interval_features(rpeaks, waves, fs=100):
    """
    Extracts timing/interval features from one lead's beat detections.
    All times reported in seconds.
    Returns flat dict with keys: hr_mean, rr_mean, rr_std, rr_cv,
    qrs_mean, qrs_std, qt_mean, qtc_bazett, qtc_fridericia, pr_mean
    """
    feats = {}
    
    if len(rpeaks) < 2:
        return {k: np.nan for k in
                ["hr_mean", "rr_mean", "rr_std", "rr_cv",
                 "qrs_mean", "qrs_std", "qt_mean",
                 "qtc_bazett", "qtc_fridericia", "pr_mean"]}
    
    rr = np.diff(rpeaks) / fs                    # RR intervals in seconds
    rr = rr[(rr > 0.3) & (rr < 2.0)]            # remove physiologically impossible RR
    
    feats["hr_mean"]   = 60 / np.mean(rr) if len(rr) else np.nan
    feats["rr_mean"]   = float(np.mean(rr))
    feats["rr_std"]    = float(np.std(rr))
    feats["rr_cv"]     = feats["rr_std"] / feats["rr_mean"] if feats["rr_mean"] else np.nan
    
    # QRS duration
    q_on  = np.array([x for x in waves.get("ECG_R_Onsets",  []) if x is not None and not np.isnan(x)], dtype=float)
    q_off = np.array([x for x in waves.get("ECG_R_Offsets", []) if x is not None and not np.isnan(x)], dtype=float)
    if len(q_on) and len(q_off):
        n = min(len(q_on), len(q_off))
        qrs_dur = (q_off[:n] - q_on[:n]) / fs
        qrs_dur = qrs_dur[(qrs_dur > 0.04) & (qrs_dur < 0.30)]
        feats["qrs_mean"] = float(np.nanmean(qrs_dur))
        feats["qrs_std"]  = float(np.nanstd(qrs_dur))
    else:
        feats["qrs_mean"] = feats["qrs_std"] = np.nan
    
    # QT interval
    t_off = np.array([x for x in waves.get("ECG_T_Offsets", []) if x is not None and not np.isnan(x)], dtype=float)
    if len(q_on) and len(t_off):
        n = min(len(q_on), len(t_off))
        qt = (t_off[:n] - q_on[:n]) / fs
        qt = qt[(qt > 0.2) & (qt < 0.7)]
        feats["qt_mean"] = float(np.nanmean(qt))
        rr_mean = feats["rr_mean"]
        feats["qtc_bazett"]     = feats["qt_mean"] / np.sqrt(rr_mean)     if rr_mean else np.nan
        feats["qtc_fridericia"] = feats["qt_mean"] / (rr_mean ** (1/3))   if rr_mean else np.nan
    else:
        feats["qt_mean"] = feats["qtc_bazett"] = feats["qtc_fridericia"] = np.nan
    
    # PR interval
    p_on = np.array([x for x in waves.get("ECG_P_Onsets", []) if x is not None and not np.isnan(x)], dtype=float)
    if len(p_on) and len(q_on):
        n = min(len(p_on), len(q_on))
        pr = (q_on[:n] - p_on[:n]) / fs
        pr = pr[(pr > 0.08) & (pr < 0.40)]
        feats["pr_mean"] = float(np.nanmean(pr)) if len(pr) else np.nan
    else:
        feats["pr_mean"] = np.nan
    
    return feats
```

---

## 4. Amplitude Features (Per Lead — Voltage Matters by Lead)

Voltage magnitude is directly diagnostic: tall R-waves indicate hypertrophy; deep Q-waves indicate MI; ST shifts indicate ischaemia/STTC.

| Feature | Why it matters |
|---|---|
| **R-wave amplitude** | Voltage criteria for hypertrophy (must check specific leads) |
| **Q-wave depth & width** | Pathological Q = prior MI (depth > 25% of R, duration > 40 ms) |
| **S-wave amplitude** | Used in Sokolow-Lyon for LVH |
| **ST level at J+60ms** | Gold standard for ischaemia; elevation = acute MI, depression = subendocardial |
| **T-wave amplitude** | Negative T → ischaemia/STTC |
| **T-wave inversion fraction** | Proportion of beats with inverted T in this lead |
| **QRS amplitude (R + |S|)** | Total voltage for hypertrophy criteria |

```python
def amplitude_features(signal_1d, waves, fs=100):
    """
    Extracts amplitude features for one lead.
    Returns flat dict with keys: r_amp_mean, r_amp_max, q_amp_mean,
    s_amp_mean, t_amp_mean, t_inverted_frac, st_level_j60, qrs_amp_mean
    """
    feats = {}
    
    def _safe_idx(wave_key):
        return np.array([x for x in waves.get(wave_key, [])
                         if x is not None and not np.isnan(x)], dtype=int)
    
    # R amplitude
    r_idx = _safe_idx("ECG_R_Peaks")
    r_idx = r_idx[r_idx < len(signal_1d)]
    r_amp = signal_1d[r_idx] if len(r_idx) else np.array([np.nan])
    feats["r_amp_mean"] = float(np.nanmean(r_amp))
    feats["r_amp_max"]  = float(np.nanmax(r_amp))
    
    # Q amplitude (should be negative — depth = absolute value)
    q_idx = _safe_idx("ECG_Q_Peaks")
    q_idx = q_idx[q_idx < len(signal_1d)]
    q_amp = signal_1d[q_idx] if len(q_idx) else np.array([np.nan])
    feats["q_amp_mean"]  = float(np.nanmean(np.abs(q_amp)))   # store as positive depth
    feats["q_depth_frac"] = float(np.nanmean(np.abs(q_amp) > 0.25 * np.nanmean(np.abs(r_amp)))) if len(q_amp) else np.nan
    
    # S amplitude
    s_idx = _safe_idx("ECG_S_Peaks")
    s_idx = s_idx[s_idx < len(signal_1d)]
    s_amp = signal_1d[s_idx] if len(s_idx) else np.array([np.nan])
    feats["s_amp_mean"] = float(np.nanmean(np.abs(s_amp)))
    
    # QRS total amplitude (R + |S|)
    feats["qrs_amp_mean"] = feats["r_amp_mean"] + feats["s_amp_mean"]
    
    # ST level: signal value 60 ms after J-point (QRS offset)
    j_off = _safe_idx("ECG_R_Offsets")
    st_idx = (j_off + int(0.06 * fs))
    st_idx = st_idx[st_idx < len(signal_1d)]
    feats["st_level_j60"] = float(np.nanmean(signal_1d[st_idx])) if len(st_idx) else np.nan
    
    # T amplitude and polarity
    t_idx = _safe_idx("ECG_T_Peaks")
    t_idx = t_idx[t_idx < len(signal_1d)]
    t_amp = signal_1d[t_idx] if len(t_idx) else np.array([np.nan])
    feats["t_amp_mean"]       = float(np.nanmean(t_amp))
    feats["t_inverted_frac"]  = float(np.nanmean(t_amp < 0)) if len(t_amp) else np.nan
    
    return feats
```

---

## 5. Heart Rate Variability (HRV) — Rhythm & Autonomic Features

HRV captures rhythm regularity, autonomic tone, and arrhythmia risk. On a 10-second strip you can compute time-domain and short-term nonlinear features reliably. Frequency-domain HRV needs longer strips (≥ 5 min) — treat those as unreliable and optional.

```python
def hrv_features(rpeaks, fs=100):
    """
    Computes HRV features from R-peak indices.
    Returns dict with time-domain + nonlinear HRV metrics.
    Falls back to empty dict on failure.
    """
    try:
        hrv_df = nk.hrv(rpeaks, sampling_rate=fs, show=False)
        # NeuroKit2 returns a 1-row DataFrame with ~80 HRV metrics
        d = hrv_df.iloc[0].to_dict()
        # Keep only the most interpretable and stable features
        keep = ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
                "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
                "HRV_SampEn", "HRV_CVNN"]
        return {k: float(d[k]) for k in keep if k in d}
    except Exception:
        return {}
```

| HRV Feature | Clinical signal |
|---|---|
| **SDNN** | Overall heart rate variability; reduced in heart failure, ischaemia |
| **RMSSD** | Short-term beat-to-beat variation; parasympathetic activity |
| **pNN50** | Proportion of consecutive RR differences > 50 ms; autonomic tone |
| **SD1/SD2** | Poincaré geometry: SD1 = short-term variability, SD2 = long-term |
| **SampEn** | Sample entropy; low in pathological rhythms |

---

## 6. Subclass-Targeted Clinical Features — The Highest-Value Section

Because "Abnormal" = MI ∪ STTC ∪ CD ∪ HYP, engineer features that directly encode the ECG signature of each sub-pathology. This is the most clinically informed and highest-value feature family.

### 6.1 Myocardial Infarction (MI) Features

MI has three ECG phases:
- **Acute**: ST elevation, hyperacute T waves
- **Evolving**: T-wave inversion, Q-wave development
- **Old**: Pathological Q waves, loss of R progression

**Regional lead groupings matter:**
- **Inferior**: II, III, aVF (right coronary artery territory)
- **Anterior**: V1, V2, V3, V4 (left anterior descending artery)
- **Lateral**: I, aVL, V5, V6 (circumflex artery)

```python
MI_REGIONS = {
    "inferior":  ["II", "III", "aVF"],
    "anterior":  ["V1", "V2", "V3", "V4"],
    "lateral":   ["I",  "aVL", "V5", "V6"]
}

def mi_features(signals_by_lead, waves_by_lead, fs=100):
    """
    Computes MI-specific features per anatomical region.
    Returns dict with keys: st_{region}, t_inv_{region}, qpath_{region}
    """
    feats = {}
    for region, leads in MI_REGIONS.items():
        st_vals, t_inv, q_path = [], [], []
        for L in leads:
            if L not in signals_by_lead:
                continue
            s = signals_by_lead[L]
            w = waves_by_lead[L]
            
            # ST level at J+60ms
            j_off = np.array([x for x in w.get("ECG_R_Offsets", [])
                              if x is not None and not np.isnan(x)], dtype=int)
            st_idx = j_off + int(0.06 * fs)
            st_idx = st_idx[st_idx < len(s)]
            if len(st_idx):
                st_vals.append(float(np.nanmean(s[st_idx])))
            
            # T-wave inversion fraction
            t_pk = np.array([x for x in w.get("ECG_T_Peaks", [])
                             if x is not None and not np.isnan(x)], dtype=int)
            t_pk = t_pk[t_pk < len(s)]
            if len(t_pk):
                t_amp = s[t_pk]
                t_inv.append(float(np.nanmean(t_amp < 0)))
            
            # Pathological Q waves (depth > 25% of R amplitude)
            r_pk = np.array([x for x in w.get("ECG_R_Peaks", [])
                             if x is not None and not np.isnan(x)], dtype=int)
            q_pk = np.array([x for x in w.get("ECG_Q_Peaks", [])
                             if x is not None and not np.isnan(x)], dtype=int)
            r_pk = r_pk[r_pk < len(s)]
            q_pk = q_pk[q_pk < len(s)]
            if len(r_pk) and len(q_pk):
                r_mean = np.nanmean(np.abs(s[r_pk]))
                q_amp  = np.abs(s[q_pk])
                q_path.append(float(np.nanmean(q_amp > 0.25 * r_mean)))
        
        feats[f"st_{region}"]    = float(np.nanmean(st_vals)) if st_vals else np.nan
        feats[f"t_inv_{region}"] = float(np.nanmean(t_inv))   if t_inv   else np.nan
        feats[f"qpath_{region}"] = float(np.nanmean(q_path))  if q_path  else np.nan
    
    return feats
```

### 6.2 ST/T Change (STTC) Features

STTC captures any ST depression or T-wave abnormality without a territory pattern.

```python
def sttc_features(signals_by_lead, waves_by_lead, fs=100):
    """Computes global STTC features across all 12 leads."""
    feats = {}
    st_all, t_inv_all, t_r_ratio = [], [], []
    for L in LEADS:
        if L not in signals_by_lead:
            continue
        s = signals_by_lead[L]
        w = waves_by_lead[L]
        
        j_off = np.array([x for x in w.get("ECG_R_Offsets", [])
                          if x is not None and not np.isnan(x)], dtype=int)
        j_off = j_off[j_off < len(s)]
        j60   = (j_off + int(0.06 * fs))
        j60   = j60[j60 < len(s)]
        if len(j60):
            st_all.append(float(np.nanmean(s[j60])))
        
        t_pk = np.array([x for x in w.get("ECG_T_Peaks", [])
                         if x is not None and not np.isnan(x)], dtype=int)
        r_pk = np.array([x for x in w.get("ECG_R_Peaks", [])
                         if x is not None and not np.isnan(x)], dtype=int)
        t_pk = t_pk[t_pk < len(s)]
        r_pk = r_pk[r_pk < len(s)]
        
        if len(t_pk):
            t_amp = s[t_pk]
            t_inv_all.append(float(np.nanmean(t_amp < 0)))
            if len(r_pk):
                r_mean = np.nanmean(np.abs(s[r_pk]))
                if r_mean > 0:
                    t_r_ratio.append(float(np.nanmean(np.abs(t_amp) / r_mean)))
    
    feats["st_global_mean"]    = float(np.nanmean(st_all))    if st_all    else np.nan
    feats["st_global_std"]     = float(np.nanstd(st_all))     if st_all    else np.nan
    feats["t_inv_global"]      = float(np.nanmean(t_inv_all)) if t_inv_all else np.nan
    feats["t_r_ratio_mean"]    = float(np.nanmean(t_r_ratio)) if t_r_ratio else np.nan
    return feats
```

### 6.3 Conduction Disturbance (CD) Features

CD includes bundle branch blocks and AV blocks. Diagnosed by QRS duration and PR prolongation.

```python
def cd_features(feats_per_lead):
    """
    Computes CD markers from pre-computed per-lead interval features.
    feats_per_lead: dict of {lead_name: {qrs_mean: ..., pr_mean: ...}}
    """
    feats = {}
    qrs_vals = [feats_per_lead[L]["qrs_mean"] for L in LEADS
                if L in feats_per_lead and not np.isnan(feats_per_lead[L].get("qrs_mean", np.nan))]
    pr_vals  = [feats_per_lead[L]["pr_mean"] for L in LEADS
                if L in feats_per_lead and not np.isnan(feats_per_lead[L].get("pr_mean", np.nan))]
    
    mean_qrs = float(np.nanmean(qrs_vals)) if qrs_vals else np.nan
    mean_pr  = float(np.nanmean(pr_vals))  if pr_vals  else np.nan
    
    feats["qrs_mean_global"]     = mean_qrs
    feats["pr_mean_global"]      = mean_pr
    feats["bbb_flag"]            = float(mean_qrs >= 0.12)         # ≥120 ms = BBB
    feats["av_block_flag"]       = float(mean_pr > 0.20)           # >200 ms = 1st degree
    feats["lbbb_v1_flag"]        = np.nan  # needs V1 morphology (rS pattern)
    return feats
```

### 6.4 Hypertrophy (HYP) Voltage Criteria

Hypertrophy is diagnosed from voltage amplitudes in specific leads. These are direct numerical criteria used in clinical practice.

```python
def hyp_features(r_amp_by_lead, s_amp_by_lead):
    """
    Computes voltage-based hypertrophy criteria.
    r_amp_by_lead: dict {lead_name: mean R-wave amplitude in mV}
    s_amp_by_lead: dict {lead_name: mean S-wave amplitude in mV}
    
    Note: PTB-XL signals are in mV after loading.
    Amplitudes here are already Z-normalised in the pipeline,
    so Sokolow-Lyon raw thresholds are approximate — train a model
    to learn the threshold rather than hard-coding.
    """
    feats = {}
    
    # Sokolow-Lyon (LVH): S(V1) + max(R(V5), R(V6)) > 3.5 mV
    sv1  = s_amp_by_lead.get("V1", np.nan)
    rv5  = r_amp_by_lead.get("V5", np.nan)
    rv6  = r_amp_by_lead.get("V6", np.nan)
    if not np.isnan(sv1) and not (np.isnan(rv5) and np.isnan(rv6)):
        sokolow = sv1 + max(x for x in [rv5, rv6] if not np.isnan(x))
        feats["sokolow_lyon"] = float(sokolow)
    else:
        feats["sokolow_lyon"] = np.nan
    
    # Cornell (LVH): R(aVL) + S(V3)
    ravl = r_amp_by_lead.get("aVL", np.nan)
    sv3  = s_amp_by_lead.get("V3",  np.nan)
    feats["cornell_voltage"] = float(ravl + sv3) if not (np.isnan(ravl) or np.isnan(sv3)) else np.nan
    
    # R/S ratio in V1 (RVH marker when > 1)
    rv1 = r_amp_by_lead.get("V1", np.nan)
    sv1_r = s_amp_by_lead.get("V1", np.nan)
    feats["rs_ratio_v1"] = float(rv1 / sv1_r) if not (np.isnan(rv1) or np.isnan(sv1_r) or sv1_r == 0) else np.nan
    
    return feats
```

---

## 7. Spectral & Statistical Features (Cheap Signal Descriptors)

These require no fiducial detection — compute directly from the filtered signal.

```python
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def spectral_statistical_features(signal_1d, fs=100):
    """
    Computes spectral and statistical features for one lead.
    No fiducial detection required — robust to poor-quality signals.
    """
    feats = {}
    
    # Statistical descriptors
    feats["mean"]     = float(np.mean(signal_1d))
    feats["std"]      = float(np.std(signal_1d))
    feats["skewness"] = float(skew(signal_1d))
    feats["kurtosis"] = float(kurtosis(signal_1d))
    feats["energy"]   = float(np.sum(signal_1d ** 2))
    feats["zcr"]      = float(np.mean(np.diff(np.sign(signal_1d)) != 0))
    
    # Hjorth parameters (complexity indicators)
    d1 = np.diff(signal_1d)
    d2 = np.diff(d1)
    var0, var1, var2 = np.var(signal_1d), np.var(d1), np.var(d2)
    feats["hjorth_activity"]   = float(var0)
    feats["hjorth_mobility"]   = float(np.sqrt(var1 / var0)) if var0 else np.nan
    feats["hjorth_complexity"] = float(np.sqrt(var2 / var1) / feats["hjorth_mobility"]) \
                                  if var1 and feats["hjorth_mobility"] else np.nan
    
    # Welch spectral power in clinical frequency bands
    freq, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    for lo, hi, name in [(0, 5, "vlf"), (5, 15, "lf"), (15, 40, "hf")]:
        mask = (freq >= lo) & (freq < hi)
        feats[f"power_{name}"] = float(np.sum(psd[mask]))
    total_power = float(np.sum(psd))
    feats["power_total"]    = total_power
    feats["lf_hf_ratio"]    = feats["power_lf"] / feats["power_hf"] \
                               if feats["power_hf"] > 0 else np.nan
    feats["spec_entropy"]   = float(-np.sum((psd / (total_power + 1e-12)) *
                                            np.log(psd / (total_power + 1e-12) + 1e-12)))
    feats["dominant_freq"]  = float(freq[np.argmax(psd)])
    
    return feats
```

---

## 8. Building the Full Feature Matrix (Leak-Free)

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def extract_all_features(signal_12lead, fs=100):
    """
    Master extractor: runs all feature families on a (1000, 12) signal.
    Returns a flat dict of all features for one record.
    NaN is used for missing/failed features — imputed later.
    """
    feats = {}
    signals_by_lead = {}
    waves_by_lead   = {}
    rpeaks_by_lead  = {}
    interval_by_lead = {}
    r_amp_by_lead = {}
    s_amp_by_lead = {}
    
    for i, L in enumerate(LEADS):
        sig = signal_12lead[:, i]
        signals_by_lead[L] = sig
        
        # Fiducial detection
        rpeaks, waves, quality = delineate_lead(sig, fs)
        waves_by_lead[L]  = waves
        rpeaks_by_lead[L] = rpeaks
        feats[f"{L}_detection_quality"] = quality
        
        if len(rpeaks) >= 2:
            # Interval features
            int_f = interval_features(rpeaks, waves, fs)
            interval_by_lead[L] = int_f
            for k, v in int_f.items():
                feats[f"{L}_{k}"] = v
            
            # HRV features
            hrv_f = hrv_features(rpeaks, fs)
            for k, v in hrv_f.items():
                feats[f"{L}_{k}"] = v
        else:
            interval_by_lead[L] = {k: np.nan for k in
                ["hr_mean", "rr_mean", "rr_std", "rr_cv",
                 "qrs_mean", "qrs_std", "qt_mean",
                 "qtc_bazett", "qtc_fridericia", "pr_mean"]}
            feats[f"{L}_detection_failed"] = 1.0
        
        # Amplitude features (need rpeaks and waves)
        amp_f = amplitude_features(sig, waves, fs)
        for k, v in amp_f.items():
            feats[f"{L}_{k}"] = v
        r_amp_by_lead[L] = amp_f.get("r_amp_mean", np.nan)
        s_amp_by_lead[L] = amp_f.get("s_amp_mean", np.nan)
        
        # Spectral/statistical features (always computable)
        stat_f = spectral_statistical_features(sig, fs)
        for k, v in stat_f.items():
            feats[f"{L}_{k}"] = v
    
    # Cross-lead / subclass-targeted features
    feats.update(mi_features(signals_by_lead, waves_by_lead, fs))
    feats.update(sttc_features(signals_by_lead, waves_by_lead, fs))
    feats.update(cd_features(interval_by_lead))
    feats.update(hyp_features(r_amp_by_lead, s_amp_by_lead))
    
    return feats


def build_feature_matrix(metadata_csv, base_dir, fs=100):
    """
    Builds the complete feature matrix for all records in the metadata CSV.
    Returns: X (DataFrame), y (Series)
    """
    df = pd.read_csv(metadata_csv)
    rows = []
    labels = []
    
    for _, row in df.iterrows():
        rec_path = f"{base_dir}/{row['filename_lr']}"
        signal   = load_and_preprocess(rec_path, fs=fs)
        if signal is None:
            continue
        feat_row = extract_all_features(signal, fs=fs)
        rows.append(feat_row)
        labels.append(0 if row["class"] == "Normal" else 1)
    
    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")
    print(f"Feature matrix: {X.shape[0]} records × {X.shape[1]} features")
    print(f"Missing value rate: {X.isna().mean().mean():.1%}")
    return X, y
```

### Leakage-Free Imputation & Scaling

```python
# CRITICAL: fit imputer and scaler ONLY on training fold, apply to val/test
def fit_preprocessors(X_train):
    imputer = SimpleImputer(strategy="median")   # median — robust to outliers
    scaler  = StandardScaler()
    X_tr = imputer.fit_transform(X_train)
    X_tr = scaler.fit_transform(X_tr)
    return X_tr, imputer, scaler

def apply_preprocessors(X, imputer, scaler):
    return scaler.transform(imputer.transform(X))
```

---

## 9. Feature Selection

Reduce ~300 raw features to the most stable and informative 50–100:

```python
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold

def select_features(X_train, y_train, X_test=None, top_k=80):
    """
    Three-step feature selection (all done on training data only):
    1. Drop near-constant features (low variance)
    2. Drop high-missing features (> 40% NaN before imputation)
    3. Rank by LightGBM feature importance, keep top_k
    """
    # Step 1: variance filter
    selector = VarianceThreshold(threshold=0.01)
    X_tr = selector.fit_transform(X_train)
    cols = X_train.columns[selector.get_support()]
    
    # Step 2: quick LightGBM for importance ranking
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf.fit(X_tr, y_train)
    
    importance = pd.Series(clf.feature_importances_, index=cols)
    top_cols = importance.nlargest(top_k).index.tolist()
    
    print(f"Selected {len(top_cols)} features from {X_train.shape[1]}")
    
    if X_test is not None:
        return X_train[top_cols], X_test[top_cols], top_cols
    return X_train[top_cols], top_cols
```

---

## 10. Modeling With the Feature Matrix

The recommended model stack, in priority order:

```python
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Option A: LightGBM (the recommended primary model)
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,       # prevents overfit at N=2000
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight=None,          # dataset is balanced 1000/1000
    verbose=-1
)

# Option B: Logistic Regression (strong interpretable baseline)
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=0.1)

# Option C: Stacking ensemble — average ResNet + LightGBM probabilities
def ensemble_predict(resnet_probs, lgb_probs, weight_resnet=0.5):
    return weight_resnet * resnet_probs + (1 - weight_resnet) * lgb_probs
```

Keep the **same evaluation protocol** as the ResNet:
- 5-fold patient-disjoint Stratified K-Fold
- Out-of-Fold (OOF) predictions
- Nested Platt scaling on the validation slice
- Sensitivity-targeted threshold (≥ 0.85)
- Bootstrap 95% CIs

---

## 11. Expected Impact on Current Results

| Contribution | Expected Effect |
|---|---|
| **Interval features** (QRS, QT, PR) | CD recall improvement; QRS > 120ms directly flags bundle branch block |
| **ST features** | MI and STTC discrimination; single most powerful individual feature |
| **Pathological Q features** | MI specificity; reduces false positives on Normal |
| **Voltage criteria** | HYP discrimination |
| **LightGBM ensemble** | 2–4% AUC gain, steadier specificity across folds |

> [!TIP]
> The subclass-targeted features in Section 6 are the single highest-leverage addition. The ResNet sees raw waveforms and must learn ST elevation = MI from 2,000 examples. A feature that directly computes "mean ST level in the inferior leads" already **knows** what to look for.

---

## 12. Install

```bash
pip install neurokit2 wfdb lightgbm scikit-learn scipy shap
```

Run the feature extraction pipeline:

```bash
python feature_engineering.py --metadata dataset_1d/subset_metadata_2000.csv \
                               --signals dataset_1d/raw \
                               --output features_2000.csv
```

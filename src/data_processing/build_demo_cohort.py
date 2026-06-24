"""
Build the committed demo cohort for the CardioAI Streamlit MVP.

Selects a small, class-balanced set of PTB-XL records from patients that were
NEVER used in training (patient-disjoint from both training metadata CSVs),
copies their raw 100 Hz waveforms into ``data/raw/`` and writes
``data/unseen_demo_metadata.csv``. Everything it produces is small enough to
commit, so the app runs on a fresh clone / Streamlit Cloud with zero setup and
the in-app leakage audit honestly shows 0 overlap with the training sets.

Provenance / inputs (not committed; large or re-downloadable):
  - data/ptbxl_database.csv, data/scp_statements.csv  (PTB-XL index; auto-downloaded if absent)
  - a directory containing PTB-XL ``records100/`` raw signals (``--src``)

Usage:
  python src/data_processing/build_demo_cohort.py --src "data/Imagenes eco" --per-class 8
"""
import argparse
import ast
import os
import shutil
import urllib.request

import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"


def ensure_index(data_dir):
    """Download the two small PTB-XL index files if they are not present."""
    for name in ("scp_statements.csv", "ptbxl_database.csv"):
        dest = os.path.join(data_dir, name)
        if not os.path.exists(dest):
            print(f"  downloading {name} ...")
            urllib.request.urlretrieve(PTBXL_BASE + name, dest)
    return (os.path.join(data_dir, "ptbxl_database.csv"),
            os.path.join(data_dir, "scp_statements.csv"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    data_dir = os.path.join(repo, "data")

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(data_dir, "raw"),
                    help="directory that contains a 'records100/' folder of raw signals")
    ap.add_argument("--per-class", type=int, default=8,
                    help="target records per superclass (NORM/MI/STTC/CD/HYP)")
    args = ap.parse_args()
    src_root = args.src if os.path.isabs(args.src) else os.path.join(repo, args.src)

    db_path, scp_path = ensure_index(data_dir)

    # scp code -> diagnostic superclass (diagnostic statements only)
    scp = pd.read_csv(scp_path, index_col=0)
    scp = scp[scp.diagnostic == 1.0]
    scp_to_class = scp["diagnostic_class"].dropna().to_dict()

    def superclasses_of(scp_codes_str):
        out = set()
        for code, conf in ast.literal_eval(scp_codes_str).items():
            if code in scp_to_class and conf > 50:
                out.add(scp_to_class[code])
        return out

    # Patients used anywhere in training -> excluded from the demo cohort
    train_patients = set()
    for f in ("subset_metadata_2000.csv", "subset_multiclass_metadata.csv"):
        train_patients |= set(pd.read_csv(os.path.join(data_dir, f))["patient_id"].dropna())

    db = pd.read_csv(db_path)
    db["cls"] = db["scp_codes"].apply(superclasses_of)
    db["on_disk"] = db["filename_lr"].apply(
        lambda f: os.path.exists(os.path.join(src_root, f + ".hea"))
        and os.path.exists(os.path.join(src_root, f + ".dat")))

    cand = db[(~db["patient_id"].isin(train_patients))
              & db["on_disk"]
              & (db["cls"].apply(len) > 0)
              & (db["age"].between(0, 95))].copy()
    cand = cand.drop_duplicates(subset="patient_id")
    print(f"  patient-unseen, on-disk, labeled candidates: {len(cand)}")

    # Class-balanced selection, one record per patient, no duplicate ecg_ids
    selected, seen = [], set()
    for sc in SUPERCLASSES:
        pool = cand[cand["cls"].apply(lambda s: sc in s) & ~cand["ecg_id"].isin(seen)]
        pick = pool.sort_values("ecg_id").head(args.per_class)
        selected.append(pick)
        seen |= set(pick["ecg_id"])
    demo = pd.concat(selected).drop_duplicates("ecg_id").reset_index(drop=True)

    for sc in SUPERCLASSES:
        demo[f"label_{sc}"] = demo["cls"].apply(lambda s: int(sc in s))
    keep = ["ecg_id", "patient_id", "age", "sex", "height", "weight",
            "report", "scp_codes", "filename_lr"] + [f"label_{sc}" for sc in SUPERCLASSES]
    demo = demo[[c for c in keep if c in demo.columns]]

    # Copy raw signals into data/raw/<filename_lr>.{dat,hea}
    raw_dir = os.path.join(data_dir, "raw")
    copied = 0
    for fn in demo["filename_lr"]:
        for ext in (".dat", ".hea"):
            dst = os.path.join(raw_dir, fn + ext)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(src_root, fn + ext), dst)
        copied += 1

    out_csv = os.path.join(data_dir, "unseen_demo_metadata.csv")
    demo.to_csv(out_csv, index=False)

    print(f"\n  Demo cohort: {len(demo)} records / {demo['patient_id'].nunique()} patients, {copied} signals copied")
    for sc in SUPERCLASSES:
        print(f"    {sc}: {int(demo[f'label_{sc}'].sum())}")
    print(f"  Wrote {out_csv}")
    print(f"  Overlap with training patients: {len(set(demo['patient_id']) & train_patients)} (must be 0)")


if __name__ == "__main__":
    main()

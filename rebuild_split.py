"""
rebuild_split.py
================
Rebuilds dataset_split folders using the cluster-level patient-disjoint
assignment from regroup_suggestion.csv. No near-duplicates will straddle
train/val/test after this.
"""
import os
import csv
import shutil

CSV_PATH = '/Users/felipedeleon/Desktop/Deep Ler,Project/docs/regroup_suggestion.csv'
SPLIT_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split'
BACKUP_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split_leaked_backup'

def main():
    # 1. Back up the old (leaked) split
    if not os.path.exists(BACKUP_DIR):
        print(f"Backing up old split to {BACKUP_DIR}...")
        shutil.copytree(SPLIT_DIR, BACKUP_DIR)
    else:
        print(f"Backup already exists at {BACKUP_DIR}, skipping.")

    # 2. Clear existing split directories
    for split in ('train', 'val', 'test'):
        for cls in ('Normal', 'Abnormal'):
            d = os.path.join(SPLIT_DIR, split, cls)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    # 3. Read the CSV and copy files to their new_split location
    moved = 0
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['path']
            new_split = row['new_split']
            cls = row['class']
            fname = os.path.basename(src)

            # Source may now be in the backup since we cleared the split dir
            # Try original path first, then backup
            if not os.path.exists(src):
                # Reconstruct from backup
                old_split = row['old_split']
                src = os.path.join(BACKUP_DIR, old_split, cls, fname)

            dst = os.path.join(SPLIT_DIR, new_split, cls, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                moved += 1
            else:
                print(f"  WARNING: source not found: {src}")

    print(f"\nMoved {moved} images into patient-disjoint split.")

    # 4. Report new counts
    for split in ('train', 'val', 'test'):
        for cls in ('Normal', 'Abnormal'):
            d = os.path.join(SPLIT_DIR, split, cls)
            n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/{cls}: {n}")

if __name__ == '__main__':
    main()

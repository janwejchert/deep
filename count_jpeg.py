import os
from collections import Counter

base_dir = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_new_extracted"

jpeg_files = []
other_files = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        p = os.path.join(root, f)
        if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg'):
            jpeg_files.append(p)
        else:
            other_files.append(p)

print(f"Total JPEGs: {len(jpeg_files)}")
print(f"Total other files: {len(other_files)}")
print("\nFirst 10 JPEGs:")
for jp in jpeg_files[:10]:
    print(" ", jp.replace(base_dir, ""))

print("\nOther files details:")
for of in other_files:
    print(" ", of.replace(base_dir, ""))

# Let's count JPEGs per subdirectory
subdirs = Counter([os.path.dirname(jp).replace(base_dir, "") for jp in jpeg_files])
print("\nJPEG counts by subdirectory:")
for sd, count in subdirs.items():
    print(f"  {sd}: {count}")

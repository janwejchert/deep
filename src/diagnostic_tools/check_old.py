import os

normal_dir = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary/Normal"
abnormal_dir = "/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary/Abnormal"

normal_files = os.listdir(normal_dir)
abnormal_files = os.listdir(abnormal_dir)

print(f"Total Normal in dataset_binary: {len(normal_files)}")
print(f"Sample Normal: {sorted(normal_files)[:10]}")

print(f"Total Abnormal in dataset_binary: {len(abnormal_files)}")
print(f"Sample Abnormal: {sorted(abnormal_files)[:10]}")

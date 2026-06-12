import os
import shutil
import random

def split_data():
    source_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_binary'
    dest_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split'
    
    classes = ['Abnormal', 'Normal']
    splits = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    for split in splits.keys():
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        files = os.listdir(cls_dir)
        # Random shuffle with seed for reproducibility
        random.seed(42)
        random.shuffle(files)
        
        train_split = int(len(files) * splits['train'])
        val_split = train_split + int(len(files) * splits['val'])
        
        train_files = files[:train_split]
        val_files = files[train_split:val_split]
        test_files = files[val_split:]
        
        print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        for f in train_files:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(dest_dir, 'train', cls, f))
        for f in val_files:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(dest_dir, 'val', cls, f))
        for f in test_files:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(dest_dir, 'test', cls, f))

if __name__ == '__main__':
    split_data()
    print("Dataset splitting complete!")

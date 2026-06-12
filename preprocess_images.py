import os
from pathlib import Path
from PIL import Image

# Paths
SRC_ROOT = Path('dataset_split')
DST_ROOT = Path('clean_dataset')

TARGET_SIZE = (224, 224)
FORMAT = 'PNG'

def process_split(split):
    for cls in ['Normal', 'Abnormal']:
        src_dir = SRC_ROOT / split / cls
        dst_dir = DST_ROOT / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            continue
        for img_file in src_dir.iterdir():
            if not img_file.is_file():
                continue
            try:
                with Image.open(img_file) as img:
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    dst_path = dst_dir / (img_file.stem + '.' + FORMAT.lower())
                    img.save(dst_path, FORMAT)
            except Exception as e:
                print(f'Failed {img_file}: {e}')

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        process_split(split)
    print('Preprocessing complete. Clean dataset saved to clean_dataset/')

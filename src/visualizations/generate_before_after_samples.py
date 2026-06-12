import os, random
from PIL import Image

# Configuration
SRC_ROOT = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_split'
DST_ROOT = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset_clean'
OUTPUT_DIR = '/Users/felipedeleon/Desktop/Deep Ler,Project/sample_visuals'
NUM_SAMPLES_PER_CLASS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = ['Normal', 'Abnormal']
for cls in classes:
    src_dir = os.path.join(SRC_ROOT, 'train', cls)
    dst_dir = os.path.join(DST_ROOT, 'train', cls)
    if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
        continue
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    sample_files = random.sample(files, min(NUM_SAMPLES_PER_CLASS, len(files)))
    for i, fn in enumerate(sample_files, 1):
        src_path = os.path.join(src_dir, fn)
        dst_path = os.path.join(dst_dir, os.path.splitext(fn)[0] + '.png')
        try:
            im_before = Image.open(src_path).convert('RGB')
            im_after = Image.open(dst_path).convert('RGB')
            # Resize both to 224x224 for consistent display (after is already 224)
            im_before = im_before.resize((224, 224), Image.LANCZOS)
            # Combine side‑by‑side
            combined = Image.new('RGB', (224 * 2, 224))
            combined.paste(im_before, (0, 0))
            combined.paste(im_after, (224, 0))
            out_path = os.path.join(OUTPUT_DIR, f'{cls}_sample_{i}.png')
            combined.save(out_path)
        except Exception as e:
            print(f'Error processing {fn}: {e}')
print('Sample visualisations saved to', OUTPUT_DIR)

import argparse
import os
import hashlib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random

import numpy as np

REENCODE_FORMAT = "PNG"      # lossless, identical for all -> no JPEG-quality signal
EXT_IN = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def autocrop_to_content(im, bg=255, margin=4):
    """Crop a grayscale PIL image to its dark-content bounding box + margin."""
    arr = np.array(im)
    # Rows/cols that contain at least one pixel darker than background - 10
    rows = np.any(arr < (bg - 10), axis=1)
    cols = np.any(arr < (bg - 10), axis=0)
    if not rows.any() or not cols.any():
        return im  # nothing to crop
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add small margin, clamped to image bounds
    rmin = max(0, rmin - margin)
    rmax = min(arr.shape[0] - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(arr.shape[1] - 1, cmax + margin)
    return im.crop((cmin, rmin, cmax + 1, rmax + 1))

def standardize_one(path_in, path_out, target_size):
    target = target_size[0]  # square, so just use one dimension
    im = Image.open(path_in).convert("L")

    # 1) Asymmetric crop — kill the header banner (top) and footer/padding (bottom).
    #    This is what Grad-CAM showed the model reading.
    w, h = im.size
    top, bot, side = int(h * 0.12), int(h * 0.10), int(w * 0.03)
    im = im.crop((side, top, w - side, h - bot))

    # 2) Autocrop to the dark trace, drop whitespace.
    im = autocrop_to_content(im)

    # 3) Binarize and Skeletonize — keep only the topological structure of the trace
    from skimage.morphology import skeletonize
    arr = np.array(im)
    # skimage skeletonize expects True for the foreground (the trace).
    bool_arr = (arr < 110)
    skel = skeletonize(bool_arr)
    im = Image.fromarray(np.where(skel, 0, 255).astype("uint8"))

    # 4) Pad to square (identical white for all -> no class signal) + resize.
    im = ImageOps.pad(im, (max(im.size), max(im.size)), color=255)
    im = im.resize((target, target), Image.LANCZOS).convert("RGB")

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    im.save(path_out, REENCODE_FORMAT)

def create_sample_sheet(image_paths, dst_dir, target_size):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    for img_path, ax in zip(image_paths[:9], axes):
        im = Image.open(img_path)
        ax.imshow(im)
        ax.axis('off')
    
    # Hide any unused axes
    for ax in axes[len(image_paths):]:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(dst_dir, '_sample_sheet.png'), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root with {split}/{class}/ images")
    ap.add_argument("--dst", required=True, help="output root (mirrors structure)")
    ap.add_argument("--size", type=int, default=384, help="Target size for resizing")
    args = ap.parse_args()
    
    target_size = (args.size, args.size)

    n = 0
    sizes_by_class = {}
    seen_hashes = set()
    duplicate_count = 0
    saved_images = []
    
    for split in ("train", "val", "test"):
        for cls in ("Normal", "Abnormal"):
            d = os.path.join(args.src, split, cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(EXT_IN):
                    continue
                src = os.path.join(d, fn)
                stem = os.path.splitext(fn)[0]
                dst = os.path.join(args.dst, split, cls, stem + ".png")
                try:
                    standardize_one(src, dst, target_size)
                    with open(dst, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        os.remove(dst)
                        duplicate_count += 1
                    else:
                        seen_hashes.add(file_hash)
                        n += 1
                        saved_images.append(dst)
                        sizes_by_class.setdefault(cls, set()).add(Image.open(dst).size)
                except Exception as e:
                    print(f"  skip {src}: {e}")

    # Generate the sample sheet
    if saved_images:
        random.shuffle(saved_images)
        create_sample_sheet(saved_images, args.dst, target_size)

    print(f"Standardized {n} images (removed {duplicate_count} duplicates) -> {args.dst}")
    for cls, sizes in sizes_by_class.items():
        print(f"  {cls}: sizes = {sizes}")
    if all(s == {target_size} for s in sizes_by_class.values()):
        print("  OK: Dimension shortcut removed.")
    else:
        print("  WARNING: sizes still vary.")

if __name__ == "__main__":
    main()

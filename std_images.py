import argparse
import os
import hashlib
from PIL import Image, ImageOps
TARGET = (224, 224)          # final size fed to EfficientNetB0
EDGE_CROP_FRAC = 0.06        # crop 6% off each edge to strip headers/labels/borders
REENCODE_FORMAT = "PNG"      # lossless, identical for all -> no JPEG-quality signal
EXT_IN = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def standardize_one(path_in, path_out):
    im = Image.open(path_in)
    # 1) Identical color mode for every image (kills mode-based source signal).
    im = im.convert("L").convert("RGB")
    # 2) Class-blind edge crop: strip a fixed fraction off all four sides.
    w, h = im.size
    dx, dy = int(w * EDGE_CROP_FRAC), int(h * EDGE_CROP_FRAC)
    im = im.crop((dx, dy, w - dx, h - dy))
    # 3) Pad to square BEFORE resizing so aspect ratio is normalized identically
    im = ImageOps.pad(im, (max(im.size), max(im.size)), color=(255, 255, 255))
    # 4) Identical resize for every image -> destroys the dimension shortcut.
    im = im.resize(TARGET, Image.LANCZOS)
    # 5) Re-encode identically (same format, no quality variation).
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    im.save(path_out, REENCODE_FORMAT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root with {split}/{class}/ images")
    ap.add_argument("--dst", required=True, help="output root (mirrors structure)")
    args = ap.parse_args()

    n = 0
    sizes_by_class = {}
    seen_hashes = set()
    duplicate_count = 0
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
                    standardize_one(src, dst)
                    # Compute MD5 hash of the saved image for duplicate detection
                    with open(dst, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        os.remove(dst)
                        duplicate_count += 1
                    else:
                        seen_hashes.add(file_hash)
                        n += 1
                        sizes_by_class.setdefault(cls, set()).add(Image.open(dst).size)
                except Exception as e:
                    print(f"  skip {src}: {e}")

    print(f"Standardized {n} images (removed {duplicate_count} duplicates) -> {args.dst}")
    print("\nPost-standardization size check (must be identical across classes):")
    for cls, sizes in sizes_by_class.items():
        print(f"  {cls}: sizes = {sizes} | formats = PNG, mode = RGB")
    if all(s == {TARGET} for s in sizes_by_class.values()):
        print("  OK: every image is identical in size/format/mode. Dimension shortcut removed.")
        print("  NEXT: re-run the source-bias probe (metadata-AUC must -> ~0.50), then gradcam_fixed.py to confirm waveform attention.")
    else:
        print("  WARNING: sizes still vary — investigate before retraining.")


if __name__ == "__main__":
    main()

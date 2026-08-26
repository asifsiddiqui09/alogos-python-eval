"""
B) First quantitative evaluation on the London-DB (real faces) set.

All London-DB images are REAL, so the ground-truth label is "real" for every
image. We run the Python detector over the folder and measure how often it
correctly says "real" (i.e. the false-positive rate of the detector on genuine
photos). This is the first column of the eventual confusion matrix.

Usage:
  python evaluate_londondb.py <folder> [--max-size 256] [--limit 0] [--seed 42] [--threshold 0.5]
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alogos import SyntheticImageDetector, ImageData, DetectorOptions
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_rgba(path: Path, max_size: int):
    img = Image.open(path).convert("RGBA")
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.BILINEAR)
    w, h = img.size
    return ImageData(width=w, height=h, data=img.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--max-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="londondb_results.csv")
    args = ap.parse_args()

    folder = Path(args.folder)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]

    print(f"Found {len(images)} images in {folder}")
    print(f"Config: max_size={args.max_size}, threshold={args.threshold}, seed={args.seed}\n")

    detector = SyntheticImageDetector(DetectorOptions(threshold=args.threshold))

    rows = []
    correct = 0  # correctly classified as REAL
    scores = []
    t0 = time.time()

    for i, path in enumerate(images, 1):
        random.seed(args.seed)  # reproducibility: deterministic per-image result
        try:
            image = load_rgba(path, args.max_size)
            r = detector.analyse(image)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        ground_truth = "real"
        predicted = "AI" if r.is_synthetic else "real"
        is_correct = predicted == ground_truth
        correct += int(is_correct)
        scores.append(r.raw_score)

        rows.append({
            "file": path.name,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "raw_score": round(r.raw_score, 6),
            "confidence": round(r.confidence, 6),
            "primary_variance": round(r.metadata.primary_variance, 6),
            "coherence": round(r.metadata.coherence, 6),
            "kurtosis": round(r.metadata.kurtosis, 6),
        })

        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)} processed...")

    elapsed = time.time() - t0
    n = len(rows)

    out_path = Path(__file__).resolve().parent / args.out
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- Summary ---
    tp_real = correct
    fp_ai = n - correct
    mean_score = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 60)
    label = folder.parent.name if folder.name.lower() in ("test", "train", "val", "validation") else folder.name
    print(f"{label.upper()} EVALUATION")
    print("=" * 60)
    print(f"Images evaluated     : {n}")
    print(f"Classified as REAL   : {tp_real}" + (f"  ({tp_real/n:.1%})" if n else ""))
    print(f"Classified as AI     : {fp_ai}" + (f"  ({fp_ai/n:.1%})" if n else ""))
    print(f"Mean raw score       : {mean_score:.4f}  (threshold {args.threshold})")
    print(f"Time                 : {elapsed:.1f}s  ({elapsed / n:.2f}s/image)" if n else "")
    print("=" * 60)
    print(f"Per-image CSV written to: {out_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_ROOT = HERE.parent
JS_ROOT = PY_ROOT / "Alogos"

sys.path.insert(0, str(PY_ROOT))
import decode_image
import run_python

IMG_EXTS = {".jpg", ".jpeg", ".png"}
NPX = "npx.cmd" if platform.system() == "Windows" else "npx"


def run_js(bin_path: str):
    proc = subprocess.run(
        [NPX, "tsx", "run_one.ts", bin_path],
        cwd=JS_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"JS run failed: {proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-size", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--out", default="js_python_comparison.csv")
    args = ap.parse_args()

    folder = Path(args.folder)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        images = images[: args.limit]

    print(f"Found {len(images)} images in {folder}")
    keys = ["raw_score", "confidence", "primary_variance", "coherence"]
    max_diff = {k: 0.0 for k in keys}
    mismatches = []
    rows = []
    bin_path = HERE / "_tmp_dataset_compare.bin"

    t0 = time.time()
    for i, img_path in enumerate(images, 1):
        decode_image.decode_to_rgba(str(img_path), str(bin_path), args.max_size)
        py = run_python.run(str(bin_path), runs=1)[0]
        js = run_js(str(bin_path))

        row = {"file": img_path.name}
        all_match = True
        for k in keys:
            diff = abs(py[k] - js[k])
            max_diff[k] = max(max_diff[k], diff)
            row[f"py_{k}"] = round(py[k], 6)
            row[f"js_{k}"] = round(js[k], 6)
            row[f"diff_{k}"] = diff
            if diff > args.tolerance:
                all_match = False
        row["match"] = all_match
        row["py_verdict"] = "AI" if py["is_synthetic"] else "REAL"
        row["js_verdict"] = "AI" if js["is_synthetic"] else "REAL"
        row["verdict_match"] = row["py_verdict"] == row["js_verdict"]
        rows.append(row)
        if not all_match:
            mismatches.append(row)

        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)} processed...")

    bin_path.unlink(missing_ok=True)
    bin_path.with_suffix(".json").unlink(missing_ok=True)
    elapsed = time.time() - t0

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_match = sum(1 for r in rows if r["match"])
    n_verdict_match = sum(1 for r in rows if r["verdict_match"])

    print("\n" + "=" * 60)
    print(f"JS vs PYTHON - FULL DATASET COMPARISON ({folder.name})")
    print("=" * 60)
    print(f"Images compared      : {n}")
    print(f"Metrics matched (tol={args.tolerance}): {n_match}/{n}")
    print(f"Verdicts matched     : {n_verdict_match}/{n}")
    print(f"Time                 : {elapsed:.1f}s  ({elapsed/n:.2f}s/image)")
    print("\nMax absolute difference observed, per metric:")
    for k in keys:
        print(f"  {k:<18}: {max_diff[k]:.2e}")
    if mismatches:
        print(f"\n{len(mismatches)} image(s) exceeded tolerance:")
        for r in mismatches[:10]:
            print(f"  {r['file']}: " + ", ".join(
                f"{k} diff={r[f'diff_{k}']:.2e}" for k in keys if r[f"diff_{k}"] > args.tolerance
            ))
    else:
        print("\nNo mismatches above tolerance in any image.")
    print(f"\nPer-image CSV written to: {args.out}")


if __name__ == "__main__":
    main()

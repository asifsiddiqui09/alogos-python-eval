"""
A) JS vs PYTHON comparison.

Decodes ONE image to raw RGBA (so both detectors see byte-identical input),
runs the Python detector and the JavaScript detector N times each, and prints
a side-by-side table.

Key forensic points this demonstrates:
  * gradient coherence is deterministic  -> should match EXACTLY across languages
  * raw_score / confidence / primary_variance use a random-initialised PCA
    power iteration -> small run-to-run variance; we report mean +/- std.

Usage:
  python compare.py <image_path> [--runs 5] [--max-size 256]
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_ROOT = HERE.parent
JS_ROOT = PY_ROOT.parent / "alogos-main"

import decode_image
import run_python


def summarise(results, key):
    vals = [r[key] for r in results]
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def fmt(mean, std):
    return f"{mean:.6f} +/- {std:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=256,
                    help="downscale longest side for speed (0 = full size)")
    args = ap.parse_args()

    tmp = HERE / "_tmp_compare.bin"
    decode_image.decode_to_rgba(args.image, str(tmp), args.max_size)

    print(f"\nRunning Python detector x{args.runs} ...")
    py = run_python.run(str(tmp), args.runs)

    print(f"Running JavaScript detector x{args.runs} ...")
    js_proc = subprocess.run(
        ["node", "run_js.mjs", str(tmp), str(args.runs)],
        cwd=JS_ROOT, capture_output=True, text=True,
    )
    if js_proc.returncode != 0:
        print("JS run failed:\n", js_proc.stderr)
        sys.exit(1)
    js = json.loads(js_proc.stdout.strip().splitlines()[-1])

    keys = ["raw_score", "confidence", "primary_variance", "coherence"]
    print(f"JS vs PYTHON  |  image={Path(args.image).name}  runs={args.runs}")
    print(f"{'metric':<20}{'Python (mean+/-std)':<27}{'JavaScript (mean+/-std)':<27}")
    for k in keys:
        pm, ps = summarise(py, k)
        jm, js_ = summarise(js, k)
        print(f"{k:<20}{fmt(pm, ps):<27}{fmt(jm, js_):<27}")
    print(f"Python verdict : {'AI' if py[0]['is_synthetic'] else 'REAL'}")
    print(f"JS verdict     : {'AI' if js[0]['is_synthetic'] else 'REAL'}")
    print(f"Pixels analysed: {py[0]['pixels']:,}")
    print("Note: coherence is deterministic -> matches exactly.")
    print("      score/variance use random PCA init -> tiny run-to-run spread.")

    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()

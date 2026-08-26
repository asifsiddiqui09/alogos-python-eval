"""
Run the PYTHON detector on a raw RGBA .bin file, N times, and print metrics
as JSON. Used by compare.py to validate the re-implementation against JS.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alogos import SyntheticImageDetector, ImageData, DetectorOptions


def run(bin_path: str, runs: int = 5, seed=None):
    meta = json.loads(Path(bin_path).with_suffix(".json").read_text())
    raw = Path(bin_path).read_bytes()

    image = ImageData(width=meta["width"], height=meta["height"], data=raw)
    detector = SyntheticImageDetector(DetectorOptions())  # defaults

    results = []
    for i in range(runs):
        if seed is not None:
            random.seed(seed)  # reproducibility demo: same seed -> identical output
        r = detector.analyse(image)
        results.append({
            "raw_score": r.raw_score,
            "confidence": r.confidence,
            "is_synthetic": r.is_synthetic,
            "primary_variance": r.metadata.primary_variance,
            "coherence": r.metadata.coherence,
            "pixels": r.metadata.pixels_analysed,
        })
    return results


if __name__ == "__main__":
    bin_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(json.dumps(run(bin_path, runs, seed)))

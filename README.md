# Alogos (Python) — Personal Analysis Fork

A Python library for detecting synthetic (AI-generated) images using **luminance-gradient PCA analysis**.

This is a personal analysis fork of the team's Python port of [`Alogos`](https://github.com/REllwood/Alogos) (original TypeScript implementation by R. Ellwood), built for the WTP IT Security Project (Topic 7a) at Otto-von-Guericke-Universität Magdeburg. Real photographs produce coherent gradient fields tied to physical lighting and camera sensors, while diffusion-generated images contain unstable high-frequency structure from the denoising process. Alogos converts an image to luminance, computes its gradient field, and evaluates the covariance via PCA to expose that difference.

**This fork adds:**
- `kurtosis` exposed in the detector's output (computed internally by the original scoring formula, but previously discarded rather than returned)
- A full-scale evaluation across 7 datasets (6,204 images total — 3 real, 4 AI-generated)
- Scoring-formula reweighting experiments testing whether adjusting the kurtosis weight and/or classification threshold can recover correct classification on real images (see `evaluation/`)

## Installation
```bash
pip install -e .
```
Requires Python 3.8+. The core library has **no dependencies**. For loading real images in the demo, install Pillow:
```bash
pip install -e ".[demo]"
```

## Quick start
```python
from alogos_python import detect_synthetic_image, ImageData, DetectorOptions

# image_data holds raw RGBA bytes (length = width * height * 4)
image_data = ImageData(width=128, height=128, data=[...])

result = detect_synthetic_image(image_data, DetectorOptions(threshold=0.6))
print("AI-generated" if result.is_synthetic else "Real photo")
print(f"Raw score:  {result.raw_score:.4f}")
print(f"Confidence: {result.confidence * 100:.1f}%")
print(f"Kurtosis:   {result.metadata.kurtosis:.4f}")
```

### Run the demo on an image file
```bash
python demo.py path/to/image.jpg --threshold 0.6
```

## API

### `detect_synthetic_image(image_data, options=None) -> DetectionResult`
Convenience function — analyses a single image in one call.

### `SyntheticImageDetector(options=None)`
Reusable detector class.
- **`analyse(image_data) -> DetectionResult`** — full analysis of an image.
- **`analyse_gradients(image_data) -> GradientField`** — returns just the gradient field.
- **`get_options()` / `set_options(options)`** — read or update configuration.

### `DetectorOptions`
| Field | Default | Description |
|-------|---------|-------------|
| `threshold` | `0.5` | Score at/above which an image is flagged synthetic. |
| `num_components` | `5` | Number of PCA components. |
| `normalise_gradients` | `True` | Normalise luminance before computing gradients. |
| `min_image_size` | `64` | Minimum width/height (pixels). |
| `filter_compression_artifacts` | `True` | Filter JPEG-style compression noise. |

### `DetectionResult`
- `is_synthetic: bool`
- `confidence: float` (0–1)
- `raw_score: float`
- `metadata`: `pixels_analysed`, `primary_variance`, `coherence`, **`kurtosis`** *(added in this fork — see `alogos_python/pca.py::compute_pca_kurtosis`)*

### `ImageData`
- `width: int`, `height: int`
- `data`: flat list of RGBA byte values, length `width * height * 4`

## Project structure
```
alogos-python-eval/
├── Alogos/                # original TypeScript implementation (reference only, gitignored)
├── alogos_python/          # this Python port
│   ├── detector.py         # SyntheticImageDetector, detect_synthetic_image
│   ├── luminance.py        # luminance / brightness calculations
│   ├── pca.py               # PCA, scoring, and kurtosis extraction
│   ├── gradients.py         # gradient-field computation & coherence
│   └── types.py             # ImageData, DetectorOptions, DetectionResult, ...
├── evaluation/             # full evaluation: scripts + per-image CSV results
│   ├── evaluate_londondb.py      # runs the detector over a dataset folder
│   ├── londondb_sweep.py         # kurtosis-weight sensitivity sweep
│   ├── weight_threshold_grid.py  # joint weight/threshold grid search
│   ├── generalization_check.py   # tests whether findings generalise across real datasets
│   └── *.csv                     # per-image results, 7 datasets, 200- and 1,000-image runs
├── tests/                  # pytest test suite
└── demo.py                 # CLI demo (requires Pillow)
```

## Evaluation summary

Across 7 datasets (London-DB, FFHQ, CelebA — real; StarGAN, PGGAN v1, PGGAN v2, FaceApp — AI-generated; 6,204 images total), the detector classifies every image as AI regardless of content. Root cause: the scoring formula's kurtosis term saturates the score's clamp to `[0,1]` for essentially any photographic gradient distribution — confirmed both analytically and by measuring kurtosis directly (every image measured, real or AI, exceeds the saturation threshold). Attempting to fix this by reweighting the formula does not generalise across real datasets: see `evaluation/generalization_check.py` and its output for details. Full write-up in the accompanying project report.

## Testing
```bash
pytest
```

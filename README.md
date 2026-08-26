# Alogos (Python)

A Python library for detecting synthetic (AI-generated) images using **luminance-gradient PCA analysis**.

This is the Python port of [`alogos-main`](../alogos-main) (TypeScript). Real photographs produce coherent gradient fields tied to physical lighting and camera sensors, while diffusion-generated images contain unstable high-frequency structure from the denoising process. Alogos converts an image to luminance, computes its gradient field, and evaluates the covariance via PCA to expose that difference.

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
from alogos import detect_synthetic_image, ImageData, DetectorOptions

# image_data holds raw RGBA bytes (length = width * height * 4)
image_data = ImageData(width=128, height=128, data=[...])

result = detect_synthetic_image(image_data, DetectorOptions(threshold=0.6))

print("AI-generated" if result.is_synthetic else "Real photo")
print(f"Raw score:  {result.raw_score:.4f}")
print(f"Confidence: {result.confidence * 100:.1f}%")
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
- `metadata`: `pixels_analysed`, `primary_variance`, `coherence`

### `ImageData`
- `width: int`, `height: int`
- `data`: flat list of RGBA byte values, length `width * height * 4`

## Project structure

```
alogos-python/
├── alogos/
│   ├── detector.py    # SyntheticImageDetector, detect_synthetic_image
│   ├── luminance.py   # luminance / brightness calculations
│   ├── pca.py         # Principal Component Analysis
│   ├── gradients.py   # gradient-field computation & coherence
│   └── types.py       # ImageData, DetectorOptions, DetectionResult, ...
├── tests/             # pytest test suite
└── demo.py            # CLI demo (requires Pillow)
```

## Testing

```bash
pytest
```

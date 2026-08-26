from .detector import SyntheticImageDetector, detect_synthetic_image
from .types import ImageData, GradientField, PCAResult, DetectionResult, DetectorOptions
from .luminance import rgb_to_luminance, image_to_luminance_matrix, normalise_luminance, filter_compression_artifacts
from .gradients import compute_gradients, flatten_gradient_field, compute_gradient_coherence
from .pca import perform_pca, compute_pca_score, compute_covariance_matrix

VERSION = "1.0.0"

__all__ = [
    "SyntheticImageDetector",
    "detect_synthetic_image",
    "ImageData",
    "GradientField",
    "PCAResult",
    "DetectionResult",
    "DetectorOptions",
    "rgb_to_luminance",
    "image_to_luminance_matrix",
    "normalise_luminance",
    "filter_compression_artifacts",
    "compute_gradients",
    "flatten_gradient_field",
    "compute_gradient_coherence",
    "perform_pca",
    "compute_pca_score",
    "compute_covariance_matrix",
    "VERSION",
]

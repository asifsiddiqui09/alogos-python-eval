from .types import ImageData, DetectionResult, DetectionMetadata, DetectorOptions, GradientField
from .luminance import image_to_luminance_matrix, normalise_luminance, filter_compression_artifacts
from .gradients import compute_gradients, flatten_gradient_field, compute_gradient_coherence
from .pca import perform_pca, compute_pca_score, compute_pca_kurtosis

_DEFAULT_OPTIONS = DetectorOptions(
    threshold=0.5,
    num_components=5,
    normalise_gradients=True,
    min_image_size=64,
    filter_compression_artifacts=True,
)


class SyntheticImageDetector:
    def __init__(self, options: DetectorOptions = None):
        base = DetectorOptions(
            threshold=_DEFAULT_OPTIONS.threshold,
            num_components=_DEFAULT_OPTIONS.num_components,
            normalise_gradients=_DEFAULT_OPTIONS.normalise_gradients,
            min_image_size=_DEFAULT_OPTIONS.min_image_size,
            filter_compression_artifacts=_DEFAULT_OPTIONS.filter_compression_artifacts,
        )
        if options:
            if options.threshold is not None:
                base.threshold = options.threshold
            if options.num_components is not None:
                base.num_components = options.num_components
            if options.normalise_gradients is not None:
                base.normalise_gradients = options.normalise_gradients
            if options.min_image_size is not None:
                base.min_image_size = options.min_image_size
            if options.filter_compression_artifacts is not None:
                base.filter_compression_artifacts = options.filter_compression_artifacts
        self._options = base

    def analyse(self, image_data: ImageData) -> DetectionResult:
        self._validate(image_data)

        if image_data.width < self._options.min_image_size or image_data.height < self._options.min_image_size:
            raise ValueError(
                f"Image too small. Minimum size is "
                f"{self._options.min_image_size}x{self._options.min_image_size} pixels"
            )

        luminance = image_to_luminance_matrix(image_data)

        if self._options.filter_compression_artifacts:
            luminance = filter_compression_artifacts(luminance)

        processed = normalise_luminance(luminance) if self._options.normalise_gradients else luminance

        gradient_field = compute_gradients(processed)
        gradient_matrix = flatten_gradient_field(gradient_field)
        pca_result = perform_pca(gradient_matrix, self._options.num_components)
        raw_score = compute_pca_score(pca_result)
        kurtosis = compute_pca_kurtosis(pca_result)

        coherence = compute_gradient_coherence(gradient_field)
        primary_variance = pca_result.explained_variance[0] if pca_result.explained_variance else 0.0

        is_synthetic = raw_score >= self._options.threshold
        confidence = min(1.0, abs(raw_score - self._options.threshold) / self._options.threshold)

        return DetectionResult(
            is_synthetic=is_synthetic,
            confidence=confidence,
            raw_score=raw_score,
            metadata=DetectionMetadata(
                pixels_analysed=image_data.width * image_data.height,
                primary_variance=primary_variance,
                coherence=coherence,
                kurtosis=kurtosis,
            ),
        )

    def analyse_gradients(self, image_data: ImageData) -> GradientField:
        self._validate(image_data)
        luminance = image_to_luminance_matrix(image_data)
        processed = normalise_luminance(luminance) if self._options.normalise_gradients else luminance
        return compute_gradients(processed)

    def set_options(self, options: DetectorOptions) -> None:
        if options.threshold is not None:
            self._options.threshold = options.threshold
        if options.num_components is not None:
            self._options.num_components = options.num_components
        if options.normalise_gradients is not None:
            self._options.normalise_gradients = options.normalise_gradients
        if options.min_image_size is not None:
            self._options.min_image_size = options.min_image_size
        if options.filter_compression_artifacts is not None:
            self._options.filter_compression_artifacts = options.filter_compression_artifacts

    def get_options(self) -> DetectorOptions:
        return DetectorOptions(
            threshold=self._options.threshold,
            num_components=self._options.num_components,
            normalise_gradients=self._options.normalise_gradients,
            min_image_size=self._options.min_image_size,
            filter_compression_artifacts=self._options.filter_compression_artifacts,
        )

    def _validate(self, image_data: ImageData) -> None:
        if image_data is None:
            raise ValueError("Image data is required")
        if (
            not isinstance(image_data.width, int)
            or not isinstance(image_data.height, int)
            or image_data.width <= 0
            or image_data.height <= 0
        ):
            raise ValueError("Invalid image dimensions")
        if not image_data.data or len(image_data.data) == 0:
            raise ValueError("Image data is empty")
        expected = image_data.width * image_data.height * 4
        if len(image_data.data) != expected:
            raise ValueError(
                f"Image data length mismatch. Expected {expected} bytes (RGBA), "
                f"got {len(image_data.data)}"
            )


def detect_synthetic_image(image_data: ImageData, options: DetectorOptions = None) -> DetectionResult:
    """Convenience function — analyse a single image with optional custom options."""
    return SyntheticImageDetector(options).analyse(image_data)

from dataclasses import dataclass, field
from typing import Union, List


@dataclass
class ImageData:
    width: int
    height: int
    data: Union[bytes, bytearray, List[int]]  # RGBA format, 4 bytes per pixel


@dataclass
class GradientField:
    gx: List[List[float]]
    gy: List[List[float]]
    width: int
    height: int


@dataclass
class PCAResult:
    components: List[List[float]]
    explained_variance: List[float]
    projection: List[float]
    total_variance: float


@dataclass
class DetectionMetadata:
    pixels_analysed: int
    primary_variance: float
    coherence: float
    kurtosis: float = 0.0


@dataclass
class DetectionResult:
    is_synthetic: bool
    confidence: float
    raw_score: float
    metadata: DetectionMetadata


@dataclass
class DetectorOptions:
    threshold: float = 0.5
    num_components: int = 5
    normalise_gradients: bool = True
    min_image_size: int = 64
    filter_compression_artifacts: bool = True

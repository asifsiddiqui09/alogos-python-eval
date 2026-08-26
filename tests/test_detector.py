import pytest
from alogos.detector import SyntheticImageDetector, detect_synthetic_image
from alogos.types import ImageData, DetectorOptions


def create_test_image(width: int, height: int, pattern: str) -> ImageData:
    data = []
    for y in range(height):
        for x in range(width):
            if pattern == 'uniform':
                value = 128
            elif pattern == 'gradient':
                value = int((x / width) * 255)
            elif pattern == 'checkerboard':
                value = 255 if (x + y) % 2 == 0 else 0
            else:
                value = 0
            data += [value, value, value, 255]  # RGBA
    return ImageData(width=width, height=height, data=data)


class TestConstructor:
    def test_default_options(self):
        d = SyntheticImageDetector()
        opts = d.get_options()
        assert opts.threshold == 0.5
        assert opts.num_components == 5
        assert opts.normalise_gradients is True
        assert opts.min_image_size == 64

    def test_custom_options(self):
        d = SyntheticImageDetector(DetectorOptions(threshold=0.7, num_components=3))
        opts = d.get_options()
        assert opts.threshold == 0.7
        assert opts.num_components == 3


class TestAnalyse:
    def test_valid_image(self):
        d = SyntheticImageDetector()
        img = create_test_image(100, 100, 'gradient')
        result = d.analyse(img)
        assert isinstance(result.is_synthetic, bool)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.raw_score <= 1

    def test_metadata(self):
        d = SyntheticImageDetector()
        img = create_test_image(100, 100, 'gradient')
        result = d.analyse(img)
        assert result.metadata.pixels_analysed == 10000
        assert 0 <= result.metadata.primary_variance <= 1
        assert 0 <= result.metadata.coherence <= 1

    def test_too_small_raises(self):
        d = SyntheticImageDetector(DetectorOptions(min_image_size=64))
        img = create_test_image(32, 32, 'gradient')
        with pytest.raises(ValueError, match="Image too small"):
            d.analyse(img)

    def test_invalid_dimensions_raises(self):
        d = SyntheticImageDetector()
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            d.analyse(ImageData(width=0, height=0, data=[]))

    def test_data_length_mismatch_raises(self):
        d = SyntheticImageDetector()
        with pytest.raises(ValueError, match="Image data length mismatch"):
            d.analyse(ImageData(width=10, height=10, data=[0] * 100))  # should be 400

    def test_different_patterns(self):
        d = SyntheticImageDetector()
        for pattern in ('uniform', 'gradient', 'checkerboard'):
            result = d.analyse(create_test_image(100, 100, pattern))
            assert 0 <= result.raw_score <= 1


class TestAnalyseGradients:
    def test_returns_gradient_field(self):
        d = SyntheticImageDetector()
        img = create_test_image(100, 100, 'gradient')
        gf = d.analyse_gradients(img)
        assert gf.width == 100
        assert gf.height == 100
        assert len(gf.gx) == 100
        assert len(gf.gy) == 100


class TestSetOptions:
    def test_updates_threshold(self):
        d = SyntheticImageDetector(DetectorOptions(threshold=0.5))
        d.set_options(DetectorOptions(threshold=0.8))
        assert d.get_options().threshold == 0.8

    def test_merges_existing(self):
        d = SyntheticImageDetector(DetectorOptions(threshold=0.5, num_components=5))
        d.set_options(DetectorOptions(threshold=0.8))
        assert d.get_options().threshold == 0.8
        assert d.get_options().num_components == 5


class TestDetectSyntheticImage:
    def test_default_options(self):
        img = create_test_image(100, 100, 'gradient')
        result = detect_synthetic_image(img)
        assert hasattr(result, 'is_synthetic')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'raw_score')

    def test_custom_options(self):
        img = create_test_image(100, 100, 'gradient')
        result = detect_synthetic_image(img, DetectorOptions(threshold=0.7))
        assert hasattr(result, 'is_synthetic')

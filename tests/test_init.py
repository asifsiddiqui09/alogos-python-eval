import alogos


class TestPackageExports:
    def test_exports_detector_class(self):
        assert hasattr(alogos, 'SyntheticImageDetector')
        assert callable(alogos.SyntheticImageDetector)

    def test_exports_detect_function(self):
        assert hasattr(alogos, 'detect_synthetic_image')
        assert callable(alogos.detect_synthetic_image)

    def test_exports_utility_functions(self):
        assert callable(alogos.rgb_to_luminance)
        assert callable(alogos.image_to_luminance_matrix)
        assert callable(alogos.normalise_luminance)
        assert callable(alogos.compute_gradients)
        assert callable(alogos.flatten_gradient_field)
        assert callable(alogos.compute_gradient_coherence)
        assert callable(alogos.perform_pca)
        assert callable(alogos.compute_pca_score)
        assert callable(alogos.compute_covariance_matrix)

    def test_exports_version(self):
        assert hasattr(alogos, 'VERSION')
        assert isinstance(alogos.VERSION, str)
        assert alogos.VERSION == '1.0.0'

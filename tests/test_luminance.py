import pytest
from alogos.luminance import rgb_to_luminance, image_to_luminance_matrix, normalise_luminance
from alogos.types import ImageData


class TestRgbToLuminance:
    def test_pure_red(self):
        assert abs(rgb_to_luminance(255, 0, 0) - 0.2126 * 255) < 0.01

    def test_pure_green(self):
        assert abs(rgb_to_luminance(0, 255, 0) - 0.7152 * 255) < 0.01

    def test_pure_blue(self):
        assert abs(rgb_to_luminance(0, 0, 255) - 0.0722 * 255) < 0.01

    def test_white(self):
        assert abs(rgb_to_luminance(255, 255, 255) - 255) < 0.01

    def test_black(self):
        assert rgb_to_luminance(0, 0, 0) == 0

    def test_mid_grey(self):
        assert abs(rgb_to_luminance(128, 128, 128) - 128) < 0.01


class TestImageToLuminanceMatrix:
    def test_2x2_image(self):
        image_data = ImageData(
            width=2, height=2,
            data=[
                255, 0, 0, 255,    # Red pixel
                0, 255, 0, 255,    # Green pixel
                0, 0, 255, 255,    # Blue pixel
                255, 255, 255, 255, # White pixel
            ],
        )
        matrix = image_to_luminance_matrix(image_data)
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert abs(matrix[0][0] - 0.2126 * 255) < 0.01
        assert abs(matrix[0][1] - 0.7152 * 255) < 0.01
        assert abs(matrix[1][0] - 0.0722 * 255) < 0.01
        assert abs(matrix[1][1] - 255) < 0.01

    def test_empty_image(self):
        image_data = ImageData(width=0, height=0, data=[])
        assert image_to_luminance_matrix(image_data) == []

    def test_ignores_alpha(self):
        image_data = ImageData(width=1, height=1, data=[128, 128, 128, 0])
        matrix = image_to_luminance_matrix(image_data)
        assert abs(matrix[0][0] - 128) < 0.01


class TestNormaliseLuminance:
    def test_normalises_to_0_1(self):
        matrix = [[0, 128, 255], [64, 192, 128]]
        normalised = normalise_luminance(matrix)
        for row in normalised:
            for val in row:
                assert 0 <= val <= 1
        assert normalised[0][0] == 0
        assert normalised[0][2] == 1

    def test_uniform_values(self):
        matrix = [[128, 128], [128, 128]]
        normalised = normalise_luminance(matrix)
        for row in normalised:
            for val in row:
                assert val == 0

    def test_empty_matrix(self):
        assert normalise_luminance([]) == []

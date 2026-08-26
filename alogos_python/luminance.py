from typing import List
from .types import ImageData


def rgb_to_luminance(r: float, g: float, b: float) -> float:
    """Convert RGB to luminance using the standard photometric formula.
    L = 0.2126*R + 0.7152*G + 0.0722*B
    """
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def image_to_luminance_matrix(image_data: ImageData) -> List[List[float]]:
    """Convert RGBA image data to a 2D luminance matrix."""
    width, height, data = image_data.width, image_data.height, image_data.data
    matrix: List[List[float]] = []

    for y in range(height):
        row: List[float] = []
        for x in range(width):
            idx = (y * width + x) * 4  # RGBA: 4 bytes per pixel
            r = data[idx]
            g = data[idx + 1]
            b = data[idx + 2]
            # Alpha channel (idx+3) is ignored
            row.append(rgb_to_luminance(r, g, b))
        matrix.append(row)

    return matrix


def normalise_luminance(luminance_matrix: List[List[float]]) -> List[List[float]]:
    """Normalise luminance values to the range [0, 1]."""
    if not luminance_matrix:
        return []
    if not luminance_matrix[0]:
        return []

    min_val = float('inf')
    max_val = float('-inf')

    for row in luminance_matrix:
        for value in row:
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value

    range_val = max_val - min_val
    if range_val == 0:
        return [[0.0 for _ in row] for row in luminance_matrix]

    return [[(v - min_val) / range_val for v in row] for row in luminance_matrix]


def filter_compression_artifacts(luminance_matrix: List[List[float]]) -> List[List[float]]:
    """Apply a 3x3 high-pass filter to reduce JPEG compression block artifacts."""
    height = len(luminance_matrix)
    if height == 0:
        return []

    width = len(luminance_matrix[0])
    if width == 0 or height < 3 or width < 3:
        return luminance_matrix

    filtered: List[List[float]] = []

    for y in range(height):
        row: List[float] = []
        for x in range(width):
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                row.append(luminance_matrix[y][x])
            else:
                local_mean = (
                    luminance_matrix[y - 1][x - 1] + luminance_matrix[y - 1][x] + luminance_matrix[y - 1][x + 1] +
                    luminance_matrix[y][x - 1]     + luminance_matrix[y][x]     + luminance_matrix[y][x + 1] +
                    luminance_matrix[y + 1][x - 1] + luminance_matrix[y + 1][x] + luminance_matrix[y + 1][x + 1]
                ) / 9.0
                high_freq = luminance_matrix[y][x] - local_mean
                row.append(luminance_matrix[y][x] + high_freq * 0.5)
        filtered.append(row)

    return filtered

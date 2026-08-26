import math
from typing import List
from .types import GradientField


def compute_gradients(luminance_matrix: List[List[float]]) -> GradientField:
    """Compute spatial gradients using central differences (forward/backward at edges)."""
    height = len(luminance_matrix)
    if height == 0:
        return GradientField(gx=[], gy=[], width=0, height=0)

    width = len(luminance_matrix[0])
    if width == 0:
        return GradientField(gx=[], gy=[], width=0, height=0)

    gx: List[List[float]] = []
    gy: List[List[float]] = []

    for y in range(height):
        gx_row: List[float] = []
        gy_row: List[float] = []

        for x in range(width):
            # X-gradient (horizontal)
            if x == 0 and width > 1:
                grad_x = luminance_matrix[y][x + 1] - luminance_matrix[y][x]
            elif x == width - 1 and width > 1:
                grad_x = luminance_matrix[y][x] - luminance_matrix[y][x - 1]
            elif width == 1:
                grad_x = 0.0
            else:
                grad_x = (luminance_matrix[y][x + 1] - luminance_matrix[y][x - 1]) / 2.0

            # Y-gradient (vertical)
            if y == 0 and height > 1:
                grad_y = luminance_matrix[y + 1][x] - luminance_matrix[y][x]
            elif y == height - 1 and height > 1:
                grad_y = luminance_matrix[y][x] - luminance_matrix[y - 1][x]
            elif height == 1:
                grad_y = 0.0
            else:
                grad_y = (luminance_matrix[y + 1][x] - luminance_matrix[y - 1][x]) / 2.0

            gx_row.append(grad_x)
            gy_row.append(grad_y)

        gx.append(gx_row)
        gy.append(gy_row)

    return GradientField(gx=gx, gy=gy, width=width, height=height)


def flatten_gradient_field(gradient_field: GradientField) -> List[List[float]]:
    """Flatten gradient field into an N×2 matrix where each row is [gx, gy] for one pixel."""
    gx, gy = gradient_field.gx, gradient_field.gy
    matrix: List[List[float]] = []

    for y in range(gradient_field.height):
        for x in range(gradient_field.width):
            matrix.append([gx[y][x], gy[y][x]])

    return matrix


def compute_gradient_coherence(gradient_field: GradientField) -> float:
    """Compute coherence of gradient field — ratio of vector sum to scalar sum.
    Higher = more aligned (real images). Lower = chaotic (synthetic images).
    """
    gx, gy = gradient_field.gx, gradient_field.gy
    width, height = gradient_field.width, gradient_field.height

    if width == 0 or height == 0:
        return 0.0

    sum_magnitude = 0.0
    sum_x = 0.0
    sum_y = 0.0

    for y in range(height):
        for x in range(width):
            dx = gx[y][x]
            dy = gy[y][x]
            magnitude = math.sqrt(dx * dx + dy * dy)
            sum_magnitude += magnitude
            sum_x += dx
            sum_y += dy

    if sum_magnitude == 0:
        return 0.0

    vector_sum = math.sqrt(sum_x * sum_x + sum_y * sum_y)
    return vector_sum / sum_magnitude

import pytest
from alogos.gradients import compute_gradients, flatten_gradient_field, compute_gradient_coherence
from alogos.types import GradientField


class TestComputeGradients:
    def test_simple_matrix(self):
        luminance = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        g = compute_gradients(luminance)
        assert g.width == 3
        assert g.height == 3
        assert len(g.gx) == 3
        assert len(g.gy) == 3
        assert abs(g.gx[1][1] - 1.0) < 1e-9
        assert abs(g.gy[1][1] - 0.0) < 1e-9

    def test_edge_cases(self):
        luminance = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        g = compute_gradients(luminance)
        assert g.gx[0][0] == 1   # forward difference
        assert g.gy[0][0] == 3   # forward difference
        assert g.gx[2][2] == 1   # backward difference
        assert g.gy[2][2] == 3   # backward difference

    def test_1x1_matrix(self):
        g = compute_gradients([[5]])
        assert g.width == 1
        assert g.height == 1
        assert g.gx[0][0] == 0
        assert g.gy[0][0] == 0

    def test_empty_matrix(self):
        g = compute_gradients([])
        assert g.width == 0
        assert g.height == 0
        assert g.gx == []
        assert g.gy == []


class TestFlattenGradientField:
    def test_flattens_to_nx2(self):
        gf = GradientField(
            gx=[[1, 2], [3, 4]],
            gy=[[5, 6], [7, 8]],
            width=2, height=2,
        )
        flat = flatten_gradient_field(gf)
        assert len(flat) == 4
        assert flat[0] == [1, 5]
        assert flat[1] == [2, 6]
        assert flat[2] == [3, 7]
        assert flat[3] == [4, 8]

    def test_empty_field(self):
        gf = GradientField(gx=[], gy=[], width=0, height=0)
        assert flatten_gradient_field(gf) == []


class TestComputeGradientCoherence:
    def test_aligned_gradients_high_coherence(self):
        gf = GradientField(gx=[[1, 1], [1, 1]], gy=[[0, 0], [0, 0]], width=2, height=2)
        assert abs(compute_gradient_coherence(gf) - 1.0) < 1e-6

    def test_random_gradients_low_coherence(self):
        gf = GradientField(gx=[[1, -1], [-1, 1]], gy=[[1, -1], [-1, 1]], width=2, height=2)
        assert compute_gradient_coherence(gf) < 0.5

    def test_zero_gradients(self):
        gf = GradientField(gx=[[0, 0], [0, 0]], gy=[[0, 0], [0, 0]], width=2, height=2)
        assert compute_gradient_coherence(gf) == 0.0

import pytest
from alogos.pca import compute_covariance_matrix, perform_pca, compute_pca_score
from alogos.types import PCAResult


class TestComputeCovarianceMatrix:
    def test_simple_data(self):
        data = [[1, 2], [2, 4], [3, 6]]
        cov = compute_covariance_matrix(data)
        assert len(cov) == 2
        assert len(cov[0]) == 2
        assert abs(cov[0][1] - cov[1][0]) < 1e-9  # symmetric
        assert cov[0][0] > 0
        assert cov[1][1] > 0

    def test_uniform_data(self):
        data = [[5, 5], [5, 5], [5, 5]]
        cov = compute_covariance_matrix(data)
        for row in cov:
            for val in row:
                assert abs(val) < 1e-10

    def test_empty_matrix(self):
        assert compute_covariance_matrix([]) == []


class TestPerformPCA:
    def test_computes_components(self):
        data = [
            [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2],
            [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1],
            [1.5, 1.6], [1.1, 0.9],
        ]
        result = perform_pca(data, 2)
        assert len(result.components) > 0
        assert len(result.explained_variance) > 0
        assert len(result.projection) == len(data)
        assert result.explained_variance[0] > 0
        total = sum(result.explained_variance)
        assert abs(total - 1.0) < 0.1

    def test_single_component(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = perform_pca(data, 1)
        assert len(result.components) == 1
        assert len(result.explained_variance) == 1

    def test_empty_data(self):
        result = perform_pca([], 2)
        assert result.components == []
        assert result.explained_variance == []
        assert result.projection == []
        assert result.total_variance == 0.0


class TestComputePCAScore:
    def test_score_in_range(self):
        result = PCAResult(
            components=[[0.707, 0.707]],
            explained_variance=[0.9],
            projection=[1, 2, 3, -1, -2, -3],
            total_variance=10,
        )
        score = compute_pca_score(result)
        assert 0 <= score <= 1

    def test_empty_result_returns_zero(self):
        result = PCAResult(components=[], explained_variance=[], projection=[], total_variance=0)
        assert compute_pca_score(result) == 0.0

    def test_high_variance_concentration_real(self):
        result = PCAResult(
            components=[[1, 0]],
            explained_variance=[0.95, 0.05],
            projection=[1, 1.1, 0.9, 1.2, 0.8],
            total_variance=10,
        )
        assert compute_pca_score(result) < 0.5

    def test_low_variance_concentration_synthetic(self):
        result = PCAResult(
            components=[[1, 0]],
            explained_variance=[0.4, 0.3, 0.3],
            projection=[10, 1, 1, 1, -10, 1],
            total_variance=10,
        )
        assert compute_pca_score(result) > 0.3

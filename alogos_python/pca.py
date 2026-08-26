import math
import random
from typing import List, Tuple
from .types import PCAResult


def _compute_mean(matrix: List[List[float]]) -> List[float]:
    if not matrix:
        return []
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    means = [0.0] * num_cols
    for j in range(num_cols):
        total = sum(matrix[i][j] for i in range(num_rows))
        means[j] = total / num_rows
    return means


def _centre_matrix(matrix: List[List[float]]) -> List[List[float]]:
    means = _compute_mean(matrix)
    return [[v - means[j] for j, v in enumerate(row)] for row in matrix]


def compute_covariance_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Compute covariance matrix C = (1/N) * M^T * M on the centred data."""
    if not matrix:
        return []

    centred = _centre_matrix(matrix)
    num_rows = len(centred)
    num_cols = len(centred[0])

    covariance: List[List[float]] = []
    for i in range(num_cols):
        row: List[float] = []
        for j in range(num_cols):
            total = sum(centred[k][i] * centred[k][j] for k in range(num_rows))
            row.append(total / num_rows)
        covariance.append(row)

    return covariance


def _compute_eigen_decomposition(
    matrix: List[List[float]], num_components: int
) -> Tuple[List[float], List[List[float]]]:
    """Power iteration to find the top eigenvalues/eigenvectors of a symmetric matrix."""
    n = len(matrix)
    if n == 0:
        return [], []

    eigenvalues: List[float] = []
    eigenvectors: List[List[float]] = []

    work = [row[:] for row in matrix]  # copy

    for _ in range(min(num_components, n)):
        # Random init
        vector = [random.random() - 0.5 for _ in range(n)]
        norm = math.sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]

        eigenvalue = 0.0
        for _ in range(100):
            new_vector = [
                sum(work[i][j] * vector[j] for j in range(n))
                for i in range(n)
            ]

            eigenvalue = sum(vector[i] * new_vector[i] for i in range(n))

            norm = math.sqrt(sum(v * v for v in new_vector))
            if norm < 1e-10:
                break

            normalised = [v / norm for v in new_vector]
            diff = sum(abs(vector[i] - normalised[i]) for i in range(n))
            vector = normalised

            if diff < 1e-6:
                break

        eigenvalues.append(eigenvalue)
        eigenvectors.append(vector)

        # Deflate: remove this component from the work matrix
        for i in range(n):
            for j in range(n):
                work[i][j] -= eigenvalue * vector[i] * vector[j]

    return eigenvalues, eigenvectors


def perform_pca(data: List[List[float]], num_components: int = 5) -> PCAResult:
    """Perform PCA on the data matrix and return components, variance, and projection."""
    if not data:
        return PCAResult(components=[], explained_variance=[], projection=[], total_variance=0.0)

    centred = _centre_matrix(data)
    covariance = compute_covariance_matrix(data)
    eigenvalues, eigenvectors = _compute_eigen_decomposition(covariance, num_components)

    total_variance = sum(abs(v) for v in eigenvalues)

    explained_variance = [
        abs(v) / total_variance if total_variance > 0 else 0.0
        for v in eigenvalues
    ]

    projection: List[float] = []
    if eigenvectors:
        first = eigenvectors[0]
        for row in centred:
            proj = sum(row[j] * first[j] for j in range(len(first)))
            projection.append(proj)

    return PCAResult(
        components=eigenvectors,
        explained_variance=explained_variance,
        projection=projection,
        total_variance=total_variance,
    )


def compute_pca_score(pca_result: PCAResult) -> float:
    """Compute a synthetic likelihood score from PCA results.
    Higher values indicate more likely synthetic. Score is in [0, 1].
    """
    explained_variance = pca_result.explained_variance
    projection = pca_result.projection

    if not explained_variance or not projection:
        return 0.0

    primary_variance = explained_variance[0]

    n = len(projection)
    mean = sum(projection) / n
    variance = sum((v - mean) ** 2 for v in projection) / n
    std_dev = math.sqrt(variance)

    kurtosis = 0.0
    if std_dev > 0:
        fourth_moment = sum(((v - mean) / std_dev) ** 4 for v in projection) / n
        kurtosis = fourth_moment - 3  # excess kurtosis

    score = (1 - primary_variance) * 0.6 + max(0.0, kurtosis) * 0.4
    return max(0.0, min(1.0, score))


def compute_pca_kurtosis(pca_result: PCAResult) -> float:
    # Mirrors the internal calculation inside compute_pca_score() so it can
    # be exposed separately for evaluation/reporting, without modifying
    # compute_pca_score itself (kept exactly as validated against the
    # original JS/TS implementation).
    projection = pca_result.projection
    if not projection:
        return 0.0
    n = len(projection)
    mean = sum(projection) / n
    variance = sum((v - mean) ** 2 for v in projection) / n
    std_dev = math.sqrt(variance)
    kurtosis = 0.0
    if std_dev > 0:
        fourth_moment = sum(((v - mean) / std_dev) ** 4 for v in projection) / n
        kurtosis = fourth_moment - 3
    return kurtosis

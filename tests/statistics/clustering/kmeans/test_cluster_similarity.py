"""Tests for pairwise similarities derived from fitted K-Means centres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

from pyspoc._utils import numerical as numerical_utils
from pyspoc.statistics.clustering.kmeans.statistics import (
    KMeansClusterSimilarity,
)


@dataclass
class ModelStub:
    """Minimal fitted model exposing deterministic cluster centres."""

    cluster_centers_: np.ndarray


@dataclass
class EstimatorStub:
    """Minimal estimator exposing a fitted model to the Statistic."""

    model: ModelStub

    def _get_model(self) -> ModelStub:
        return self.model


CENTRES = np.array(
    [
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 2.0],
    ],
)

DATA = np.array(
    [
        [-1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        [4.0, 3.0, 5.0],
        [5.0, 6.0, 4.0],
        [6.0, 5.0, 7.0],
        [7.0, 8.0, 6.0],
    ],
)


def make_estimator(centres: np.ndarray = CENTRES) -> EstimatorStub:
    """Return an estimator stub containing a defensive centre copy."""
    return EstimatorStub(ModelStub(centres.copy()))


def assert_similarity_contract(result: np.ndarray, k: int) -> None:
    """Assert the shared mathematical contract for a similarity matrix."""
    assert result.shape == (k, k)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), np.ones(k))
    assert np.isfinite(result).all()
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_metadata_properties_are_stable_and_read_only() -> None:
    """Metadata properties should expose immutable configuration values."""
    statistic = KMeansClusterSimilarity(k=2)

    assert statistic.name == "KMeans - Cluster Similarity"
    assert statistic.identifier == "kmeans-cs"
    assert statistic.labels == ("non-linear",)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_at_least_two_clusters_are_required(k: int) -> None:
    """A pairwise cluster matrix requires at least two fitted clusters."""
    with pytest.raises(ValueError, match="k"):
        KMeansClusterSimilarity(k=k)


@pytest.mark.parametrize("gamma", [-1.0, 0.0])
def test_gamma_must_be_strictly_positive(gamma: float) -> None:
    """Explicit decay coefficients should be strictly positive."""
    with pytest.raises(ValueError, match="gamma"):
        KMeansClusterSimilarity(k=2, gamma=gamma)


def test_unsupported_kernel_is_rejected_by_runtime_type_checking() -> None:
    """Constructor type checking should reject unknown kernel names."""
    with pytest.raises(TypeError, match="kernel"):
        KMeansClusterSimilarity(k=2, kernel="unknown")  # type: ignore[arg-type]


def test_default_gamma_is_inverse_feature_count() -> None:
    """An omitted gamma should resolve from the computed dataset."""
    statistic = KMeansClusterSimilarity(k=3, kernel="rbf")

    result = statistic._get_result(DATA, make_estimator())
    expected = rbf_kernel(CENTRES, gamma=1 / DATA.shape[1])

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "kernel",
    [
        "rbf",
        "mahalanobis_rbf",
        "laplacian",
        "inverse_distance",
        "mahalanobis_inverse_distance",
        "cosine",
        "correlation",
    ],
)
def test_every_kernel_satisfies_similarity_matrix_contract(kernel: str) -> None:
    """Every supported kernel should produce a bounded symmetric matrix."""
    statistic = KMeansClusterSimilarity(
        k=3,
        kernel=kernel,  # type: ignore[arg-type]
        gamma=0.25,
    )

    result = statistic._get_result(DATA, make_estimator())

    assert_similarity_contract(result, k=3)


def test_laplacian_delegates_to_sklearn_definition() -> None:
    """Laplacian similarity should use L1 distance and the configured gamma."""
    statistic = KMeansClusterSimilarity(
        k=3,
        kernel="laplacian",
        gamma=0.4,
    )

    result = statistic._get_result(DATA, make_estimator())

    np.testing.assert_allclose(
        result,
        laplacian_kernel(CENTRES, gamma=0.4),
    )


def test_inverse_distance_uses_each_unordered_pair_once() -> None:
    """Inverse-distance output should be mirrored without changing values."""
    statistic = KMeansClusterSimilarity(
        k=3,
        kernel="inverse_distance",
        gamma=0.5,
    )

    result = statistic._get_result(DATA, make_estimator())

    expected_01 = 1 / (1 + 0.5 * np.linalg.norm(CENTRES[0] - CENTRES[1]))
    assert result[0, 1] == pytest.approx(expected_01)
    assert result[1, 0] == pytest.approx(expected_01)
    assert_similarity_contract(result, k=3)


def test_mahalanobis_rbf_uses_feature_covariance_whitening() -> None:
    """Mahalanobis RBF should equal RBF on pseudo-whitened centres."""
    statistic = KMeansClusterSimilarity(
        k=3,
        kernel="mahalanobis_rbf",
        gamma=0.3,
    )

    result = statistic._get_result(DATA, make_estimator())
    covariance = np.atleast_2d(np.cov(DATA, rowvar=False))
    inverse_sqrt = numerical_utils.spsd_matrix_power(covariance, -0.5)
    expected = rbf_kernel(CENTRES @ inverse_sqrt, gamma=0.3)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1.0),
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 0.0),
        ([4.0, 4.0, 4.0], [4.0, 4.0, 4.0], 1.0),
        ([4.0, 4.0, 4.0], [1.0, 2.0, 3.0], 0.5),
        ([4.0, 4.0, 4.0], [7.0, 7.0, 7.0], 0.5),
    ],
)
def test_correlation_similarity_handles_profile_edge_cases(
    x: list[float], y: list[float], expected: float
) -> None:
    """Correlation should define deterministic zero-variance conventions."""
    statistic = KMeansClusterSimilarity(k=2, kernel="correlation")

    result = statistic._compute_manual_kernel(
        np.asarray(x),
        np.asarray(y),
        kernel="correlation",
        gamma=1.0,
    )

    assert result == pytest.approx(expected)


def test_singular_covariance_uses_pseudo_inverse_square_root() -> None:
    """Mahalanobis kernels should remain finite for dependent features."""
    singular_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [9.0, 18.0, 27.0],
            [10.0, 20.0, 30.0],
            [11.0, 22.0, 33.0],
        ],
    )
    centres = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
    )

    for kernel in ("mahalanobis_rbf", "mahalanobis_inverse_distance"):
        statistic = KMeansClusterSimilarity(
            k=2,
            kernel=kernel,  # type: ignore[arg-type]
            gamma=0.2,
        )
        result = statistic._get_result(
            singular_data,
            make_estimator(centres),
        )

        assert_similarity_contract(result, k=2)


def test_single_feature_mahalanobis_similarity_uses_matrix_covariance() -> None:
    """Scalar covariance output should be promoted to a one-by-one matrix."""
    data = np.array([[0.0], [1.0], [4.0], [5.0]])
    centres = np.array([[0.5], [4.5]])
    statistic = KMeansClusterSimilarity(
        k=2,
        kernel="mahalanobis_rbf",
        gamma=0.5,
    )

    result = statistic._get_result(data, make_estimator(centres))

    assert_similarity_contract(result, k=2)


def test_compute_integrates_with_cached_kmeans_estimator() -> None:
    """The public computation path should fit K-Means and return similarities."""
    data = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
        ],
    )
    statistic = KMeansClusterSimilarity(
        k=2,
        kernel="rbf",
        gamma=0.2,
        random_seed=0,
    )

    result = statistic.compute(data)

    assert_similarity_contract(result, k=2)
    assert statistic._estimator_._get_model().cluster_centers_.shape == (2, 2)


@pytest.mark.parametrize(
    ("method_name", "kernel"),
    [
        ("_compute_sklearn_kernel", "inverse_distance"),
        ("_compute_manual_kernel", "rbf"),
    ],
)
def test_private_kernel_helpers_reject_incompatible_dispatch(method_name: str, kernel: str) -> None:
    """Internal helpers should fail clearly when routed an invalid kernel."""
    statistic = KMeansClusterSimilarity(k=2)
    method: Any = getattr(statistic, method_name)

    with pytest.raises(ValueError, match="Unsupported kernel"):
        if method_name == "_compute_sklearn_kernel":
            method(CENTRES, kernel, gamma=1.0)
        else:
            method(CENTRES[0], CENTRES[1], kernel, gamma=1.0)

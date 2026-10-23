"""Validation of the current K-Medoids adapter."""
from importlib import import_module

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel


pytestmark = pytest.mark.filterwarnings("error::typeguard.TypeHintWarning")


def _classes():
    stats = import_module("pyspoc.statistics.clustering.kclustering.kmedoids.statistics")
    bases = import_module("pyspoc.statistics.clustering.kclustering.kmedoids._base")
    return stats.KMedoidsSimilarity, bases.KMedoidsEstimator


def test_current_kmedoids_adapter_is_importable():
    statistic, estimator = _classes()
    from pyspoc.statistics.base import Statistic
    assert issubclass(statistic, Statistic)
    assert callable(estimator.get_or_create)


@pytest.mark.parametrize("k", [0, -1])
def test_kmedoids_rejects_nonpositive_cluster_count(k):
    statistic, _ = _classes()
    with pytest.raises((TypeError, ValueError)):
        statistic(k=k)


def test_kmedoids_similarity_matches_fitted_medoids():
    statistic, _ = _classes()
    data = np.random.default_rng(18).normal(size=(24, 3))
    operation = statistic(k=2, random_seed=7, kernel="rbf", gamma=0.3)
    result = operation.compute(data)
    model = operation._estimator_._get_model()
    np.testing.assert_allclose(result, rbf_kernel(model.cluster_centers_, gamma=0.3))
    assert result.shape == (2, 2)
    assert all(any(np.array_equal(centre, row) for row in data)
               for centre in model.cluster_centers_)


def test_kmedoids_kernel_is_runtime_validated():
    statistic, _ = _classes()
    with pytest.raises((TypeError, ValueError)):
        statistic(k=2, kernel="not-a-kernel")


@pytest.fixture(autouse=True)
def isolate_estimator_cache():
    """Prevent cached fits from depending on test order or prior modules."""
    _, estimator = _classes()
    estimator._reset_cache()
    yield
    estimator._reset_cache()


@pytest.mark.parametrize("kernel", [
    "rbf", "mahalanobis_rbf", "laplacian", "inverse_distance",
    "mahalanobis_inverse_distance", "cosine", "correlation",
])
def test_supported_kernels_return_bounded_symmetric_similarities(kernel):
    """All kernels must retain the shared k-by-k similarity contract."""
    statistic, _ = _classes()
    data = np.random.default_rng(18).normal(size=(24, 4))
    operation = statistic(k=3, random_seed=7, kernel=kernel, gamma=0.3)

    result = operation.compute(data)

    assert result.shape == (3, 3)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 1)
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.parametrize("dim", ["p", "n"])
def test_similarity_uses_medoid_rows_from_the_selected_orientation(dim):
    """Dimension selection changes fitting data, not the k-by-k output shape."""
    statistic, _ = _classes()
    data = np.random.default_rng(23).normal(size=(16, 5))
    original = data.copy()
    oriented = data if dim == "p" else data.T
    operation = statistic(k=2, dim=dim, random_seed=7, gamma=0.3)

    result = operation.compute(data)
    model = operation._estimator_._get_model()

    np.testing.assert_array_equal(model.cluster_centers_, oriented[model.medoid_indices_])
    np.testing.assert_allclose(result, rbf_kernel(model.cluster_centers_, gamma=0.3))
    np.testing.assert_array_equal(data, original)


def test_similarity_options_reuse_the_same_fitted_estimator():
    """Gamma changes similarity extraction, so it must not duplicate the fit."""
    statistic, _ = _classes()
    data = np.random.default_rng(23).normal(size=(16, 5))
    first = statistic(k=2, random_seed=7, gamma=0.3)
    second = statistic(k=2, random_seed=7, gamma=0.8)

    first_result = first.compute(data)
    second_result = second.compute(data.copy())

    assert first._estimator_ is second._estimator_
    centres = first._estimator_._get_model().cluster_centers_
    np.testing.assert_allclose(first_result, rbf_kernel(centres, gamma=0.3))
    np.testing.assert_allclose(second_result, rbf_kernel(centres, gamma=0.8))


def test_similarity_metadata_identifies_kmedoids():
    """The single-fit statistic should expose its own family-specific metadata."""
    statistic, _ = _classes()
    operation = statistic(k=2)
    assert operation.get_identifier() == "kmedoids-cs"
    assert operation.get_long_name() == "KMedoids - Cluster Similarity"

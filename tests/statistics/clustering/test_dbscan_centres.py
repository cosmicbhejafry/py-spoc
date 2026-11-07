"""Centroid extraction from real DBSCAN fits, including noise and border points."""

import numpy as np
import pytest

from pyspoc.settings import settings
from pyspoc.statistics.clustering.dbscan._base import DBScanEstimator


def make_estimator(eps=0.111, min_samples=3):
    return DBScanEstimator(
        eps=eps, min_samples=min_samples, algorithm="brute", leaf_size=30,
        metric="euclidean", minkowski_p=None, n_jobs=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_centres_include_border_points_and_exclude_noise(dtype):
    # Endpoints are border observations; averaging only core points produces a
    # different answer. The far-away final observation is noise, not a cluster.
    data = np.array([[0.], [0.1], [0.2], [0.29],
                     [10.], [10.1], [10.2], [10.29], [100.]], dtype=dtype)
    original = data.copy()
    with settings.override(max_worker_threads=2):
        estimator = make_estimator().fit(data)
    model = estimator._get_model()
    assert set(model.labels_) == {-1, 0, 1}
    assert len(model.core_sample_indices_) < np.count_nonzero(model.labels_ >= 0)

    result = estimator._get_centres()
    expected = np.vstack([
        data[model.labels_ == label].mean(axis=0, dtype=np.float64) for label in (0, 1)
    ])
    assert result.shape == (2, 1)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(data, original)


def test_all_noise_returns_empty_matrix_with_feature_axis():
    data = np.array([[0., 0.], [10., 10.], [20., 20.]])
    with settings.override(max_worker_threads=2):
        estimator = make_estimator().fit(data)
    result = estimator._get_centres()
    assert result.shape == (0, 2)
    assert result.dtype == np.float64


def test_one_cluster_returns_one_centroid_row():
    data = np.array([[0., 1.], [0.1, 1.], [0.2, 1.]])
    with settings.override(max_worker_threads=2):
        estimator = make_estimator(eps=1.0).fit(data)
    result = estimator._get_centres()
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, data.mean(axis=0, keepdims=True))


def test_centres_require_fitted_model():
    with settings.override(max_worker_threads=2):
        estimator = make_estimator()
    with pytest.raises(ValueError, match="fit"):
        estimator._get_centres()

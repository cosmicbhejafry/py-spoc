"""Nearest-core DBSCAN prediction on new, filtered and reordered observations."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from pyspoc.settings import settings
from pyspoc.statistics.clustering.dbscan._base import DBScanEstimator


def fitted_estimator(data, **overrides):
    """Fit independently of the shared estimator cache and external worker settings."""
    options = dict(eps=0.3, min_samples=2, algorithm="brute", leaf_size=30,
                   metric="euclidean", minkowski_p=None, n_jobs=1)
    options.update(overrides)
    with settings.override(max_worker_threads=2):
        estimator = DBScanEstimator(**options)
        estimator.fit(data)
    return estimator


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("metric,p", [
    ("euclidean", None), ("manhattan", None),
    ("minkowski", 0.5), ("minkowski", 1.5), ("minkowski", 3.),
])
def test_new_observations_match_independent_nearest_core_distances(dtype, metric, p):
    training = np.array([[0., 0.], [.1, 0.], [5., 5.], [5.1, 5.], [100., 100.]], dtype=dtype)
    queries = np.array([[5.05, 5.05], [.05, .05], [2., 2.]], dtype=dtype)
    estimator = fitted_estimator(training, metric=metric, minkowski_p=p)
    model = estimator._get_model()
    original_training, original_queries = training.copy(), queries.copy()
    original_labels = model.labels_.copy()
    kwargs = {"p": p} if metric == "minkowski" else {}
    distances = pairwise_distances(queries, model.components_, metric=metric, **kwargs)
    nearest = distances.argmin(axis=1)
    expected = np.where(distances[np.arange(len(queries)), nearest] <= model.eps,
                        model.labels_[model.core_sample_indices_[nearest]], -1)

    # Small working memory forces chunk boundaries without changing results.
    from sklearn import config_context
    with config_context(working_memory=0.0001):
        result = estimator._predict(queries)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, [1, 0, -1])
    assert result.shape == (len(queries),)
    assert np.issubdtype(result.dtype, np.integer)
    np.testing.assert_array_equal(training, original_training)
    np.testing.assert_array_equal(queries, original_queries)
    np.testing.assert_array_equal(model.labels_, original_labels)


def test_training_subsets_and_reordering_are_not_returned_in_training_order():
    training = np.array([[0., 0.], [.1, 0.], [5., 5.], [5.1, 5.], [100., 100.]])
    estimator = fitted_estimator(training)
    order = np.array([4, 3, 1, 2, 0])
    np.testing.assert_array_equal(estimator._predict(training[order]), [-1, 1, 0, 1, 0])
    subset = training[[3, 1]]
    subset.flags.writeable = False
    np.testing.assert_array_equal(estimator._predict(subset), [1, 0])


def test_eps_boundary_is_inclusive_and_exact_ties_use_training_order():
    training = np.array([[-1., 0.], [-1., 0.], [1., 0.], [1., 0.]])
    estimator = fitted_estimator(training, eps=1.)
    # The middle query is exactly eps from two disconnected clusters. The
    # earliest core training row wins; querying does not merge the clusters.
    queries = np.array([[0., 0.], [-2., 0.], [2., 0.], [0., 1.]])
    np.testing.assert_array_equal(estimator._predict(queries), [0, 0, 1, -1])


def test_border_points_do_not_extend_cluster_support():
    estimator = fitted_estimator(np.array([[0.], [.1], [.2], [.29]]),
                                 eps=.111, min_samples=3)
    assert 0 not in estimator._get_model().core_sample_indices_
    # Near the border point at 0, but outside eps of every core point.
    np.testing.assert_array_equal(estimator._predict(np.array([[-.1]])), [-1])


def test_all_noise_and_empty_queries():
    estimator = fitted_estimator(np.array([[0., 0.], [10., 10.]]))
    np.testing.assert_array_equal(estimator._predict(np.zeros((3, 2))), [-1, -1, -1])
    assert estimator._predict(np.empty((0, 2))).shape == (0,)


def test_prediction_validates_feature_count_even_without_core_points():
    estimator = fitted_estimator(np.array([[0., 0.], [10., 10.]]))
    with pytest.raises(ValueError, match="features"):
        estimator._predict(np.zeros((1, 3)))


def test_prediction_rejects_nonfinite_queries():
    estimator = fitted_estimator(np.array([[0., 0.], [.1, 0.]]))
    with pytest.raises(ValueError):
        estimator._predict(np.array([[np.nan, 0.]]))

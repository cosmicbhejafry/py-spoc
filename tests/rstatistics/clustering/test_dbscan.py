"""Public DBSCAN sweep regressions, excluding noise/signal dispersion measures."""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from pyspoc.rstatistics.clustering.dbscan.statistics import (
    DBScanDispersionSummaries, DBScanSILScores,
)
from pyspoc.statistics.clustering.dbscan._base import DBScanEstimator


pytestmark = pytest.mark.filterwarnings("error::typeguard.TypeHintWarning")
SUMMARIES = ("mean", "median", "sum", "min", "max", "stdev", "iqr", "range")
MEASURES = ("within-cluster", "across-cluster", "within-across-ratio")


@pytest.fixture(autouse=True)
def isolate_cache():
    DBScanEstimator._reset_cache()
    yield
    DBScanEstimator._reset_cache()


@pytest.fixture
def data():
    """Two compact clusters plus two distant noise observations."""
    return np.array([[0., 0.], [0., .1], [.1, 0.],
                     [5., 5.], [5., 5.1], [5.1, 5.], [100., 100.], [200., 200.]])


def make_operation(kind="dispersion", **overrides):
    options = dict(min_eps=.2, max_eps=.3, n_results=2, min_samples=2,
                   algorithm="brute", leaf_size=30, metric="euclidean",
                   minkowski_p=None, n_jobs=1)
    options.update(overrides)
    cls = DBScanDispersionSummaries if kind == "dispersion" else DBScanSILScores
    return cls(**options)


def summarize(array, name):
    if name == "iqr":
        return np.percentile(array, 75) - np.percentile(array, 25)
    if name == "range":
        return np.ptp(array)
    return (np.std if name == "stdev" else getattr(np, name))(array)


def expected_dispersion(data, labels, measure, inner, outer, p):
    """Independent direct distances; noise is never treated as a cluster."""
    keep = labels >= 0
    data, labels = data[keep], labels[keep]
    groups = np.unique(labels)
    if not len(groups):
        return np.nan
    within, across = [], []
    for label in groups:
        own, other = data[labels == label], data[labels != label]
        i, j = np.triu_indices(len(own), 1)
        d = np.linalg.norm(own[i] - own[j], ord=p, axis=1)
        within.append(summarize(d, inner) if d.size else 0.)
        d = np.linalg.norm(own[:, None, :] - other[None, :, :], ord=p, axis=2).ravel()
        # Equivalent multisets should yield identical summaries regardless of
        # which cluster is on the first axis of the independent calculation.
        d.sort()
        across.append(summarize(d, inner) if d.size else 0.)
    w, a = summarize(np.asarray(within), outer), summarize(np.asarray(across), outer)
    if measure == "within-cluster":
        return w
    if measure == "across-cluster":
        return a if len(groups) > 1 else np.nan
    # A single cluster has an undefined across score, but the explicitly
    # chosen ratio convention treats its denominator as zero.
    if len(groups) == 1:
        a = 0.
    return 1. if w == a == 0 else np.divide(w, a)


@pytest.mark.parametrize("inner", SUMMARIES)
@pytest.mark.parametrize("outer", SUMMARIES)
@pytest.mark.parametrize("measure", MEASURES)
def test_dispersion_matches_independent_signal_cluster_calculation(data, inner, outer, measure):
    operation = make_operation(inner_summary=inner, outer_summary=outer,
                               dispersion_measure=measure, dispersion_minkowski_p=2.)
    original = data.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = [expected_dispersion(data, DBSCAN(eps=eps, min_samples=2).fit_predict(data),
                    measure, inner, outer, 2.) for eps in (.2, .3)]
        result = operation.compute(data)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert result.shape == (2,)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("case", ["all-noise", "one-cluster", "one-plus-noise",
                                  "two-clusters", "two-plus-noise"])
def test_signal_only_silhouette_and_undefined_fits(data, case):
    matrix = data if case == "two-plus-noise" else data[:6]
    if case == "one-plus-noise":
        matrix = np.vstack((data[:3], data[-2:]))
    eps = .001 if case == "all-noise" else 1000. if case == "one-cluster" else .2
    operation = make_operation("silhouette", min_eps=eps, max_eps=eps, n_results=1)
    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(matrix)
    keep = labels >= 0
    expected = (silhouette_score(matrix[keep], labels[keep])
                if len(np.unique(labels[keep])) > 1 else np.nan)
    np.testing.assert_allclose(operation.compute(matrix), [expected])


@pytest.mark.parametrize("inner", SUMMARIES)
@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize("case", ["all-noise", "one-cluster", "zero-within"])
def test_degenerate_dispersion_conventions(data, inner, measure, case):
    matrix = np.zeros((3, 2)) if case == "zero-within" else data[:6]
    eps = .001 if case == "all-noise" else 1000.
    operation = make_operation(min_eps=eps, max_eps=eps, n_results=1,
                               inner_summary=inner, dispersion_measure=measure)
    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = expected_dispersion(matrix, labels, measure, inner, "mean", 2.)
        result = operation.compute(matrix)
    np.testing.assert_allclose(result, [expected])


@pytest.mark.parametrize("p", [.5, 1., 2., 3.])
@pytest.mark.parametrize("mode", ["python", "numba"])
def test_dispersion_metric_and_dispatch_modes(data, p, mode):
    from pyspoc.settings import settings

    labels = DBSCAN(eps=.2, min_samples=2).fit_predict(data)
    operation = make_operation(dispersion_measure="across-cluster", dispersion_minkowski_p=p)
    with settings.override(numba_mode=mode):
        result = operation.compute(data)
    expected = expected_dispersion(data, labels, "across-cluster", "mean", "mean", p)
    np.testing.assert_allclose(result, [expected, expected])


@pytest.mark.parametrize("policy", ["row-omit", "col-omit"])
def test_nan_omission_preserves_input(data, policy):
    data[0, 0] = np.nan
    original = data.copy()
    clean = (data[~np.isnan(data).any(axis=1)] if policy == "row-omit"
             else data[:, ~np.isnan(data).any(axis=0)])
    expected = make_operation().compute(clean)
    np.testing.assert_allclose(make_operation(nan_policy=policy).compute(data), expected)
    np.testing.assert_array_equal(data, original)


def test_sweep_continues_after_undefined_silhouette_fit(data):
    """The first eps gives all noise; later eps values must still be processed."""
    result = make_operation("silhouette", min_eps=.001, max_eps=.3, n_results=3).compute(data)
    assert result.shape == (3,)
    assert np.isnan(result[0])
    assert np.isfinite(result[1:]).all()

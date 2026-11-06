"""Numerical and configuration contracts for K-clustering summary sweeps."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

from pyspoc.rstatistics.clustering.kclustering.kmeans.statistics import (
    KMeansDispersionSummaries, KMeansInertia, KMeansSILScores,
)
from pyspoc.rstatistics.clustering.kclustering.kmedoids.statistics import (
    KMedoidsDispersionSummaries, KMedoidsInertia, KMedoidsSILScores,
)
from pyspoc.statistics.clustering.kclustering.kmeans._base import KMeansEstimator
from pyspoc.statistics.clustering.kclustering.kmedoids._base import KMedoidsEstimator
from pyspoc.rstatistics.clustering._mixin import ClusteringDispersionSummaryMixin


pytestmark = pytest.mark.filterwarnings("error::typeguard.TypeHintWarning")

SUMMARIES = ("mean", "median", "sum", "min", "max", "stdev", "iqr", "range")
DISTANCES = ("within-cluster", "across-cluster", "within-across-ratio")


@pytest.fixture(params=["kmeans", "kmedoids"])
def family(request):
    """Exercise the same contracts through both concrete estimator families."""
    estimator = KMeansEstimator if request.param == "kmeans" else KMedoidsEstimator
    estimator._reset_cache()
    yield request.param
    estimator._reset_cache()


@pytest.fixture
def data():
    """Non-square, nondegenerate data makes orientation and summaries distinct."""
    return np.random.default_rng(19).normal(size=(16, 5))


def make_operation(family, kind, **overrides):
    """Use fixed inexpensive fits and retain each family's public argument names."""
    classes = {
        "kmeans": (KMeansSILScores, KMeansDispersionSummaries, KMeansInertia),
        "kmedoids": (KMedoidsSILScores, KMedoidsDispersionSummaries, KMedoidsInertia),
    }
    options = dict(min_k=2, max_k=3, max_iter=40, random_seed=17,
                   initializer="random")
    options.update(dict(algorithm="lloyd", tol=0.02) if family == "kmeans"
                   else dict(algorithm="alternate", metric="euclidean"))
    if kind == "dispersion":
        options.update(inner_summary="mean", outer_summary="mean",
                       dispersion_measure="within-cluster", dispersion_minkowski_p=2.)
    options.update(overrides)
    index = {"silhouette": 0, "dispersion": 1, "inertia": 2}[kind]
    return classes[family][index](**options)


def reference_model(family, data, k, metric="euclidean"):
    """Fit independently rather than using pySPoC's estimator or result helpers."""
    common = dict(n_clusters=k, max_iter=40, random_state=17, init="random")
    if family == "kmeans":
        model = KMeans(**common, algorithm="lloyd", tol=0.02, n_init="auto")
    else:
        model = KMedoids(**common, method="alternate", metric=metric)
    return model.fit(data)


def summarize(values, summary):
    """Explicit independent definitions, including population standard deviation."""
    if summary == "iqr":
        return np.percentile(values, 75) - np.percentile(values, 25)
    if summary == "range":
        return np.ptp(values)
    function = np.std if summary == "stdev" else getattr(np, summary)
    return function(values)


def reference_dispersion(data, labels, distance, inner, outer):
    """Build pair distances independently, then summarize within and across clusters."""
    within, across = [], []
    for label in np.unique(labels):
        points = data[labels == label]
        i, j = np.triu_indices(len(points), k=1)
        distances = np.linalg.norm(points[i] - points[j], axis=1)
        # Singleton dispersion is zero even for reductions such as min/quantile.
        within.append(summarize(distances, inner) if len(distances) else 0.)
        other = data[labels != label]
        distances = np.linalg.norm(points[:, None, :] - other[None, :, :], axis=2).ravel()
        # With two clusters both groups see the same distance multiset, in
        # different row orders. Canonical ordering keeps their mean/sum exactly
        # equal, avoiding spurious nonzero outer dispersion from summation order.
        distances.sort()
        across.append(summarize(distances, inner) if len(distances) else 0.)
    within_summary = summarize(np.asarray(within), outer)
    across_summary = summarize(np.asarray(across), outer)
    if distance == "within-cluster":
        return within_summary
    if distance == "across-cluster":
        return across_summary
    if within_summary == across_summary == 0:
        return 1.
    return within_summary / across_summary


@pytest.mark.parametrize("dim", ["p", "n"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_silhouette_matches_independent_fits(family, data, dim, metric):
    """The inclusive k sweep must score predicted labels with the requested metric."""
    original = data.copy()
    operation = make_operation(family, "silhouette", dim=dim, silhouette_metric=metric)
    oriented = data if dim == "p" else data.T
    expected = [silhouette_score(oriented, reference_model(family, oriented, k).predict(oriented),
                                 metric=metric) for k in (2, 3)]
    result = operation.compute(data)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected)
    assert np.all((result >= -1) & (result <= 1))
    np.testing.assert_array_equal(data, original)


def test_kmedoids_silhouette_defaults_to_clustering_metric(data):
    """Omitting silhouette_metric must inherit a non-Euclidean fit metric."""
    KMedoidsEstimator._reset_cache()
    try:
        operation = make_operation("kmedoids", "silhouette", metric="manhattan")
        expected = [silhouette_score(data, reference_model("kmedoids", data, k,
                    metric="manhattan").predict(data), metric="manhattan") for k in (2, 3)]
        np.testing.assert_allclose(operation.compute(data), expected)
    finally:
        KMedoidsEstimator._reset_cache()


@pytest.mark.parametrize("dim", ["p", "n"])
@pytest.mark.parametrize("inner", SUMMARIES)
@pytest.mark.parametrize("outer", SUMMARIES)
@pytest.mark.parametrize("distance", DISTANCES)
def test_dispersion_matches_independent_calculation(family, data, dim, inner, outer, distance):
    """Cover all inner/outer summaries, both orientations and each distance measure."""
    operation = make_operation(family, "dispersion", dim=dim, inner_summary=inner,
                               outer_summary=outer, dispersion_measure=distance)
    original = data.copy()
    oriented = data if dim == "p" else data.T
    # Zero denominators are not regular finite-value comparisons. Keep NaN/inf
    # checks active below, but suppress the expected arithmetic warnings here.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = [reference_dispersion(
            oriented, reference_model(family, oriented, k).predict(oriented),
            distance, inner, outer,
        ) for k in (2, 3)]
        result = operation.compute(data)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("numerator,expected", [(0., 1.), (1e-9, np.inf), (1., np.inf)])
def test_zero_denominator_ratio_convention(family, numerator, expected):
    """Only exact zero/zero means one; nonzero/zero retains infinity."""
    operation = make_operation(family, "dispersion", dispersion_measure="within-across-ratio")
    # Isolate aggregation from fitting: no across pairs represents a single
    # cluster, while its within summary can be independently controlled.
    estimator = SimpleNamespace(
        _predict=lambda data: np.zeros(len(data), dtype=int),
        _compute_within_cluster_minkowski_distances=lambda *args, **kwargs: {
            0: np.array([numerator])},
        _compute_cross_cluster_minkowski_distances=lambda *args, **kwargs: {},
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = ClusteringDispersionSummaryMixin._get_result(
            operation, np.zeros((2, 2)), estimator, dispersion_measure="within-across-ratio")
    assert result == expected


@pytest.mark.parametrize("kind", ["silhouette", "dispersion"])
def test_single_k_preserves_vector_output(family, data, kind):
    """One score remains a length-one vector rather than becoming a scalar."""
    result = make_operation(family, kind, min_k=2, max_k=2).compute(data)
    assert result.shape == (1,)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("kind", ["silhouette", "dispersion"])
@pytest.mark.parametrize("dim", ["p", "n"])
@pytest.mark.parametrize("policy", ["row-omit", "col-omit"])
def test_nan_omission_matches_clean_input(family, data, kind, dim, policy):
    """Missing values are removed after orientation, without changing caller data."""
    data[0, 0] = np.nan
    original = data.copy()
    oriented = data if dim == "p" else data.T
    clean = (oriented[~np.isnan(oriented).any(axis=1)] if policy == "row-omit"
             else oriented[:, ~np.isnan(oriented).any(axis=0)])
    expected = make_operation(family, kind).compute(clean)
    result = make_operation(family, kind, dim=dim, nan_policy=policy).compute(data)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kind", ["silhouette", "dispersion"])
def test_strict_nan_policy_rejects_missing_values(family, data, kind):
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        make_operation(family, kind, nan_policy="raise").compute(data)


@pytest.mark.parametrize("kind", ["silhouette", "dispersion"])
@pytest.mark.parametrize("bounds", [dict(min_k=1), dict(min_k=4, max_k=3)])
def test_invalid_bounds_are_rejected(family, kind, bounds):
    with pytest.raises(ValueError):
        make_operation(family, kind, **bounds)


@pytest.mark.parametrize("kind", ["silhouette", "dispersion"])
def test_sweep_cannot_exceed_oriented_sample_count(family, data, kind):
    with pytest.raises(ValueError, match="number of data points"):
        make_operation(family, kind, dim="n", max_k=6).compute(data)


def test_silhouette_rejects_one_cluster_per_observation(family, data):
    """Silhouette is undefined at k=n even though clustering itself permits it."""
    with pytest.raises(ValueError, match="labels"):
        make_operation(family, "silhouette", min_k=5, max_k=5, dim="n").compute(data)


@pytest.mark.parametrize("argument", ["inner_summary", "outer_summary", "dispersion_measure",
                                     "silhouette_metric"])
def test_invalid_extraction_options_are_rejected(family, argument):
    kind = "silhouette" if argument == "silhouette_metric" else "dispersion"
    with pytest.raises((TypeError, ValueError), match=argument):
        make_operation(family, kind, **{argument: "unknown"})


def test_summary_options_share_fits_with_inertia_sweep(family, data):
    """Extraction settings must not leak into fitted-estimator cache identity."""
    operations = [make_operation(family, "inertia"),
                  make_operation(family, "silhouette", silhouette_metric="manhattan"),
                  make_operation(family, "dispersion", inner_summary="median")]
    for operation in operations:
        operation.compute(data.copy())
    assert operations[0]._estimator_ is operations[1]._estimator_
    assert operations[0]._estimator_ is operations[2]._estimator_
    assert operations[2].params["inner_summary"] == "median"
    assert operations[1].params["silhouette_metric"] == "manhattan"


@pytest.mark.parametrize("kind,suffix,title", [
    ("silhouette", "sil", "Silhouette Score"),
    ("dispersion", "dispersion", "Dispersion Summary"),
])
def test_summary_metadata(family, kind, suffix, title):
    operation = make_operation(family, kind)
    assert operation.get_identifier() == f"{family}-{suffix}"
    display_name = "KMeans" if family == "kmeans" else "KMedoids"
    assert operation.get_long_name() == f"{display_name} - {title} Distribution"


@pytest.mark.parametrize("p", [0.5, 1., 1.5, 2.])
def test_minkowski_kmedoids_inertia_centres_and_predictions(data, p):
    """Precomputed fitting must agree with a square reference and predict raw query rows."""
    from sklearn.metrics import pairwise_distances

    KMedoidsEstimator._reset_cache()
    try:
        distances = pairwise_distances(data, metric="minkowski", p=p)
        expected = KMedoids(n_clusters=2, max_iter=40, random_state=17,
                            init="random", method="alternate", metric="precomputed").fit(distances)
        operation = make_operation("kmedoids", "inertia", min_k=2, max_k=2,
                                   metric="minkowski", minkowski_p=p)
        np.testing.assert_allclose(operation.compute(data), [expected.inertia_])
        estimator = operation._estimator_
        centres = data[expected.medoid_indices_]
        np.testing.assert_array_equal(estimator._get_centres(), centres)
        queries = data[[4, 1, 7]] + 0.01
        expected_labels = pairwise_distances(queries, centres, metric="minkowski", p=p).argmin(1)
        np.testing.assert_array_equal(estimator._predict(queries), expected_labels)
    finally:
        KMedoidsEstimator._reset_cache()

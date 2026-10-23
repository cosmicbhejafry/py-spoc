"""Regression tests for KMedoids sweeps and their shared cached estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn_extra.cluster import KMedoids

from pyspoc.rstatistics.clustering.kclustering.kmedoids.statistics import KMedoidsErrors
from pyspoc.settings import settings
from pyspoc.statistics.clustering.kclustering.kmedoids._base import KMedoidsEstimator


# An unresolved annotation must fail a test, not silently skip runtime checking.
pytestmark = pytest.mark.filterwarnings("error::typeguard.TypeHintWarning")


@pytest.fixture(autouse=True)
def isolate_estimator_cache():
    """Make cache-hit assertions independent of test order and earlier fits."""
    KMedoidsEstimator._reset_cache()
    yield
    KMedoidsEstimator._reset_cache()


@pytest.fixture
def data():
    """Unequal axis lengths make missing transposition observable."""
    return np.random.default_rng(19).normal(size=(16, 5))


def _estimator_args(**overrides):
    """Return a complete, inexpensive estimator configuration."""
    return {
        "k": 2, "max_iter": 40, "random_seed": 17, "metric": "euclidean",
        "initializer": "random", "algorithm": "alternate", **overrides,
    }


def _reference_inertias(data, min_k=2, max_k=3, metric="euclidean", algorithm="alternate"):
    """Calculate expected values independently through sklearn-extra's public API."""
    return np.array([
        KMedoids(
            n_clusters=k, max_iter=40, random_state=17, metric=metric,
            init="random", method=algorithm,
        ).fit(data).inertia_
        for k in range(min_k, max_k + 1)
    ])


def _sweep(**overrides):
    """Use the same explicit options as the independent reference fits."""
    return KMedoidsErrors(**{
        "min_k": 2, "max_k": 3, "max_iter": 40, "random_seed": 17,
        "metric": "euclidean", "initializer": "random", "algorithm": "alternate",
        **overrides,
    })


@pytest.mark.parametrize("dim", ["p", "n"])
@pytest.mark.parametrize("algorithm", ["alternate", "pam"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_sweep_matches_sklearn_for_selected_dimension(data, dim, algorithm, metric):
    """Public compute must orient the data before fitting every inclusive k."""
    original = data.copy()
    result = _sweep(dim=dim, algorithm=algorithm, metric=metric).compute(data)
    expected_data = data if dim == "p" else data.T

    assert result.shape == (2,)
    np.testing.assert_allclose(
        result, _reference_inertias(expected_data, metric=metric, algorithm=algorithm))
    np.testing.assert_array_equal(data, original)


def test_single_value_sweep_returns_one_dimensional_result(data):
    """Equal bounds run once without squeezing the output to a scalar."""
    result = _sweep(min_k=2, max_k=2).compute(data)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, _reference_inertias(data, 2, 2))


@pytest.mark.parametrize("dim", ["p", "n"])
@pytest.mark.parametrize("policy", ["row-omit", "col-omit"])
def test_nan_omission_matches_explicit_cleaning(data, dim, policy):
    """Omit on the oriented matrix without modifying the caller's array."""
    data[0, 0] = np.nan
    original = data.copy()
    oriented = data if dim == "p" else data.T
    clean = (
        oriented[~np.isnan(oriented).any(axis=1)]
        if policy == "row-omit"
        else oriented[:, ~np.isnan(oriented).any(axis=0)]
    )

    result = _sweep(dim=dim, nan_policy=policy).compute(data)

    np.testing.assert_allclose(result, _reference_inertias(clean))
    np.testing.assert_array_equal(data, original)


def test_strict_nan_policy_rejects_missing_data(data):
    """The strict policy should reject NaNs before sklearn is called."""
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _sweep(nan_policy="raise").compute(data)


def test_metric_and_configuration_reach_sklearn(data):
    """Retain public options and forward only estimator-specific arguments."""
    operation = _sweep(algorithm="pam", metric="manhattan")
    operation.compute(data)
    model = operation._estimator_._get_model()

    assert model.metric == "manhattan"
    assert model.max_iter == 40
    assert model.init == "random"
    assert model.method == "pam"
    assert model.random_state == 17
    assert operation.params["metric"] == "manhattan"
    assert operation.params["algorithm"] == "pam"
    assert set(operation._estimator_kwargs) == {"metric", "initializer", "algorithm"}


def test_none_seed_is_resolved_before_cache_lookup(data):
    """Different active defaults must not reuse a model frozen to the first seed."""
    arguments = _estimator_args(random_seed=None)
    with settings.override(random_seed=41):
        first = KMedoidsEstimator.get_or_create(data, **arguments)
    with settings.override(random_seed=43):
        second = KMedoidsEstimator.get_or_create(data, **arguments)

    assert first is not second
    assert first.random_seed == 41
    assert second.random_seed == 43
    assert first._get_init_args()["random_seed"] == 41
    assert second._get_init_args()["random_seed"] == 43

    # An explicit equivalent seed must find the implicit-seed entry. A data
    # copy also checks value-based rather than identity-only cache lookup.
    same = KMedoidsEstimator.get_or_create(
        data.copy(), **_estimator_args(random_seed=41))
    assert same is first


def test_direct_estimator_keeps_frozen_seed_when_settings_change(data):
    """Freezing applies to direct construction, not only cache factories."""
    with settings.override(random_seed=41):
        estimator = KMedoidsEstimator(**_estimator_args(random_seed=None))
    with settings.override(random_seed=43):
        estimator.fit(data)
        assert estimator.random_seed == 41
        assert estimator._get_model().random_state == 41


def test_metric_participates_in_cache_identity(data):
    """A changed fitted-model option must create a separate cache entry."""
    first = KMedoidsEstimator.get_or_create(data, **_estimator_args())
    same = KMedoidsEstimator.get_or_create(data.copy(), **_estimator_args())
    different = KMedoidsEstimator.get_or_create(data, **_estimator_args(metric="manhattan"))

    assert same is first
    assert different is not first
    first.fit(data)
    different.fit(data)
    assert first._get_model().metric == "euclidean"
    assert different._get_model().metric == "manhattan"


@pytest.mark.parametrize("bounds", [{"min_k": 1}, {"min_k": 3, "max_k": 2}])
def test_invalid_sweep_bounds_are_rejected(bounds):
    """Reject invalid ranges while preserving support for equal endpoints."""
    with pytest.raises(ValueError):
        _sweep(**bounds)


def test_sweep_cannot_exceed_observation_count(data):
    """After transposition, the five original features become observations."""
    with pytest.raises(ValueError, match="number of data points"):
        _sweep(dim="n", max_k=6).compute(data)

def test_sweep_has_distinct_kmedoids_metadata():
    """A copied KMeans identifier would make the two families indistinguishable."""
    operation = _sweep()
    assert operation.get_identifier() == "kmedoids-errors"
    assert operation.get_long_name() == "KMedoids - Error Distribution"


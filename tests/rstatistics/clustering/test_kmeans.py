"""Regression tests for KMeans sweeps and their shared cached estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans

from pyspoc.rstatistics.clustering.kclustering.kmeans.statistics import KMeansErrors
from pyspoc.settings import settings
from pyspoc.statistics.clustering.kclustering.kmeans._base import KMeansEstimator


# An unresolved annotation must fail a test, not silently skip runtime checking.
pytestmark = pytest.mark.filterwarnings("error::typeguard.TypeHintWarning")


@pytest.fixture(autouse=True)
def isolate_estimator_cache():
    """Make cache-hit assertions independent of test order and earlier fits."""
    KMeansEstimator._reset_cache()
    yield
    KMeansEstimator._reset_cache()


@pytest.fixture
def data():
    """Unequal axis lengths make missing transposition observable."""
    return np.random.default_rng(19).normal(size=(16, 5))


def _estimator_args(**overrides):
    """Return a complete, inexpensive estimator configuration."""
    return {
        "k": 2, "max_iter": 40, "random_seed": 17, "tol": 0.02,
        "initializer": "random", "algorithm": "lloyd", **overrides,
    }


def _reference_inertias(data, min_k=2, max_k=3):
    """Calculate expected values independently through sklearn's public API."""
    return np.array([
        KMeans(
            n_clusters=k, max_iter=40, random_state=17, tol=0.02,
            init="random", algorithm="lloyd", n_init="auto",
        ).fit(data).inertia_
        for k in range(min_k, max_k + 1)
    ])


def _sweep(**overrides):
    """Use the same explicit options as the independent reference fits."""
    return KMeansErrors(**{
        "min_k": 2, "max_k": 3, "max_iter": 40, "random_seed": 17,
        "tol": 0.02, "initializer": "random", "algorithm": "lloyd",
        **overrides,
    })


@pytest.mark.parametrize("dim", ["p", "n"])
def test_sweep_matches_sklearn_for_selected_dimension(data, dim):
    """Public compute must orient the data before fitting every inclusive k."""
    original = data.copy()
    result = _sweep(dim=dim).compute(data)
    expected_data = data if dim == "p" else data.T

    assert result.shape == (2,)
    np.testing.assert_allclose(result, _reference_inertias(expected_data))
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


def test_tolerance_and_configuration_reach_sklearn(data):
    """Retain public options and forward only estimator-specific arguments."""
    operation = _sweep(algorithm="elkan")
    operation.compute(data)
    model = operation._estimator_._get_model()

    assert model.tol == 0.02
    assert model.max_iter == 40
    assert model.init == "random"
    assert model.algorithm == "elkan"
    assert model.random_state == 17
    assert operation.params["tol"] == 0.02
    assert operation.params["algorithm"] == "elkan"
    assert set(operation._estimator_kwargs) == {"tol", "initializer", "algorithm"}


def test_none_seed_is_resolved_before_cache_lookup(data):
    """Different active defaults must not reuse a model frozen to the first seed."""
    arguments = _estimator_args(random_seed=None)
    with settings.override(random_seed=41):
        first = KMeansEstimator.get_or_create(data, **arguments)
    with settings.override(random_seed=43):
        second = KMeansEstimator.get_or_create(data, **arguments)

    assert first is not second
    assert first.random_seed == 41
    assert second.random_seed == 43
    assert first._get_init_args()["random_seed"] == 41
    assert second._get_init_args()["random_seed"] == 43

    # An explicit equivalent seed must find the implicit-seed entry. A data
    # copy also checks value-based rather than identity-only cache lookup.
    same = KMeansEstimator.get_or_create(
        data.copy(), **_estimator_args(random_seed=41))
    assert same is first


def test_direct_estimator_keeps_frozen_seed_when_settings_change(data):
    """Freezing applies to direct construction, not only cache factories."""
    with settings.override(random_seed=41):
        estimator = KMeansEstimator(**_estimator_args(random_seed=None))
    with settings.override(random_seed=43):
        estimator.fit(data)
        assert estimator.random_seed == 41
        assert estimator._get_model().random_state == 41


def test_tolerance_participates_in_cache_identity(data):
    """A changed fitted-model option must create a separate cache entry."""
    first = KMeansEstimator.get_or_create(data, **_estimator_args())
    same = KMeansEstimator.get_or_create(data.copy(), **_estimator_args())
    different = KMeansEstimator.get_or_create(data, **_estimator_args(tol=0.03))

    assert same is first
    assert different is not first
    first.fit(data)
    different.fit(data)
    assert first._get_model().tol == 0.02
    assert different._get_model().tol == 0.03


@pytest.mark.parametrize("bounds", [{"min_k": 1}, {"min_k": 3, "max_k": 2}])
def test_invalid_sweep_bounds_are_rejected(bounds):
    """Reject invalid ranges while preserving support for equal endpoints."""
    with pytest.raises(ValueError):
        _sweep(**bounds)


def test_sweep_cannot_exceed_observation_count(data):
    """After transposition, the five original features become observations."""
    with pytest.raises(ValueError, match="number of data points"):
        _sweep(dim="n", max_k=6).compute(data)

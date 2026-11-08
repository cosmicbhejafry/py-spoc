"""Bandwidth-aware integration and zero-density conventions for KDE KL."""

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from pyspoc.settings import settings
from pyspoc.statistics.distribution.divergence.kl import KullbackLeiblerDivergence


pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


@pytest.fixture(autouse=True)
def execution_settings():
    with settings.override(max_worker_threads=1, pairwise_error_policy="first"):
        yield


@pytest.mark.parametrize("bandwidth", [2.0, "scott", "silverman"])
def test_gaussian_grid_covers_tails_and_matches_analytic_kl(bandwidth):
    # Repeated observations at 0 and 1 produce exact N(0,h^2) and N(1,h^2)
    # KDEs. Their KL is known independently: 1/(2*h^2). The former 10%-range
    # padding severely underestimated it, especially for bandwidth=2.
    data = np.column_stack((np.zeros(16), np.ones(16)))
    kde = KernelDensity(bandwidth=bandwidth).fit(data[:, :1])
    h = kde.bandwidth_
    statistic = KullbackLeiblerDivergence(kde_bw=bandwidth, kl_resolution=4097)
    result = statistic.compute(data)

    np.testing.assert_allclose(result, [[0, 1 / (2 * h**2)], [1 / (2 * h**2), 0]],
                               rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(statistic._support, (-8 * h, 1 + 8 * h))


@pytest.mark.parametrize("kernel", ["tophat", "epanechnikov", "linear", "cosine"])
def test_disjoint_compact_supports_give_infinite_cross_kl_and_zero_diagonal(kernel):
    data = np.column_stack((np.zeros(12), np.full(12, 4.0)))
    statistic = KullbackLeiblerDivergence(
        kde_kernel=kernel, kde_bw=0.5, kl_resolution=1001)

    # Both distributions vanish over much of the grid. Those locations must
    # contribute zero, not NaN; positive source mass outside q gives infinity.
    with np.errstate(divide="raise", invalid="raise"):
        result = statistic.compute(data)
    np.testing.assert_array_equal(np.diag(result), [0.0, 0.0])
    assert np.isposinf(result[0, 1])
    assert np.isposinf(result[1, 0])
    np.testing.assert_allclose(statistic._support, (-0.5, 4.5))


@pytest.mark.parametrize("kernel", ["tophat", "epanechnikov", "linear", "cosine"])
def test_identical_compact_kdes_do_not_generate_nan_outside_support(kernel):
    column = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    data = np.column_stack((column, column))
    statistic = KullbackLeiblerDivergence(
        kde_kernel=kernel, kde_bw=0.2, kl_resolution=1001)
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(statistic.compute(data), np.zeros((2, 2)))


def test_source_support_contained_in_target_is_finite_only_in_forward_direction():
    # The narrow source's support is contained in the target mixture's support.
    # This checks directionality of the zero-density test, not just disjointness.
    source = np.zeros(12)
    target = np.tile([-2.0, 0.0, 2.0], 4)
    statistic = KullbackLeiblerDivergence(
        kde_kernel="tophat", kde_bw=0.4, kl_resolution=4001)
    with np.errstate(divide="raise", invalid="raise"):
        result = statistic.compute(np.column_stack((source, target)))
    np.testing.assert_allclose(result[0, 1], np.log(3), atol=0.01)
    assert np.isposinf(result[1, 0])


def test_exponential_padding_uses_resolved_bandwidth():
    statistic = KullbackLeiblerDivergence(
        kde_kernel="exponential", kde_bw="scott", kl_resolution=2001)
    data = np.ones((16, 2))
    h = KernelDensity(kernel="exponential", bandwidth="scott").fit(data[:, :1]).bandwidth_
    result = statistic.compute(data)
    np.testing.assert_array_equal(result, np.zeros((2, 2)))
    np.testing.assert_allclose(statistic._support, (1 - 32 * h, 1 + 32 * h))


def test_each_column_is_fitted_once_and_cache_is_reset_between_datasets(monkeypatch):
    fit_calls = []
    original_fit = KernelDensity.fit

    def counted_fit(self, data, *args, **kwargs):
        fit_calls.append(data.copy())
        return original_fit(self, data, *args, **kwargs)

    monkeypatch.setattr(KernelDensity, "fit", counted_fit)
    rng = np.random.default_rng(42)
    first = rng.normal(size=(12, 3))
    second = rng.normal(loc=[0, 2, 4], size=(12, 3))
    statistic = KullbackLeiblerDivergence(kl_resolution=1001)
    statistic.compute(first)
    assert len(fit_calls) == 3
    reused = statistic.compute(second)
    assert len(fit_calls) == 6
    fresh = KullbackLeiblerDivergence(kl_resolution=1001).compute(second)
    np.testing.assert_allclose(reused, fresh)


def test_nonfinite_density_failure_is_not_silently_integrated_as_zero():
    statistic = KullbackLeiblerDivergence()
    statistic._support_grid = np.array([0.0, 1.0])
    statistic._log_densities = {0: np.array([np.nan, 0.0]), 1: np.array([0.0, 0.0])}
    with pytest.raises(ValueError, match="NaN log densities"):
        statistic._pairwise_compute_indexed(np.zeros(2), np.zeros(2), (0, 1))

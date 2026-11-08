"""Numerical and parallel-execution regressions for empirical isotropy.

These tests exercise public ``compute()`` calls, including shared preparation,
seed generation, worker dispatch, and aggregation. Small seeded point clouds
and eleven transformations keep the suite fast while exercising partial chunks.
"""

import numpy as np
import pytest

from pyspoc.rstatistics.symmetry.reflection import HouseholderEmpiricalIsotropy
from pyspoc.rstatistics.symmetry.rotation import (
    GivensEmpiricalIsotropy,
    HarrEmpiricalIsotropy,
)
from pyspoc.settings import settings


pytestmark = [
    pytest.mark.filterwarnings("error::RuntimeWarning"),
    pytest.mark.filterwarnings("error::typeguard.TypeHintWarning"),
]


@pytest.fixture(params=[
    pytest.param((GivensEmpiricalIsotropy, "n_rotations"), id="givens"),
    pytest.param((HarrEmpiricalIsotropy, "n_rotations"), id="haar"),
    pytest.param((HouseholderEmpiricalIsotropy, "n_reflections"), id="householder"),
])
def make_statistic(request):
    """Use the family-specific count argument while sharing all expectations."""
    statistic_type, count_argument = request.param

    def make(*, seed=123, dim="p"):
        return statistic_type(**{
            count_argument: 11,
            "random_seed": seed,
            "dim": dim,
        })

    return make


@pytest.fixture
def point_cloud():
    return np.random.default_rng(42).standard_normal((30, 4))


@pytest.fixture(autouse=True)
def bounded_workers():
    # Never let these small tests create a pool sized to the entire host/HPC node.
    with settings.override(max_worker_threads=4):
        yield


@pytest.mark.parametrize("scale", [1e-300, 1e-90, 1e90, 1e300, -7.0])
def test_uniform_scaling_preserves_score(make_statistic, point_cloud, scale):
    # The same seed assigns the same transformations to both datasets. Both
    # squared-variance terms must scale together, leaving the ratio unchanged.
    # Extreme scales used to underflow/overflow before shared data preparation.
    expected = make_statistic().compute(point_cloud)
    actual = make_statistic().compute(point_cloud * scale)

    assert np.isfinite(expected)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", ["n", "p"])
def test_serial_and_parallel_scores_match(make_statistic, point_cloud, dim):
    data = point_cloud if dim == "p" else point_cloud.T
    statistic = make_statistic(dim=dim)

    # Reuse one instance sequentially to check that each computation recreates
    # its random streams instead of advancing persistent RNG/spawning state.
    # Different chunk boundaries must not change which seed a task receives.
    with settings.override(max_worker_threads=1):
        serial = statistic.compute(data)
    with settings.override(max_worker_threads=4):
        parallel = statistic.compute(data)
        repeated = statistic.compute(data)

    assert np.isfinite(serial)
    np.testing.assert_allclose([parallel, repeated], serial, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("constant_values", [
    pytest.param([0.0, 0.0, 0.0, 0.0], id="all-zero"),
    pytest.param([1.1, 2.2, 3.3, 4.4], id="different-column-constants"),
])
def test_constant_data_returns_nan_without_dispatch(
    make_statistic, monkeypatch, constant_values,
):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Constant data should return before dispatching worker tasks")

    monkeypatch.setattr(
        "pyspoc.rstatistics.symmetry._empirical.execute_async", unexpected_dispatch)
    data = np.tile(constant_values, (30, 1))

    # Decimal constants can have tiny rounding residuals in np.var(). Exact
    # constancy must still be detected before scaling or forming the denominator.
    assert np.isnan(make_statistic().compute(data))


def test_one_constant_column_does_not_reject_remaining_variation(make_statistic, point_cloud):
    point_cloud[:, 0] = 2.0
    statistic = make_statistic()

    assert np.isfinite(statistic.compute(point_cloud))
    assert statistic._data_var[0] == 0
    assert statistic._tot_sq_var > 0


def test_small_nonconstant_data_is_not_treated_as_constant(make_statistic, point_cloud):
    # Deliberately below np.isclose's default absolute tolerance. Small amplitude
    # alone must not be confused with coincident observations.
    data = point_cloud * 1e-12
    assert np.isclose(data, 0).all()
    assert not np.all(data == data[0])
    assert np.isfinite(make_statistic().compute(data))


@pytest.mark.parametrize("layout", ["writeable", "readonly", "strided"])
def test_compute_preserves_input_and_shares_readonly_variances(
    make_statistic, point_cloud, layout,
):
    if layout == "strided":
        backing = np.zeros((len(point_cloud), 2 * point_cloud.shape[1]))
        backing[:, ::2] = point_cloud
        data = backing[:, ::2]
        assert not data.flags.c_contiguous
    else:
        data = point_cloud.copy()
    if layout == "readonly":
        data.flags.writeable = False

    snapshot = data.copy()
    was_writeable = data.flags.writeable
    statistic = make_statistic()
    result = statistic.compute(data)

    assert np.isfinite(result)
    np.testing.assert_array_equal(data, snapshot)
    assert data.flags.writeable == was_writeable
    assert not statistic._data_var.flags.writeable
    assert not np.shares_memory(statistic._data_var, data)


def test_equal_variance_uncorrelated_cloud_has_unit_score(make_statistic):
    # The +/- coordinate axes have covariance proportional to the identity.
    # Any orthogonal transformation preserves their per-axis variance, giving
    # an independently known score of one (not merely agreement between runs).
    data = np.vstack((np.eye(4), -np.eye(4)))
    np.testing.assert_allclose(make_statistic().compute(data), 1.0, rtol=0, atol=1e-12)


def test_random_generation_does_not_consume_numpy_global_state(make_statistic, point_cloud):
    # Each task must use its own seeded Generator, not np.random's shared legacy
    # RNG. Inspect its state without reseeding or changing unrelated tests.
    before = np.random.get_state()
    make_statistic().compute(point_cloud)
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

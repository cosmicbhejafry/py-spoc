from itertools import pairwise

import numpy as np
import pytest

from pyspoc.rstatistics.fractal import base as fractal_base_module
from pyspoc.rstatistics.fractal.renyi import (
    RenyiEntropy,
    _get_renyi_entropy,
    _get_renyi_entropy_numba,
)
from pyspoc.settings import settings

Q_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
REPRESENTATIVE_Q_VALUES = (0.0, 1.0, 2.0)
SLOPE_ESTIMATION_METHODS = ("hybrid", "ols", "deshmukh")
DATA_CASES = (
    (100, 1, 0),
    (250, 3, 10),
    (500, 10, 100),
)
SCALES = np.exp(np.linspace(-6, 6, 11))


def _assert_valid_entropy(entropy: np.ndarray) -> None:
    """Assert the common shape and value invariants for an entropy curve."""
    assert entropy.shape == SCALES.shape
    assert np.all(np.isfinite(entropy))
    assert np.all(entropy >= 0)


def _assert_non_increasing(values: list[np.ndarray]) -> None:
    """Assert successive curves do not increase beyond rounding tolerance."""
    for lower_order, higher_order in pairwise(values):
        is_ordered = higher_order <= lower_order
        is_effectively_equal = np.isclose(
            higher_order,
            lower_order,
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.all(is_ordered | is_effectively_equal)


@pytest.mark.parametrize("slope_estimation_method", SLOPE_ESTIMATION_METHODS)
def test_constructor_exposes_fractal_configuration(slope_estimation_method):
    statistic = RenyiEntropy(
        slope_estimation_method=slope_estimation_method,
        q=1.5,
        r2_thresh=0.8,
        monotonic_tol=0.2,
        deshmukh_reg_proportion=0.3,
        minimum_scaling_region=0.3,
        minimum_scaling_points=24,
        use_adaptive_scaling=False,
        scale_method="log-10",
        scale_length=60,
        scale_adaption_iters=5,
    )

    assert statistic.slope_estimation_method == slope_estimation_method
    assert statistic.q == 1.5
    assert statistic.r2_thresh == 0.8
    assert statistic.monotonic_tol == 0.2
    assert statistic.deshmukh_reg_proportion == 0.3
    assert statistic.minimum_scaling_region == 0.3
    assert statistic.minimum_scaling_points == 24
    assert statistic.use_adaptive_scaling is False
    assert statistic.scale_method == "log-10"
    assert statistic.scale_length == 60
    assert statistic.scale_adaption_iters == 5


@pytest.mark.parametrize(
    ("kwargs", "expected_exception"),
    (
        ({"q": -0.01}, ValueError),
        ({"r2_thresh": -0.01}, ValueError),
        ({"r2_thresh": 1.01}, ValueError),
        ({"monotonic_tol": -0.01}, ValueError),
        ({"monotonic_tol": 1.01}, ValueError),
        ({"deshmukh_reg_proportion": 0.0}, ValueError),
        ({"deshmukh_reg_proportion": 0.51}, ValueError),
        ({"minimum_scaling_region": 0.0}, ValueError),
        ({"minimum_scaling_region": 1.01}, ValueError),
        ({"minimum_scaling_points": 19}, ValueError),
        ({"minimum_scaling_points": 20.5}, TypeError),
        ({"scale_length": 49}, ValueError),
        ({"scale_length": 10.5}, TypeError),
        ({"scale_adaption_iters": 0}, ValueError),
        ({"scale_adaption_iters": 1.5}, TypeError),
    ),
)
def test_constructor_rejects_invalid_numeric_arguments(kwargs, expected_exception):
    with pytest.raises(expected_exception):
        RenyiEntropy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"slope_estimation_method": "unknown"},
        {"scale_method": "linear"},
        {"use_adaptive_scaling": 1},
    ),
)
def test_constructor_rejects_invalid_typed_arguments(kwargs):
    with pytest.raises(TypeError):
        RenyiEntropy(**kwargs)


def test_constructor_accepts_numeric_boundaries():
    statistic = RenyiEntropy(
        q=0,
        r2_thresh=0,
        monotonic_tol=1,
        deshmukh_reg_proportion=0.5,
        minimum_scaling_region=1,
        minimum_scaling_points=20,
        scale_length=50,
        scale_adaption_iters=1,
    )

    assert statistic.q == 0.0
    assert statistic.r2_thresh == 0.0
    assert statistic.monotonic_tol == 1.0
    assert statistic.deshmukh_reg_proportion == 0.5
    assert statistic.minimum_scaling_region == 1.0
    assert statistic.minimum_scaling_points == 20
    assert statistic.scale_length == 50
    assert statistic.scale_adaption_iters == 1


@pytest.mark.parametrize(
    ("elbow_indices", "expected_points"),
    (
        ((25, 75), 50),  # Retains enough points and scale span.
        ((48, 55), 100),  # Rejects fewer than 20 retained points.
        ((38, 47), 100),  # Rejects less than 10% of the scale span.
    ),
)
def test_elbow_region_must_meet_size_requirements(
    monkeypatch,
    elbow_indices,
    expected_points,
):
    scales = np.exp(np.linspace(-4, 4, 100))
    ols_point_counts = []

    monkeypatch.setattr(
        fractal_base_module.fpy,
        "get_datseries_scales",
        lambda *args, **kwargs: scales,
    )
    monkeypatch.setattr(
        fractal_base_module.fpy,
        "find_elbow_idx",
        lambda *args, **kwargs: elbow_indices,
    )

    def fake_ols_results(x, y):
        ols_point_counts.append(x.shape[0])
        return float(x.shape[0]), 1.0

    monkeypatch.setattr(
        fractal_base_module.fpy,
        "compute_ols_results",
        fake_ols_results,
    )
    monkeypatch.setattr(
        RenyiEntropy,
        "_compute_density_estimate",
        lambda self, data, selected_scales: np.arange(selected_scales.shape[0]),
    )

    statistic = RenyiEntropy(
        slope_estimation_method="ols",
        use_adaptive_scaling=False,
        scale_length=50,
    )
    result = statistic.compute(np.zeros((10, 2)))

    assert result == float(expected_points)
    assert ols_point_counts[-1] == expected_points


@pytest.mark.parametrize(
    ("dimensions", "expected_scale_length", "expected_minimum_points"),
    (
        (1, 50, 20),
        (2, 70, 20),
        (25, 322, 33),
        (100, 461, 47),
        (200, 530, 53),
    ),
)
def test_automatic_scaling_requirements_follow_dimensionality(
    dimensions,
    expected_scale_length,
    expected_minimum_points,
):
    statistic = RenyiEntropy()
    data = np.empty((10, dimensions))

    scale_length = statistic._resolve_scale_length(data)
    minimum_points = statistic._resolve_minimum_scaling_points(scale_length)

    assert statistic.scale_length is None
    assert statistic.minimum_scaling_points == 20
    assert scale_length == expected_scale_length
    assert minimum_points == expected_minimum_points


@pytest.mark.parametrize("q", Q_VALUES)
@pytest.mark.parametrize(("n", "dim", "seed"), DATA_CASES)
def test_entropy_basic_cases(data_factory, q, n, dim, seed):
    data = data_factory(n, dim, seed)

    entropy = _get_renyi_entropy(q, data, SCALES)

    _assert_valid_entropy(entropy)


@pytest.mark.parametrize("q", REPRESENTATIVE_Q_VALUES)
@pytest.mark.parametrize(("n", "dim", "seed"), DATA_CASES)
def test_python_backend_basic_cases(data_factory, q, n, dim, seed):
    data = data_factory(n, dim, seed)

    with settings.override(numba_mode="python"):
        entropy = _get_renyi_entropy(q, data, SCALES)

    _assert_valid_entropy(entropy)


@pytest.mark.parametrize("q", Q_VALUES)
def test_python_and_numba_backends_agree(data_factory, q):
    data = data_factory(300, 4, 17)

    with settings.override(numba_mode="python"):
        python_entropy = _get_renyi_entropy(q, data, SCALES)

    numba_entropy = _get_renyi_entropy_numba(q, data, SCALES)

    np.testing.assert_allclose(numba_entropy, python_entropy)


@pytest.mark.parametrize("q", REPRESENTATIVE_Q_VALUES)
def test_fine_scale_saturation_avoids_box_id_overflow(data_factory, q):
    data = data_factory(100, 2, 23)
    descending_scales = np.exp(np.linspace(100, -100, 21))
    expected_saturated_entropy = np.log(data.shape[0])

    with settings.override(numba_mode="python"):
        python_entropy = _get_renyi_entropy(q, data, descending_scales)

    numba_entropy = _get_renyi_entropy_numba(q, data, descending_scales)

    np.testing.assert_allclose(numba_entropy, python_entropy)
    assert python_entropy[-1] == pytest.approx(expected_saturated_entropy)


@pytest.mark.parametrize("q", REPRESENTATIVE_Q_VALUES)
def test_coarse_scale_saturation_has_zero_entropy(data_factory, q):
    data = data_factory(100, 2, 29)
    ascending_scales = np.exp(np.linspace(-4, 100, 21))

    with settings.override(numba_mode="python"):
        python_entropy = _get_renyi_entropy(q, data, ascending_scales)

    numba_entropy = _get_renyi_entropy_numba(q, data, ascending_scales)

    np.testing.assert_allclose(numba_entropy, python_entropy)
    assert python_entropy[-1] == 0.0


@pytest.mark.parametrize(("n", "dim", "seed"), DATA_CASES)
def test_entropy_is_non_increasing_in_q(data_factory, n, dim, seed):
    data = data_factory(n, dim, seed)
    entropy_by_q = [
        _get_renyi_entropy(q, data, SCALES)
        for q in Q_VALUES
    ]

    _assert_non_increasing(entropy_by_q)


@pytest.mark.parametrize("q", REPRESENTATIVE_Q_VALUES)
def test_fractal_entropy_curves_are_valid(fractal_factory, q):
    for _, data, _ in fractal_factory():
        entropy = _get_renyi_entropy(q, data, SCALES)

        _assert_valid_entropy(entropy)


def test_fractal_entropy_is_non_increasing_in_q(fractal_factory):
    for _, data, _ in fractal_factory():
        entropy_by_q = [
            _get_renyi_entropy(q, data, SCALES)
            for q in Q_VALUES
        ]

        _assert_non_increasing(entropy_by_q)


@pytest.mark.parametrize("q", REPRESENTATIVE_Q_VALUES)
def test_renyi_entropy_computes_positive_finite_dimension(fractal_factory, q):
    for _, data, _ in fractal_factory():
        calculated_dimension = RenyiEntropy(q=q).compute(data)

        assert np.isfinite(calculated_dimension)
        assert calculated_dimension > 0


def test_box_counting_dimension_matches_fractal_reference(fractal_factory):
    failures = []

    for fractal_name, data, expected_dimension in fractal_factory():
        calculated_dimension = RenyiEntropy(q=0).compute(data)
        lower_bound = 0.75 * expected_dimension
        upper_bound = 1.25 * expected_dimension

        if not lower_bound < calculated_dimension < upper_bound:
            failures.append(
                f"{fractal_name!r}: expected {expected_dimension} "
                f"(allowed {lower_bound}–{upper_bound}), "
                f"got {calculated_dimension}"
            )

    assert not failures, "Unexpected q=0 dimensions:\n" + "\n".join(failures)

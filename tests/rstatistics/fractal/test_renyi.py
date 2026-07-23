import numpy as np
import pytest

from itertools import product

from pyspoc.rstatistics.fractal.renyi import (
    _get_renyi_entropy,
    _get_renyi_entropy_py,
    _get_renyi_entropy_numba_bc,
    _get_renyi_entropy_numba_alt_bc,
    RenyiEntropy
)

# --------------------------------------------------
# Functionality
# --------------------------------------------------

q_opt = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ns = [100,1000,10000]
dims = [1,2,10,100]
seeds = [0,10,100,1000]
neg_log_scales_opt = [
    np.linspace(-10, 10, 11),
    np.linspace(-12, 15, 11),
    np.linspace(-100, 100, 11)
]
scales_opt = [np.exp(x) for x in neg_log_scales_opt]
params = list(product(q_opt, scales_opt))

@pytest.mark.parametrize(
    ("q", "scales"),
    params
)
def test_basic_cases(data_factory, q, scales):
    data_params = product(ns, dims, seeds)

    for n, dim, seed in data_params:
        data = data_factory(n, dim, seed)
        H = _get_renyi_entropy(q, data, scales, 1e-6)

        assert H.shape == scales.shape, \
            f"Data params: {n=}, {dim=}, {seed=}.\n" \
            "Entropy score for each scale."
        assert np.all(np.isfinite(H)), \
            f"Data params: {n=}, {dim=}, {seed=}.\n" \
            "All entropy values should be finite."
        assert np.all(H >= 0), \
            f"Data params: {n=}, {dim=}, {seed=}.\n" \
            "All entropy values should be non-negative."
    
@pytest.mark.parametrize(
    ("q", "scales"),
    params
)
def test_basic_cases_py(data_factory, q, scales):
    data_params = product(ns, dims, seeds)

    for n, dim, seed in data_params:
        data = data_factory(n, dim, seed)
        H = _get_renyi_entropy_py(q, data, scales, 1e-6)

        assert H.shape == scales.shape, \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "Entropy score for each scale."
        assert np.all(np.isfinite(H)), \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "All entropy values should be finite."
        assert np.all(H >= 0), \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "All entropy values should be non-negative."

@pytest.mark.parametrize(
    ("q", "scales"),
    params
)
def test_basic_cases_numba(data_factory, q, scales):
    data_params = product(ns, dims, seeds)

    for n, dim, seed in data_params:
        data = data_factory(n, dim, seed)
        H = _get_renyi_entropy_numba_alt_bc(q, data, scales, 1e-6)

        assert H.shape == scales.shape, \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "Entropy score for each scale."
        assert np.all(np.isfinite(H)), \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "All entropy values should be finite."
        assert np.all(H >= 0), \
            f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
            "All entropy values should be non-negative."
    
# --------------------------------------------------
# Sense checks
# --------------------------------------------------
@pytest.mark.parametrize(
    ("scales"),
    scales_opt
)
def test_relative_values(data_factory, scales):
    data_params = product(ns, dims, seeds)

    for n, dim, seed in data_params:
        data = data_factory(n, dim, seed)
        prev_H = 0
        sorted_qs = sorted(q_opt)
        prev_q = sorted_qs[0]

        for q in sorted_qs:
            H = _get_renyi_entropy(q, data, scales, 1e-6, debug_numba="raise")
        
            assert np.all(H >= prev_H), \
                f"Data params: n: {n}, dim: {dim}, seed: {seed}.\n" \
                f"{q}-entropy should be higher than {prev_q}-entropy."

# --------------------------------------------------
# Fractal estimation
# --------------------------------------------------
@pytest.mark.parametrize(
        ("q", "scales"),
        list(product(q_opt, scales_opt))
)
def test_fractals_basic(fractal_factory, q, scales):
    test_set = fractal_factory()

    for fractal_name, data, _ in test_set:
        H = _get_renyi_entropy(q, data, scales, 1e-6, debug_numba="raise")

        assert H.shape == scales.shape, \
            f"Fractal {fractal_name}: Entropy score should exist for each scale."
        assert np.all(np.isfinite(H)), \
            f"Fractal {fractal_name}: All entropy values should be finite."
        assert np.all(H >= 0), \
            f"Fractal {fractal_name}: All entropy values should be non-negative."


@pytest.mark.parametrize(
        ("scales"),
        scales_opt
)
def test_fractal_entropy_values(fractal_factory, scales):
    test_set = fractal_factory()

    for fractal_name, data, _ in test_set:
        prev_H = 0
        sorted_qs = sorted(q_opt)
        prev_q = sorted_qs[0]

        for q in sorted_qs:
            H = _get_renyi_entropy(q, data, scales, 1e-6, debug_numba="raise")
        
            assert np.all(H >= prev_H), \
                f"Fractal: {fractal_name}\n" \
                f"Failure: {q}-entropy should be higher than {prev_q}-entropy."

@pytest.mark.parametrize(
        ("q"),
        q_opt
)
def test_fractals_class_output(fractal_factory, q):
    test_set = fractal_factory()

    for fractal_name, data, actual_fd in test_set:
        renyi = RenyiEntropy(q=q, debug_numba="raise")
        calculated_fd = renyi.compute(data)
        assert calculated_fd > 0.75 * actual_fd and calculated_fd < 1.25 * actual_fd, \
            f"Fractal: {fractal_name}\n" \
            f"Failure: Computed fractal dimension should be approximately {actual_fd}, " \
            f"got {calculated_fd}."


# --------------------------------------------------
# Full pipeline tests
# --------------------------------------------------






# --------------------------------------------------
# Comparison tests
# --------------------------------------------------

@pytest.mark.parametrize(
    ("q", "scales"),
    params
)
def test_basic_comparisons_numba(data_factory, q, scales):
    data_params = product(ns, dims, seeds)
    eps = 1e-4

    for n, dim, seed in data_params:
        data = data_factory(n, dim, seed)
        H = _get_renyi_entropy_numba_bc(q, data, scales, 1e-6)
        H_alt = _get_renyi_entropy_numba_alt_bc(q, data, scales, 1e-6)

        assert np.all(H - eps <= H_alt) and np.all(H_alt <= H + eps), \
            f"Data params: {q=}, {scales=}" \
            "Entropy score should be equal at each scale." \
            "Got:\n" \
            f"{H=}\n" \
            f"{H_alt=}"
            
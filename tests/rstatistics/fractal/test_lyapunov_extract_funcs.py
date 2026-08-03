import numpy as np
import pytest

from itertools import product

from pyspoc.rstatistics.fractal._funcs_py import (
    get_log10_scales,
    get_datseries_scales,
    get_adaptive_scales
)

from pyspoc.rstatistics.fractal._funcs_numba import (
    compute_deshmukh_slope_estimate
)

# --------------------------------------------------
# Best parsimonious model fit
# --------------------------------------------------
q_opt = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

neg_log_scales_opt = [
    np.linspace(-10, 10, 21),
    np.linspace(-12, 15, 21),
    np.linspace(-100, 100, 21)
]

scales_opt = [np.exp(x) for x in neg_log_scales_opt]

H_opt = [
    np.array([0.0, 0.0, 0.0, 0.2, 0.3, 0.2, 0.5, 0.4, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0, 0.9, 1.2, 0.9, 1.1, 1.2, 1.1]),
    np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.4, 1.7, 1.9, 2.0, 1.7, 1.8, 1.7, 2.1, 2.0, 1.9, 2.2, 2.2, 2.3, 2.2]),
    np.array([1.0, 1.0, 1.0, 0.2, 1.3, 0.5, 1.5, 1.4, 1.7, 2.0, 2.2, 2.0, 2.1, 2.2, 2.0, 3.0, 2.3, 2.4, 3.0, 3.1, 3.2]),
    np.array([1.0, 1.2, 1.4, 1.6, 2.0, 2.2, 2.4, 2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.8, 2.7, 2.6, 2.7, 2.9, 3.9, 2.3, 2.2])
]

adj_r2_tol_opt = [0.01, 0.03, 0.05, 0.1, 0.2]

data_kwargs_opt = [
    {"n": 100, "dim": 2, "seed": 0},
    {"n": 100, "dim": 2, "seed": 10},
    {"n": 1000, "dim": 10, "seed": 0},
    {"n": 100, "dim": 100, "seed": 0}
]

n_breakpoints_opt = [1, 2, 3]
shape_opt = [(50,2),(40,3),(1000,1),(1000,10)]

params = list(product(neg_log_scales_opt, H_opt, adj_r2_tol_opt))

# --------------------------------------------------
# Elbow trimming
# --------------------------------------------------

alphas_opt = [np.array([0.0, 0.7, 0.8, 0.1]),
              np.array([1.0, 0.7, 0.8, 1.1]),
              np.array([0.1, 0.7, 0.8, 1.1]),
              np.array([-0.1, 0.7, 0.8, 1.9])]

monotonic_tol_opt = [0.02, 0.05, 0.1, 0.15, 0.2]
seed_opt = [0, 101, 10862, 62]

# --------------------------------------------------
# Slope ensemble
# --------------------------------------------------

trimmed_scales_opt = [x[3:9] for x in neg_log_scales_opt]
trimmed_H_opt = [x[3:9] for x in H_opt]
prop_window_size_opt = [0.1, 0.2, 0.3, 0.5]

@pytest.mark.parametrize(
    ("trimmed_scales", "trimmed_H", "prop_window_size"),
    list(product(
        trimmed_scales_opt,
        trimmed_H_opt,
        prop_window_size_opt))
)
def test_slope_ensemble_shapes_and_values(trimmed_scales,
                                          trimmed_H,
                                          prop_window_size):
    
    fd = compute_deshmukh_slope_estimate(
        trimmed_scales,
        trimmed_H,
        prop_window_size)
    
    assert fd is not None, "Fractal dimension should not be None."
    assert isinstance(fd, (float, np.floating)), "Fractal dimension should be a float."
    assert fd > 0, "Fractal dimension should be greater than zero."
    assert np.isfinite(fd), "Fractal dimension should be finite."

# --------------------------------------------------
# Modal averaging
# --------------------------------------------------

def test_modal_average_no_mode():
    trimmed_scales = np.array([0, 1, 2, 3, 4, 5])
    trimmed_H = trimmed_scales # Single slope of m = 1
    prop_window_size = 0.2

    fd = compute_deshmukh_slope_estimate(
        trimmed_scales,
        trimmed_H,
        prop_window_size)

    assert fd is not None, "Fractal dimension should not be None."
    assert isinstance(fd, (float, np.floating)), "Fractal dimension should be a float."
    assert round(fd,2) == 1, f"Fractal dimension of {fd} should be expected value 1."


def test_modal_average_multimodal(rng):
    trimmed_scales = np.array([0, 1, 2, 3, 4, 5])
    
    # Two slopes of equal length: m_1 = 0.3, m_2 = 0.6
    trimmed_H = np.array([0, 0.3, 0.6, 1.2, 1.8, 2.4])
    prop_window_size = 0.2

    fd = compute_deshmukh_slope_estimate(
        trimmed_scales,
        trimmed_H,
        prop_window_size)

    assert fd is not None, "Fractal dimension should not be None."
    assert isinstance(fd, (float, np.floating)), "Fractal dimension should be a float."
    assert fd >= 0.3 and fd <= 0.6, f"Fractal dimension {fd} should be within support."


def test_modal_average_single_mode():
    trimmed_scales = np.array([0, 1, 2, 3, 4, 5])
    
    # Four slopes of unequal length: m_1 = 0.3, m_2 = 0.6, m_3 = 0.8, m_4 = 2
    trimmed_H = np.array([0, 0.3, 0.6, 1.2, 2.0, 4.0])
    prop_window_size = 0.2

    fd = compute_deshmukh_slope_estimate(
        trimmed_scales,
        trimmed_H,
        prop_window_size)
    
    assert fd is not None, "Fractal dimension should not be None."
    assert isinstance(fd, (float, np.floating)), "Fractal dimension should be a float."
    assert fd >= 0.3 and fd <= 2, f"Fractal dimension {fd} should be within support."

# --------------------------------------------------
# Default scale generation
# --------------------------------------------------

test_cases = [
        ((50, 2), {}, 51),
        ((20, 2), {"k": 100}, 101),
        ((40, 2), {}, 51),
    ]

@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    test_cases,
)
def test_base10_scale_basic_properties(rng, shape, kwargs, expected_len):
    xs = rng.random(shape)
    scales = get_log10_scales(xs, **kwargs)

    assert scales.shape == (expected_len,), "Scale shape should match num_scales."
    assert np.all(scales > 0), "All scales should be greater than zero."
    assert np.all(np.isfinite(scales)), "All scales should be finite."


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    test_cases,
)
def test_base10_scale_error_handling(shape, kwargs, expected_len):
    xs = np.zeros(shape)
    scales = get_log10_scales(xs, **kwargs)
    length = scales.shape[0]

    assert scales.shape == (expected_len,), \
        f"Scale shape should match {expected_len}, got {length}."
    assert np.all(scales > 0), "All scales should be greater than zero."
    assert np.all(np.isfinite(scales)), "All scales should be finite."


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    test_cases,
)
def test_datseries_scale_error_handling(shape, kwargs, expected_len):
    xs = np.zeros(shape)
    scales = get_datseries_scales(xs, **kwargs)
    length = scales.shape[0]

    assert scales.shape == (expected_len,), \
        f"Scale shape should match {expected_len}, got {length}."
    assert np.all(scales > 0), "All scales should be greater than zero."
    assert np.all(np.isfinite(scales)), "All scales should be finite."


@pytest.mark.parametrize(
        "method",
        ["log10", "datseries"]
)
@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    test_cases,
)
def test_adaptive_scale_error_handling(method, shape, kwargs, expected_len):
    xs = np.zeros(shape)
    scales = get_adaptive_scales(xs, method=method, **kwargs)
    length = scales.shape[0]

    assert scales.shape == (expected_len,), \
        f"Scale shape should match {expected_len}, got {length}."
    assert np.all(scales > 0), "All scales should be greater than zero."
    assert np.all(np.isfinite(scales)), "All scales should be finite."


@pytest.mark.parametrize(
    ("shape"),
    shape_opt
)
def test_all_scale_ends_increasing(rng, shape):
    xs = rng.random(shape)
    scales = get_log10_scales(xs)
    assert scales[0] < scales[-1], \
        "Log-10 scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = get_datseries_scales(xs)
    assert scales[0] < scales[-1], \
        "Datseries scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = get_adaptive_scales(xs, method="log-10")
    assert scales[0] < scales[-1], \
        "Log-10 adaptive scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = get_adaptive_scales(xs, method="datseries")
    assert scales[0] < scales[-1], \
        "Datseries adaptive scale: Last scale should be greater than first." \
        f"Got {scales=}"


@pytest.mark.parametrize(
    ("shape"),
    shape_opt
)
def test_all_scales_nondecreasing(rng, shape):
    xs = rng.random(shape)
    scales = get_log10_scales(xs)
    assert np.all(scales[:-1] <= scales[1:]), \
        "Log-10 scale: Inner scales should be non-decreasing."

    scales = get_datseries_scales(xs)
    assert np.all(scales[:-1] <= scales[1:]), \
        "Datseries scale: Inner scales should be non-decreasing."

    scales = get_adaptive_scales(xs, method="log-10")
    assert np.all(scales[:-1] <= scales[1:]), \
        "Log-10 adaptive scale: Inner scales should be non-decreasing."

    scales = get_adaptive_scales(xs, method="datseries")
    assert np.all(scales[:-1] <= scales[1:]), \
        "Datseries adaptive scale: Inner scales should be non-decreasing."

import numpy as np
import pytest

from piecewise_regression import Fit
from itertools import product

from pyspoc.rstatistics.fractal.funcs_py import (
    _compute_adj_r2,
    _get_segments,
    _get_log10_scales,
    _get_datseries_scales,
    _get_adaptive_scales,
    _compute_ols_results,
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
# Adjusted R²
# --------------------------------------------------
@pytest.mark.parametrize(
    ("H", "neg_log_scales", "n_breakpoints"),
    list(product(H_opt, neg_log_scales_opt, n_breakpoints_opt))
)
def test_adj_r2_is_finite_and_bounded(H, neg_log_scales, n_breakpoints):
    fit = Fit(neg_log_scales, H, n_breakpoints=n_breakpoints)
    results = fit.get_results()

    if not results["converged"]:
        assert True, "Fit not converged."
        return

    adj_r2 = _compute_adj_r2(H, fit)

    assert isinstance(adj_r2, float), \
        "R^2 is float value."
    assert np.isfinite(adj_r2), \
        "R^2 is finite."
    assert -1 <= adj_r2 <= 1, \
        "R^2 is between -1 and 1."


# --------------------------------------------------
# Elbow trimming
# --------------------------------------------------

alphas_opt = [np.array([0.0, 0.7, 0.8, 0.1]),
              np.array([1.0, 0.7, 0.8, 1.1]),
              np.array([0.1, 0.7, 0.8, 1.1]),
              np.array([-0.1, 0.7, 0.8, 1.9])]

elbow_tol_opt = [0.02, 0.05, 0.1, 0.15, 0.2]
seed_opt = [0, 101, 10862, 62]


@pytest.mark.parametrize(
    ("neg_log_scales", "H", "elbow_tol"),
    list(product(neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt))
)
def test_trim_elbows_trimmed_results_correct_shapes(
        neg_log_scales,
        H,
        elbow_tol):
    
    # Initialise construction of breakpoints and slopes.
    for seed in seed_opt:
        rng = np.random.default_rng(seed)
        prev_idx = 0

        # Add first scale value as initial breakpoint.
        breakpoint_list = [neg_log_scales[0]]
        alpha_list = []
        end = neg_log_scales.shape[0]
        i = 1
        
        # Step through scales (x-axis) and randomly allocate internal breakpoints.
        while i < end:
        
            # Choose if an internal breakpoint is added.
            if rng.uniform() > 0.5:

                # No internal breakpoint added, so continue.
                i+=1
                continue
            
            # Add internal breakpoint
            breakpoint = neg_log_scales[i]
            breakpoint_list.append(breakpoint)

            # Compute internal slope and store.
            alpha = (breakpoint - neg_log_scales[prev_idx]) / (H[i] - H[prev_idx])
            alpha_list.append(alpha)

            # Store previous index.
            prev_idx = i

            # Step forward twice (no adjacent breakpoints, otherwise no line segment!)
            i+=2
        
        # Add last scale value to breakpoint list.
        breakpoint_list.append(neg_log_scales[-1])

        # Cast to numpy.
        breakpoints = np.array(breakpoint_list)

        # If we had no internal breakpoints, compute full slope.
        if not alpha_list:
            alpha = (neg_log_scales[-1] - neg_log_scales[0]) / (H[-1] - H[0])
            alpha_list.append(alpha)

        # Cast to numpy.
        alphas = np.array(alpha_list)

        # Trim elbows.
        trimmed_scales, trimmed_H = _trim_elbows(alphas,
                                                breakpoints,
                                                neg_log_scales,
                                                H,
                                                elbow_tol)

        assert trimmed_scales.shape[0] > 1, \
            f"Data params: {seed=}" \
            "Trimmed scales must have at least 2 entries."
        assert trimmed_scales.shape[0] <= neg_log_scales.shape[0], \
            f"Data params: {seed=}" \
            "Trimmed scales length must be less than or equal to scales length."
        assert trimmed_H.shape[0] > 1, \
            f"Data params: {seed=}" \
            "Trimmed entropy must have at least 2 entries."
        assert trimmed_H.shape[0] <= H.shape[0], \
            f"Data params: {seed=}" \
            "Trimmed entropy length must be less than or equal to entropy length."
        assert trimmed_scales.shape == trimmed_H.shape, \
            f"Data params: {seed=}" \
            "Trimmed scales and trimmed entropy must have the same shape."
    

@pytest.mark.parametrize(
    ("neg_log_scales", "H", "elbow_tol"),
    list(product(neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt))
)
def test_trim_elbows_single_segment_yields_no_trimming(
        neg_log_scales,
        H,
        elbow_tol):
    
    alphas = np.array([1.0])
    breakpoints = np.array([neg_log_scales[0], neg_log_scales[-1]])
    trimmed_scales, trimmed_H = _trim_elbows(
        alphas,
        breakpoints,
        neg_log_scales,
        H,
        elbow_tol)
    
    assert trimmed_scales.shape == neg_log_scales.shape, \
        "Trimmed scales shape equal to scales shape."
    assert trimmed_H.shape == H.shape, \
        "Trimmed entropy shape equal to entropy shape."

@pytest.mark.parametrize(
    ("neg_log_scales", "H", "elbow_tol"),
    list(product(neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt))
)
def test_trim_elbows_two_segment_yields_only_end_trimming(
        rng,
        neg_log_scales,
        H,
        elbow_tol):
    
    alphas = np.array([0.0, 1.0])
    breakpoint = rng.choice(neg_log_scales[1:-1], size=1)
    breakpoints = np.array([neg_log_scales[0], breakpoint[0], neg_log_scales[-1]])
    trimmed_scales, trimmed_H = _trim_elbows(alphas,
                                             breakpoints,
                                             neg_log_scales,
                                             H,
                                             elbow_tol)
    
    assert trimmed_scales.shape[0] <= neg_log_scales.shape[0], \
        "Trimmed scales shorter than or equal to scales length."
    assert trimmed_H.shape[0] <= H.shape[0], \
        "Trimmed entropy shorter than or equal to entropy length."
    assert trimmed_scales[-1] == neg_log_scales[-1], \
        "Final trimmed scale and scale are equal."
    assert trimmed_H[-1] == H[-1], \
        "Final trimmed entropy and entropy are equal."
    

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
    
    fd = _compute_deshmukh_slope_estimate(
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

    fd = _compute_deshmukh_slope_estimate(
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

    fd = _compute_deshmukh_slope_estimate(
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

    fd = _compute_deshmukh_slope_estimate(
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
    scales = _get_log10_scales(xs, **kwargs)

    assert scales.shape == (expected_len,), "Scale shape should match num_scales."
    assert np.all(scales > 0), "All scales should be greater than zero."
    assert np.all(np.isfinite(scales)), "All scales should be finite."


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    test_cases,
)
def test_base10_scale_error_handling(shape, kwargs, expected_len):
    xs = np.zeros(shape)
    scales = _get_log10_scales(xs, **kwargs)
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
    scales = _get_datseries_scales(xs, **kwargs)
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
    scales = _get_adaptive_scales(xs, method=method, **kwargs)
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
    scales = _get_log10_scales(xs)
    assert scales[0] < scales[-1], \
        "Log-10 scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = _get_datseries_scales(xs)
    assert scales[0] < scales[-1], \
        "Datseries scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = _get_adaptive_scales(xs, method="log-10")
    assert scales[0] < scales[-1], \
        "Log-10 adaptive scale: Last scale should be greater than first." \
        f"Got {scales=}"

    scales = _get_adaptive_scales(xs, method="datseries")
    assert scales[0] < scales[-1], \
        "Datseries adaptive scale: Last scale should be greater than first." \
        f"Got {scales=}"


@pytest.mark.parametrize(
    ("shape"),
    shape_opt
)
def test_all_scales_nondecreasing(rng, shape):
    xs = rng.random(shape)
    scales = _get_log10_scales(xs)
    assert np.all(scales[:-1] <= scales[1:]), \
        "Log-10 scale: Inner scales should be non-decreasing."

    scales = _get_datseries_scales(xs)
    assert np.all(scales[:-1] <= scales[1:]), \
        "Datseries scale: Inner scales should be non-decreasing."

    scales = _get_adaptive_scales(xs, method="log-10")
    assert np.all(scales[:-1] <= scales[1:]), \
        "Log-10 adaptive scale: Inner scales should be non-decreasing."

    scales = _get_adaptive_scales(xs, method="datseries")
    assert np.all(scales[:-1] <= scales[1:]), \
        "Datseries adaptive scale: Inner scales should be non-decreasing."

# --------------------------------------------------
# End to end testing (data -> fractal dimension output)
# --------------------------------------------------
inner_configs = list(product(
        [0.005,0.01,0.05],
        [0.05,0.25,0.5],
        [0.1,0.3,0.5,1]
))

@pytest.mark.parametrize(
    ("slope"),
    [0.1,1,2,5]
)
def test_end_to_end_random(data_factory, slope):
    data_configs = product(seed_opt, shape_opt)

    for seed, shape in data_configs:
        xs = data_factory(shape[0], shape[1], seed)

        for adj_r2_thresh, elbow_thresh, deshmukh_reg_proportion in inner_configs:
            fd = run_end_to_end(seed,
                                xs,
                                slope,
                                adj_r2_thresh,
                                elbow_thresh,
                                deshmukh_reg_proportion)
            assert fd is not None, \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                "Fractal dimension should not be None."
            assert isinstance(fd, (float, np.floating)), \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                "Fractal dimension should be a float."
            assert fd >= 0, \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                f"Fractal dimension {fd} should be positive."
            assert np.isfinite(fd), \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                f"Fractal dimension {fd} should be finite."


@pytest.mark.parametrize(
    ("slope"),
    [0.1,1,2,5]
)
def test_end_to_end_degen(data_factory, slope):
    data_config = product(seed_opt, shape_opt)

    for seed, shape in data_config:
        xs = data_factory(shape[0], 1, seed)
    
        for adj_r2_thresh, elbow_thresh, deshmukh_reg_proportion in inner_configs:
            multis = data_factory(shape[1], 1, seed+1)
            xs = xs * multis.T
            fd = run_end_to_end(seed,
                                xs,
                                slope,
                                adj_r2_thresh,
                                elbow_thresh,
                                deshmukh_reg_proportion)
            assert fd is not None, \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                "Fractal dimension should not be None."
            assert isinstance(fd, (float, np.floating)), \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                "Fractal dimension should be a float."
            assert fd >= 0, \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                f"Fractal dimension {fd} should be positive."
            assert np.isfinite(fd), \
                f"Config params: {adj_r2_thresh=} {elbow_thresh=} {deshmukh_reg_proportion=}" \
                f"Data params: {shape=} {seed=}" \
                f"Fractal dimension {fd} should be finite."
            

def run_end_to_end(seed, xs, slope, adj_r2_thresh, elbow_thresh, deshmukh_reg_proportion):
    scales = _get_log10_scales(xs)
    neg_log_scales = -np.log(scales)
    rng = np.random.default_rng(seed)
    densities = neg_log_scales * slope + rng.normal(scale=slope / 2)
    
    # Perform OLS regression of the scaling curve as starting point.
    slope, adj_r2 = _compute_ols_results(neg_log_scales, densities)

    # Perform iterative piecewise linear regression of scaling curve.
    best_fit = _get_best_parsimonous_model_fit(
        neg_log_scales,
        densities,
        adj_r2,
        adj_r2_thresh)
    
    # Fallback to standard OLS slope if piecewise fails.
    if not best_fit:
        return slope
    
    # Get fitted linear segments from piecewise regression.
    alphas, breakpoints = _get_segments(
        best_fit,
        neg_log_scales)

    # Trim elbows from linear segments.
    trimmed_scales, trimmed_H = _trim_elbows(
        alphas,
        breakpoints,
        neg_log_scales,
        densities,
        elbow_thresh)
    
    # Compute fractal dimension using Deshmukh method.
    fd = _compute_deshmukh_slope_estimate(
        trimmed_scales,
        trimmed_H,
        deshmukh_reg_proportion)
    
    if not fd:
        fd = slope

    return fd

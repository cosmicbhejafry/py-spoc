import numpy as np
import pytest

from piecewise_regression import Fit
from itertools import product

from pyspoc.rstatistics.fractal.func import (
    _get_renyi_entropy,
    _get_best_parsimonous_model_fit,
    _compute_adj_r2,
    _get_pieces,
    _trim_elbows,
    _compute_slope_ensemble,
    _return_modal_average,
    _get_default_init_scale,
)

# --------------------------------------------------
# Best parsimonious model fit
# --------------------------------------------------
q_opt = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

neg_log_scales_opt = [
    np.linspace(-10, 10, 11),
    np.linspace(-12, 15, 11),
    np.linspace(-100, 100, 11)
]

scales_opt = [np.exp(x) for x in neg_log_scales_opt]

H_opt = [
    np.array([0.0, 0.0, 0.0, 0.2, 0.3, 0.2, 0.5, 0.4, 0.7, 0.7, 0.7]),
    np.array([1.0, 1.0, 1.0, 0.2, 1.3, 0.5, 1.5, 1.4, 1.7, 1.9, 2.0]),
    np.array([1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.4, 1.7, 1.9, 2.0]),
    np.array([1.0, 1.2, 1.4, 1.6, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2])
]

adj_r2_tol_opt = [0.01, 0.03, 0.05, 0.1, 0.2]

data_kwargs_opt = [
    {"n": 100, "dim": 2, "seed": 0},
    {"n": 100, "dim": 2, "seed": 10},
    {"n": 1000, "dim": 10, "seed": 0},
    {"n": 100, "dim": 100, "seed": 0}
]

n_breakpoints_opt = [0, 1, 2, 3]

params = list(product(neg_log_scales_opt, H_opt, adj_r2_tol_opt))

@pytest.mark.parametrize(
    ("neg_log_scales", "H", "adj_r2_tol"),
    params
)
def test_best_fit_returns_Fit_or_None(neg_log_scales, H, adj_r2_tol):
    best_fit = _get_best_parsimonous_model_fit(neg_log_scales,
                                               H,
                                               adj_r2_tol=adj_r2_tol)
    assert isinstance(best_fit, (Fit, None)), \
        "Piecewise fit must either by Fit object or None in case of non-convergence."
    
@pytest.mark.parametrize(
    ("neg_log_scales", "H", "adj_r2_tol"),
    params
)
def test_best_fit_exposes_expected_methods(neg_log_scales, H, adj_r2_tol):
    best_fit = _get_best_parsimonous_model_fit(neg_log_scales,
                                               H,
                                               adj_r2_tol=adj_r2_tol)

    if isinstance(best_fit, Fit):
        assert best_fit.n_breakpoints >= 0, \
            "Piecewise fit should have at least zero breakpoints."
        assert hasattr(best_fit, "get_results"), \
            "Piecewise fit should have a get_results attribute."
        assert hasattr(best_fit, "get_params"), \
            "Piecewise fit should have a get_params attribute."

@pytest.mark.parametrize(
    ("neg_log_scales", "H", "adj_r2_tol"),
    params
)
def test_best_fit_adj_r2_is_finite(neg_log_scales, H, adj_r2_tol):
    best_fit = _get_best_parsimonous_model_fit(neg_log_scales,
                                               H,
                                               adj_r2_tol=adj_r2_tol)

    if isinstance(best_fit, Fit):
        adj_r2 = _compute_adj_r2(H, best_fit)
        assert isinstance(adj_r2, float)
        assert np.isfinite(adj_r2)


# --------------------------------------------------
# Renyi entropy
# --------------------------------------------------
params = list(product(q_opt, data_kwargs_opt, scales_opt))

@pytest.mark.parametrize(
    ("q", "data_kwargs", "scales"),
    params
)
def test_renyi_entropy_basic_cases(data_factory, q, data_kwargs, scales):
    data = data_factory(**data_kwargs)
    H = _get_renyi_entropy(q, data, scales, 1e-6)

    assert H.shape == scales.shape, \
        "Entropy score for each scale."

@pytest.mark.parametrize(
    ("q", "data_kwargs", "scales"),
    params
)
def test_renyi_entropy_non_negative(q, data_factory, data_kwargs, scales):
    data = data_factory(**data_kwargs)
    H = _get_renyi_entropy(0.0, data, scales, 1e-6)
    assert np.all(np.isfinite(H)), \
        "All entropy values are finite."
    assert np.all(H >= 0), \
        "All entropy values non-negative."

# --------------------------------------------------
# Adjusted R²
# --------------------------------------------------
@pytest.mark.parametrize(
    ("H", "neg_log_scales", "n_breakpoints"),
    list(product(H_opt, neg_log_scales_opt, n_breakpoints_opt))
)
def test_adj_r2_is_finite_and_bounded(H, neg_log_scales, n_breakpoints):
    fit = Fit(neg_log_scales, H, n_breakpoints=n_breakpoints)
    adj_r2 = _compute_adj_r2(H, fit)

    assert isinstance(adj_r2, float), \
        "R^2 is float value."
    assert np.isfinite(adj_r2), \
        "R^2 is finite."
    assert -1 <= adj_r2 <= 1, \
        "R^2 is between -1 and 1."


# --------------------------------------------------
# Piece extraction
# --------------------------------------------------
@pytest.mark.parametrize(
    ("H", "neg_log_scales", "n_breakpoints", "expected_alpha_count"),
    list(product(
        H_opt,
        neg_log_scales_opt,
        n_breakpoints_opt,
        [n - 1 for n in n_breakpoints_opt]
    ))
)
def test_get_pieces_counts(H,
                           neg_log_scales,
                           n_breakpoints,
                           expected_alpha_count):
    
    fit = Fit(neg_log_scales,
              H,
              n_breakpoints=n_breakpoints)
    alphas, breakpoints = _get_pieces(fit, neg_log_scales)

    assert alphas.shape[0] == expected_alpha_count
    assert breakpoints.shape[0] == n_breakpoints
    assert breakpoints[0] == neg_log_scales[0]
    assert breakpoints[-1] == neg_log_scales[-1]


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
    ("alphas", "neg_log_scales", "H", "elbow_tol", "seed"),
    list(product(alphas_opt,
                 neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt,
                 seed_opt))
)
def test_trim_elbows_trimmed_results_type_checks(alphas,
                                                 neg_log_scales,
                                                 H,
                                                 elbow_tol,
                                                 seed):
    rng = np.random.default_rng(seed)
    breakpoints = rng.uniform(low = neg_log_scales[0],
                              high = neg_log_scales[-1],
                              size=alphas.shape)

    trimmed_scales, trimmed_H = _trim_elbows(alphas,
                                             breakpoints,
                                             neg_log_scales,
                                             H,
                                             elbow_tol)

    assert isinstance(trimmed_scales, np.ndarray), \
        "Trimmed scales is a valid numpy array."

    assert isinstance(trimmed_H, np.ndarray), \
        "Trimmed entropy is a valid numpy array."

@pytest.mark.parametrize(
    ("alphas", "neg_log_scales", "H", "elbow_tol", "seed"),
    list(product(alphas_opt,
                 neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt,
                 seed_opt))
)
def test_trim_elbows_trimmed_results_correct_shapes(alphas,
                                                    neg_log_scales,
                                                    H,
                                                    elbow_tol,
                                                    seed):
    
    rng = np.random.default_rng(seed)
    breakpoints = rng.uniform(low = neg_log_scales[0],
                              high = neg_log_scales[-1],
                              size=alphas.shape)

    trimmed_scales, trimmed_H = _trim_elbows(alphas,
                                             breakpoints,
                                             neg_log_scales,
                                             H,
                                             elbow_tol)
    
    assert trimmed_scales.shape[0] > 1, \
        "Trimmed scales has at least 2 entries."
    assert trimmed_scales.shape[0] <= neg_log_scales.shape[0], \
        "Trimmed scales length less than or equal to scales length."
    assert trimmed_H.shape[0] > 1, \
        "Trimmed entropy has at least 2 entries."
    assert trimmed_H.shape[0] <= H.shape[0], \
        "Trimmed entropy length less than or equal to entropy length."
    assert trimmed_scales.shape == trimmed_H.shape, \
        "Trimmed scales and trimmed entropy have the same shape."
    

@pytest.mark.parametrize(
    ("neg_log_scales", "H", "elbow_tol", "seed"),
    list(product(neg_log_scales_opt,
                 H_opt,
                 elbow_tol_opt,
                 seed_opt))
)
def test_trim_elbows_single_piece_yields_no_trimming(neg_log_scales,
                                                     H,
                                                     elbow_tol,
                                                     seed):
    alphas = np.array([1.0])
    rng = np.random.default_rng(seed)
    breakpoints = rng.uniform(low = neg_log_scales[0],
                              high = neg_log_scales[-1],
                              size=alphas.shape)
    
    trimmed_scales, trimmed_H = _trim_elbows(alphas,
                                             breakpoints,
                                             neg_log_scales,
                                             H,
                                             elbow_tol)
    
    assert trimmed_scales.shape == neg_log_scales.shape, \
        "Trimmed scales shape equal to scales shape."
    assert trimmed_H.shape == H.shape, \
        "Trimmed entropy shape equal to entropy shape."


def test_trim_elbows_two_piece_yields_only_end_trimming(neg_log_scales,
                                                        H,
                                                        elbow_tol,
                                                        seed):
    alphas = np.array([0.0, 1.0])
    rng = np.random.default_rng(seed)
    breakpoints = rng.uniform(low = neg_log_scales[0],
                              high = neg_log_scales[-1],
                              size=alphas.shape)
    
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
prop_window_size_opt = [0.1, 0.2, 0.3, 0.5, 1.0]

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
    
    slopes, weights = _compute_slope_ensemble(trimmed_scales,
                                              trimmed_H,
                                              prop_window_size)


    assert len(slopes) > 0, \
        "Slopes are returned successfully."
    assert slopes.shape == weights.shape, \
        "Slopes shape is the same as weights shape."
    assert np.all(np.isfinite(slopes)), \
        "All slopes are finite."

    assert np.all(weights >= 0), \
        "All weights are non-negative."
    assert 0.999 < weights.sum() < 1.001, \
        "Weights sum to approximately 1."


# --------------------------------------------------
# Modal averaging
# --------------------------------------------------

def test_modal_average_single_mode():
    slopes = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.1])

    result = _return_modal_average(slopes, weights)

    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)


def test_modal_average_multimodal(rng):
    slopes = np.concatenate([
        rng.normal(1.0, 0.1, 50),
        rng.normal(3.0, 0.1, 50),
    ])
    weights = np.ones_like(slopes)

    result = _return_modal_average(slopes, weights)

    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)


# --------------------------------------------------
# Default scale generation
# --------------------------------------------------

@pytest.mark.parametrize(
    ("shape", "kwargs", "expected_len"),
    [
        ((50, 2), {}, 401),
        ((30, 3), {"ref_scale": 0.5}, 401),
        ((20, 2), {"num_scales": 100}, 101),
        ((40, 2), {}, 401),
    ],
)
def test_default_init_scale_basic_properties(rng, shape, kwargs, expected_len):
    xs = rng.random(shape)
    scales = _get_default_init_scale(xs, **kwargs)

    assert scales.shape == (expected_len,)
    assert np.all(scales > 0)
    assert np.all(np.isfinite(scales))


def test_default_init_scale_ref_scale_decreases(rng):
    xs = rng.random((30, 3))
    scales = _get_default_init_scale(xs, ref_scale=0.5)

    assert scales[0] > scales[-1]


def test_default_init_scale_monotonic_decreasing(rng):
    xs = rng.random((40, 2))
    scales = _get_default_init_scale(xs)

    assert np.all(scales[:-1] > scales[1:])

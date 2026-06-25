from __future__ import annotations

import numpy as np
import piecewise_regression

from scipy import stats
from scipy.signal import argrelmax
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from piecewise_regression import Fit

def _get_renyi_entropy(q: float,
                       data : np.ndarray,
                       scales: np.ndarray,
                       shannon_entropy_tol: float = 1e-6) -> np.ndarray:
    """
    Compute Renyi entropy according to the formula

    :math:`H_q = \\frac{1}{1 - q} log (\\sum_{i=1}^M p_i^q)`

    for M non-overlapping boxes covering the set of points and
    p_i are the proportion of points within that box.

    Parameters:
        data (ndarray): Array of points (n, p).
        scale (float): Scale factor.

    Returns:
        list: Unique grid boxes.
    """

    points = np.expand_dims(data, axis=2)
    results = (points / scales).astype(int)
    H : np.ndarray = np.empty(shape=results.shape[2], dtype=float)

    for k in range(H.shape[0]):
        _, cnts = np.unique(results[:,:,k], axis=0, return_counts=True)

        if q == 0:
            H[k] = np.log(cnts.shape[0])
            continue

        probs = cnts / cnts.sum()

        if abs(q - 1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H

def _get_best_parsimonous_model_fit(neg_log_scales: np.ndarray,
                                    H: np.ndarray,
                                    adj_r2_tol: float) -> Fit | None:

    max_breakpoints = max(int(neg_log_scales.shape[0] / 4), 1)
    best_fit = _compute_pw_fit(neg_log_scales, H, 0)

    if best_fit is None:
        return

    prev_adj_r2 = _compute_adj_r2(H, best_fit)

    for n_breakpoints in range(1, max_breakpoints + 1):
        pw_fit = _compute_pw_fit(neg_log_scales,
                              H,
                              n_breakpoints)
        if pw_fit is None:
            break

        adj_r2 = _compute_adj_r2(H, pw_fit)
        improvement = adj_r2 - prev_adj_r2

        if improvement < adj_r2_tol:
            break
            
        prev_adj_r2 = adj_r2
        best_fit = pw_fit

    return best_fit

def _compute_pw_fit(neg_log_scales: np.ndarray,
                    H: np.ndarray,
                    n_breakpoints: int) -> Fit | None:
    
    pw_fit = piecewise_regression.Fit(neg_log_scales,
                                      H,
                                      n_breakpoints=n_breakpoints)
    results = pw_fit.get_results()

    if not results["converged"]:
        return # Terminate on non-convergence.
    
    return pw_fit

def _compute_adj_r2(log_boxes: np.ndarray,
                    piecewise_fit: Fit) -> float:
    
    results = piecewise_fit.get_results()
    n = log_boxes.shape[0]
    p = piecewise_fit.n_breakpoints * 2
    tss = log_boxes.var() * log_boxes.shape[0]
    rss = results["rss"]
    r_sq = 1 - rss / tss
    adj_r_sq = 1 - (n - 1) / (n - 1 - p) * (1 - r_sq)
    return adj_r_sq

def _get_pieces(best_fit: Fit,
                neg_log_scales: np.ndarray):

    n_breakpoints = best_fit.n_breakpoints
    fit_params = best_fit.get_params()
    alphas = np.empty(shape=n_breakpoints + 1)
    breakpoints = np.empty(shape=n_breakpoints + 2)
    breakpoints[0] = neg_log_scales[0]
    breakpoints[-1] = neg_log_scales[-1]
    
    for i in range(n_breakpoints + 1):
        j = i + 1
        alpha_key = f"alpha{j}"
        breakpoint_key = f"breakpoint{j}"
        alpha = fit_params[alpha_key]
        alphas[i] = alpha
        breakpoint_val = fit_params.get(breakpoint_key)

        if breakpoint_val:
            breakpoints[j] = breakpoint_val
        
    return alphas, breakpoints

def _trim_elbows(alphas: np.ndarray,
                 breakpoints: np.ndarray,
                 neg_log_scales: np.ndarray,
                 H: np.ndarray,
                 elbow_tol: float) -> tuple[np.ndarray, np.ndarray]:

    # Grab # of pieces.
    n_alphas = alphas.shape[0]

    # If only a single linear piece, return full series.
    if n_alphas == 1:
        return neg_log_scales, H

    # Compute total average if only two pieces, otherwise inner average.
    avg = alphas.mean() if n_alphas == 2 else alphas[1:-1].mean()

    # Initialise full interval.
    interval_start = breakpoints[0]
    interval_end = breakpoints[-1]

    # If first piece is below tolerance, trim.
    if alphas[0] < avg * elbow_tol:
        interval_start = breakpoints[1]
    
    # If last piece is below tolerance, trim.
    if alphas[-1] < avg * elbow_tol:
        interval_end = breakpoints[-2]

    # Return trimmed series.
    mask = (neg_log_scales >= interval_start) & (neg_log_scales <= interval_end)
    return neg_log_scales[mask], H[mask]

def _compute_slope_ensemble(trimmed_scales: np.ndarray,
                            trimmed_H: np.ndarray,
                            deshmukh_reg_proportion: float) -> tuple[np.ndarray, np.ndarray]:

    N = trimmed_scales.shape[0]
    n = int(deshmukh_reg_proportion * N)
    S = int((N - n - 1) * (N - n) / 2)
    slopes = np.zeros(shape=S)
    lengths = np.zeros(shape=S)
    errors = np.zeros(shape=S)
    i = 0

    for lhs in range(N - n):
        for rhs in range(lhs + n + 1, N):
            reg_length = rhs - lhs
            xs = trimmed_scales[lhs:rhs]
            ys = trimmed_H[lhs:rhs]
            m, c, r_value, p_value, _ = stats.linregress(
                xs, ys)
            slopes[i] = m
            errors[i] = np.sqrt(((ys - m * xs - c)**2).sum() / reg_length)
            lengths[i] = np.sqrt(1 + m**2) * (xs[-1] - xs[0])
            i += 1

    weights = lengths / errors
    return slopes, weights

def _return_modal_average(slopes: np.ndarray,
                          weights: np.ndarray) -> float:

    weights = weights * weights.shape[0] / weights.sum()
    samples = weights * slopes
    xs = np.linspace(samples.min(), samples.max(), 1001)
    kernel = stats.gaussian_kde(samples, bw_method="scott")
    ys = kernel(xs)
    mode_idxs = argrelmax(ys)
    modes = xs[mode_idxs[0]]
    mode_vals = ys[mode_idxs[0]]
    mode_weights = mode_vals / mode_vals.sum()
    return (modes * mode_weights).sum()


def _get_default_init_scale(xs: np.ndarray,
                            step_size: float = 0.025,
                            num_scales: int = 400,
                            num_scales_right: int = 200,
                            ref_scale: Optional[float] = None) -> np.ndarray:

    # Default scale initialization if not provided
    if not ref_scale:
        eps = 1e-20
        ref_scale = np.mean(np.linalg.norm(xs[:-1, :] - xs[1:, :], axis=1)) + eps

    num_scales_left = num_scales - num_scales_right
    init_scales = ref_scale * np.power(
       10,
       step_size * np.linspace(num_scales_left, -num_scales_right, num_scales + 1),
    )
    return init_scales



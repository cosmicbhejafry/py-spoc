from __future__ import annotations

import numpy as np
import piecewise_regression
import statsmodels.api as sm

from typing import Literal, Any, TYPE_CHECKING
from scipy.signal import savgol_filter

from . import funcs_numba as fnb

if TYPE_CHECKING:
    from piecewise_regression import Fit


_ERROR_SCALING_FAILURE = "No scaling could be determined for the provided dataset. " \
    "Returning NaN result."


def _find_elbow_idx(
    x: np.ndarray,
    y: np.ndarray,
    *,
    direction: str = "decreasing",
    smooth: bool = True,
    window_length: int | None = None,
    polyorder: int = 2,
    monotonic_tolerance: float = 0.02) -> tuple[int, int] | None:
    """
    Find the elbow point of an approximately monotonic curve.

    Uses the maximum perpendicular distance from the chord between the
    first and last normalized points.

    Parameters
    ----------
    x:
        One-dimensional x values.
    y:
        One-dimensional y values.
    direction:
        "decreasing" or "increasing".
    smooth:
        Whether to smooth y before elbow detection.
    window_length:
        Savitzky-Golay smoothing window. If None, chosen automatically.
    polyorder:
        Polynomial order for Savitzky-Golay smoothing.
    monotonic_tolerance:
        Fraction of local monotonicity violations allowed.

    Returns
    -------
    elbow_x, elbow_y, elbow_index
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if n != len(y):
        raise ValueError("x and y must have the same length.")

    if n < 3:
        raise ValueError("At least three points are required.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_work = y.copy()

    if smooth and len(y) >= 5:
        if window_length is None:
            # Rough default: around 10% of data length, forced odd.
            window_length = max(5, int(round(len(y) * 0.1)))
            
            if window_length % 2 == 0:
                window_length += 1

        # Savitzky-Golay requires an odd window <= len(y).
        window_length = min(window_length, len(y) if len(y) % 2 == 1 else len(y) - 1)

        if window_length > polyorder:
            y_work = savgol_filter(y_work, window_length, polyorder)

    # Normalize x and y to [0, 1].
    x_norm = (x - x.min()) / (x.max() - x.min())

    y_min = y_work.min() # type: ignore
    y_max = y_work.max() # type: ignore

    if y_max == y_min:
        return
    
    if not fnb._is_nearly_monotonic(
        y_work,
        direction=direction,
        tolerance=monotonic_tolerance):

        return

    y_norm = (y_work - y_min) / (y_max - y_min)
    points = np.column_stack([x_norm, y_norm])
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_norm = np.linalg.norm(chord)

    if chord_norm == 0:
        return
    
    # Perpendicular distance from each point to the line through start/end.
    distances = np.cross(chord, points - start)
    left_elbow_idx = 0
    right_elbow_idx = n
    
    candidate_idx = int(np.argmin(distances))
    
    if candidate_idx > 0:
        elbow_y = y[candidate_idx]

        if elbow_y < (y_max + y_min) / 2:
            left_elbow_idx = candidate_idx

    candidate_idx = int(np.argmax(distances))

    if candidate_idx < n - 1:
        elbow_y = y[candidate_idx]

        if elbow_y > (y_max + y_min) / 2:
            right_elbow_idx = candidate_idx

    return left_elbow_idx, right_elbow_idx


def _get_best_parsimonious_model_fit(
        neg_log_scales: np.ndarray,
        densities: np.ndarray,
        best_adj_r2: float,
        adj_r2_tol: float = 0.001) -> dict[str, Any] | None:

    max_breakpoints = max(int(neg_log_scales.shape[0] / 4), 1)
    fit_scales = neg_log_scales.copy()
    fit_densities = densities.copy()
    prev_adj_r2 = best_adj_r2
    best_fit = None

    for n_breakpoints in range(1, max_breakpoints + 1):
        pw_fit = _compute_pw_fit(fit_scales,
            fit_densities,
            n_breakpoints)

        if pw_fit is None:
            
            if not best_fit:
                break

            fit_params = best_fit.get_params()
            c = fit_params.get("const", 0)

            if not c > 0:
                break
            
            include_mask = fit_densities > c
            fit_scales = fit_scales[include_mask]
            fit_densities = fit_densities[include_mask]

            if fit_scales.shape[0] < neg_log_scales.shape[0] * 2:
                break

            continue

        adj_r2 = _compute_adj_r2(fit_densities, pw_fit)
        improvement = adj_r2 - prev_adj_r2

        if improvement < adj_r2_tol:
            break
            
        prev_adj_r2 = adj_r2
        best_fit = pw_fit

    if not best_fit:
        return
    
    best_params = best_fit.get_params()
    best_params = _add_bookend_bps(fit_scales[0], fit_scales[-1], best_params)
    return best_params


def _add_bookend_bps(
        first_bp: float,
        last_bp: float,
        fit_params: dict[str, Any]) -> dict[str, Any]:
    
    adjusted_params = dict()
    fit_params["breakpoint0"] = first_bp
    max_idx = 0
    
    for key, val in fit_params.items():
        if not key.startswith("alpha"):
            continue

        str_idx = key[5:]
        idx = int(str_idx)

        if idx > max_idx:
            max_idx = idx

        adjusted_params[f"breakpoint{idx}"] = fit_params[f"breakpoint{idx - 1}"]
        adjusted_params[f"alpha{idx}"] = fit_params[f"alpha{idx}"]

    adjusted_params[f"breakpoint{max_idx+1}"] = last_bp
    return adjusted_params


def _get_segments(
        fit_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        
    segments = list()

    for key, val in fit_params.items():
        if not key.startswith("breakpoint"):
            continue

        str_idx = key[10:]
        alpha = fit_params.get(f"alpha{str_idx}")
        bp = val
        segments.append((alpha, bp))

    if not segments:
        return
    
    sorted_segments = sorted(segments, key=lambda t: t[1])
    alphas, bps = tuple(zip(*sorted_segments))
    return np.array([a for a in alphas if a]), np.array(bps)


def _compute_pw_fit(
        neg_log_scales: np.ndarray,
        H: np.ndarray,
        n_breakpoints: int) -> Fit | None:
    
    pw_fit = piecewise_regression.Fit(neg_log_scales,
                                      H,
                                      n_breakpoints=n_breakpoints,
                                      max_iterations=100,
                                      n_boot=5)
    results = pw_fit.get_results()

    if not results["converged"]:
        return # Terminate on non-convergence.
    
    return pw_fit


def _compute_ols_results(
        neg_log_scales: np.ndarray,
        H: np.ndarray) -> tuple[float, float]:
    
    constant_added = sm.add_constant(neg_log_scales)
    ols = sm.OLS(H, constant_added)
    results = ols.fit()
    coeffs = results.params
    slope = coeffs[0] if len(coeffs) == 1 else coeffs[1]
    adj_r2 = results.rsquared_adj
    return slope, adj_r2


def _compute_adj_r2(
        log_boxes: np.ndarray,
        piecewise_fit: Fit) -> float:
    
    results = piecewise_fit.get_results()
    n = log_boxes.shape[0]
    p = piecewise_fit.n_breakpoints * 2
    tss = log_boxes.var() * n
    rss = results["rss"]
    r_sq = 1 - rss / tss
    adj_r_sq = 1 - (n - 1) / (n - 1 - p) * (1 - r_sq)
    return adj_r_sq


def _get_adaptive_scales(
        xs: np.ndarray,
        *,
        method: Literal["datseries", "log-10"],
        k: int = 50,
        max_iter: int = 5,
        max_iter_alert: bool = False,
        **kwargs):
    
    scale_func = _get_log10_scales if method == "log-10" \
        else _get_datseries_scales
    i = 0
    scales = scale_func(xs, k=k, **kwargs)
    n = xs.shape[0]

    while i < max_iter:
        
        results = fnb._get_bounded_idxs(xs, scales, 1, n)
        
        if None in results:
            raise ValueError("Unable to establish informative scales.")
    
        start_idx, end_idx, start_box_cnt, end_box_cnt = results
        start_scale, end_scale = scales[start_idx], scales[end_idx]
        length = end_idx - start_idx + 1
        range = start_box_cnt - end_box_cnt

        # Expand scales as end of the scaling curve not met.
        if start_box_cnt < 0.95 * n:
            start_scale /= 2

        # Expand scales as start of the scaling curve not met.
        if end_box_cnt > 1:
            end_scale *= 2
        
        if range >= 0.85 * n and length >= k + 1:
            break
        
        scales = _get_exp_scales(start_scale, end_scale, k)
        i += 1

    if max_iter_alert and i == max_iter:
        print("Maximum iterations were reached.")
        
    return scales
    

def _get_exp_scales(
        start: float,
        end: float,
        k: int = 50,
        zero_tol: float = 0.001) -> np.ndarray:
    
    if start < zero_tol:
        start = zero_tol

    log_scales = np.linspace(
        np.log(start),
        np.log(end),
        num=k+1)
    scales = np.exp(log_scales)
    
    if scales is None:
        raise ValueError(_ERROR_SCALING_FAILURE)
    
    return scales
        

def _get_log10_scales(
        xs: np.ndarray,
        *,
        k: int = 50,
        delta: float = 0.025,
        prop_scales_right: float = 0.5) -> np.ndarray:

    if prop_scales_right > 1:
        prop_scales_right = 1

    if prop_scales_right < 0:
        prop_scales_right = 0
    
    xs = _sort_data(xs)
    eps = 1e-20
    ref_scales = np.mean(np.linalg.norm(xs[:-1, :] - xs[1:, :], axis=1)) + eps
    num_scales_right = int(k * prop_scales_right)
    num_scales_left = k - num_scales_right
    log10_scale = delta * np.linspace(-num_scales_left, num_scales_right, k + 1)
    scales = ref_scales * np.power(10, log10_scale)

    # Raise error if no scaling could be generated.
    if scales is None:
        raise ValueError(_ERROR_SCALING_FAILURE)

    return scales


def _get_datseries_scales(
        xs: np.ndarray,
        *,
        k: int = 50,
        psi: float = 1.0,
        zeta: float = 1.0) -> np.ndarray:
    
    xs = _sort_data(xs)
    eps = 1e-6
    
    # Includes corrective term for high dimensionality.
    eps_cross = max(np.min(np.linalg.norm(xs[:-1, :] - xs[1:, :], axis=1)), eps)
    eps_within = max(np.mean(xs.max(axis=0) - xs.min(axis=0)), eps)
    log_cross = np.log(eps_cross) + psi
    log_within = np.log(eps_within) - zeta

    if log_cross > log_within:
        start = log_within
        end = log_cross
    else:
        start = log_cross
        end = log_within

    log_scales = np.linspace(start, end, num=k+1)
    scales = np.exp(log_scales)

    # Raise error if no scaling could be generated.
    if scales is None:
        raise ValueError(_ERROR_SCALING_FAILURE)

    return scales


def _argsort_data(data: np.ndarray) -> np.ndarray:
    keys = tuple(data[:, j] for j in range(data.shape[1] - 1, -1, -1))
    idx = np.lexsort(keys)
    return idx


def _sort_data(data: np.ndarray) -> np.ndarray:
    idx = _argsort_data(data)
    return data[idx]

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from typing import Literal
from scipy.signal import savgol_filter

from pyspoc._utils.arrays import sort_data
from . import _funcs_numba as fnb


_ERROR_SCALING_FAILURE = "No scaling could be determined for the provided dataset. " \
    "Returning NaN result."


def find_elbow_idx(
    x: np.ndarray[tuple[int], np.dtype[np.floating]],
    y: np.ndarray[tuple[int], np.dtype[np.floating]],
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
    
    if not fnb.is_nearly_monotonic(
        y_work,
        direction=direction,
        tolerance=monotonic_tolerance):
        return

    y_norm = (y_work - y_min) / (y_max - y_min)
    points = np.column_stack((x_norm, y_norm)) # pyright: ignore[reportCallIssue]
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_norm = np.linalg.norm(chord)

    if chord_norm == 0:
        return
    
    # Perpendicular distance from each point to the line through start/end.
    offsets = points - start
    distances = (
        chord[0] * offsets[:, 1]
        - chord[1] * offsets[:, 0]) / chord_norm
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


def compute_ols_results(
        log_scales: np.ndarray,
        H: np.ndarray) -> tuple[np.floating, np.floating]:
    
    constant_added = sm.add_constant(log_scales)
    ols = sm.OLS(H, constant_added)
    results = ols.fit()
    coeffs = results.params
    slope = coeffs[0] if len(coeffs) == 1 else coeffs[1]
    r2 = results.rsquared
    return slope, r2


def get_adaptive_box_scales(
        xs: np.ndarray,
        *,
        method: Literal["datseries", "log-10"],
        k: int = 50,
        max_iter: int = 5,
        max_iter_alert: bool = False,
        **kwargs
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    
    scale_func = (get_log10_scales
        if method == "log-10"
        else get_datseries_scales)
    scales = scale_func(xs, k=k, **kwargs)
    n = xs.shape[0]
    lower_bound = 1
    upper_bound = n

    scales, _ = get_adapted_results(
        xs=xs,
        seed_scales=scales,
        calling_method="renyi",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        k=k,
        max_iter=max_iter,
        max_iter_alert=max_iter_alert,
        reverse_scale=True)

    return scales


def get_adaptive_corr_sum_results(
        xs: np.ndarray,
        *,
        method: Literal["datseries", "log-10"],
        k: int = 50,
        max_iter: int = 5,
        max_iter_alert: bool = False,
        **kwargs
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]],
           np.ndarray[tuple[int], np.dtype[np.integer]]]:
    
    scale_func = (get_log10_scales
        if method == "log-10"
        else get_datseries_scales)
    scales = scale_func(xs, k=k, **kwargs)
    n = xs.shape[0]
    lower_bound = 1
    upper_bound = int(n * (n-1) / 2)
    scales, counts = get_adapted_results(
        xs=xs,
        seed_scales=scales,
        calling_method="corrsum",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        k=k,
        max_iter=max_iter,
        max_iter_alert=max_iter_alert,
        reverse_scale=False)

    if counts is None:
        raise RuntimeError("Unable to establish informative counts.")
        
    return scales, counts


def get_adapted_results(
        xs: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        seed_scales: np.ndarray[tuple[int], np.dtype[np.floating]],
        *,
        calling_method: Literal["renyi", "corrsum", "evt"],
        lower_bound: int,
        upper_bound: int,
        k: int = 50,
        max_iter: int = 5,
        max_iter_alert: bool = False,
        reverse_scale: bool = False,
        **kwargs
) -> (tuple[np.ndarray[tuple[int], np.dtype[np.floating]],
            np.ndarray[tuple[int], np.dtype[np.integer]] | None]):

    i = 0
    scales = seed_scales
    counts = None

    while True:
        
        if calling_method == "renyi":
            results = fnb.get_box_bounds(
                xs, scales, lower_bound, upper_bound, boxes_sorted=True)
            high_count_idx, low_count_idx, high_count, low_count = results
        elif calling_method == "corrsum":
            results = fnb.get_radii_bounds_and_counts(
                xs, scales, lower_bound, upper_bound, radii_sorted=True)
            high_count_idx, low_count_idx, high_count, low_count, counts = results
        
        if None in (high_count_idx, low_count_idx, high_count, low_count):
            raise RuntimeError("Unable to establish informative scales.")
    
        high_count_scale, low_count_scale = scales[high_count_idx], scales[low_count_idx]
        length = abs(low_count_idx - high_count_idx) + 1
        range = high_count - low_count

        # Expand scales as end of the scaling curve not met.
        if high_count < 0.95 * upper_bound:
            high_count_scale = (high_count_scale / 2
                if reverse_scale
                else high_count_scale * 2)

        # Expand scales as start of the scaling curve not met.
        if low_count > lower_bound:
            low_count_scale = (low_count_scale * 2
                if reverse_scale
                else low_count_scale / 2)
                
        if range >= 0.9 * upper_bound and length >= k + 1:
            break
        
        i += 1

        if i >= max_iter:
            break
        
        scales = (get_exp_scales(high_count_scale, low_count_scale, k)
            if reverse_scale
            else get_exp_scales(low_count_scale, high_count_scale, k))

    if max_iter_alert and i == max_iter:
        print("Maximum iterations were reached.")
        
    return scales, counts
    

def get_exp_scales(
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
        

def get_log10_scales(
        xs: np.ndarray,
        *,
        k: int = 50,
        delta: float = 0.025,
        prop_scales_right: float = 0.5) -> np.ndarray:

    if prop_scales_right > 1:
        prop_scales_right = 1

    if prop_scales_right < 0:
        prop_scales_right = 0
    
    xs = sort_data(xs)
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


def get_datseries_scales(
        xs: np.ndarray,
        *,
        k: int = 50,
        psi: float = 1.0,
        zeta: float = 1.0) -> np.ndarray:
    
    xs = sort_data(xs)
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

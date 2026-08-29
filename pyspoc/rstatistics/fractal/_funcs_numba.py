from __future__ import annotations

import numpy as np
import sys

from numba import jit_module
from numba.core import types
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List

from pyspoc.settings import settings
from pyspoc._numba import install_numba_funcs
from pyspoc._utils.arrays.numba import array_equal_numba


list_type = types.ListType(types.int64)


def get_row_hashes(x: np.ndarray) -> np.ndarray:
    """
    Compute a 64-bit hash for each row of an integer matrix.

    Parameters
    ----------
    x : np.ndarray
        Integer array of shape ``(n, p)``. Each row is treated as one box ID.

    Returns
    -------
    np.ndarray
        Unsigned 64-bit hash value for each row of ``x``.

    Notes
    -----
    This helper is intended for integer box IDs produced by discretizing data
    with ``floor(data / scale)``. Signed values are cast to ``uint64`` before
    mixing. Overflow during multiplication is intentional and gives arithmetic
    modulo ``2**64``.

    Hashes are not unique identifiers. Callers that use these hashes for
    grouping must compare candidate rows inside each hash bucket to remain
    exact under collisions.
    """
    
    #x = np.ascontiguousarray(x, dtype=np.int64)
    #x = x + x.min(axis=0)
    n, p = x.shape
    h = np.empty(n, dtype=np.uint64)

    offset = np.uint64(1469598103934665603)
    mix = np.uint64(0x9E3779B97F4A7C15)
    prime = np.uint64(1099511628211)

    for i in range(n):
        hi = offset

        for j in range(p):
            value = np.uint64(x[i, j])
            hi ^= value + mix
            hi *= prime

        h[i] = hi

    return h


def get_box_tallies(box_ids: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    """
    Count repeated box IDs using hash buckets with exact collision checks.
    Returns the tallies for the number of times a box is present.

    Parameters
    ----------
    box_ids : np.ndarray
        Integer array of shape ``(n, p)`` containing one box ID per point.
    hashes : np.ndarray
        Unsigned 64-bit hash array of shape ``(n,)``. Usually produced by
        :func:`_hash_rows` from ``box_ids``.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n,)``. Entries corresponding to representative
        rows contain the count for that unique row; all non-representative rows
        are zero. The occupied-box counts are therefore ``counts[counts > 0]``.

    Notes
    -----
    The function is exact: rows with equal hashes are compared element by
    element before their counts are merged. Hash collisions therefore affect
    performance but not correctness.
    """
    d = Dict.empty(
        key_type=types.uint64,
        value_type=list_type)
    n = hashes.shape[0]
    cnts = np.zeros(shape=n, dtype=np.int64)

    for i in range(n):
        h = hashes[i]

        if h not in d:
            d[h] = List.empty_list(types.int64)
            d[h].append(i)
            cnts[i] = 1
            continue

        existing_idxs = d[h]
        is_existing_x = False

        for existing_idx in existing_idxs:
            x_existing = box_ids[existing_idx]

            if array_equal_numba(box_ids[i], x_existing, equal_nan=True):
                is_existing_x = True
                cnts[existing_idx] += 1
                break
            
        if not is_existing_x:
            d[h].append(i)
            cnts[i] = 1

    return cnts


def get_box_count(data: np.ndarray, scale: float) -> int:
    """
    Count occupied boxes for a dataset at a single box scale.

    Parameters
    ----------
    data : np.ndarray
        Point array of shape ``(n, p)``.
    scale : float
        Positive box side length used to discretize ``data``.

    Returns
    -------
    int
        Number of unique occupied boxes at ``scale``.

    Notes
    -----
    Box IDs are computed as ``floor(data / scale)`` and grouped exactly using
    :func:`_hash_rows` and :func:`_get_counts`.
    """
    boxes = np.floor(data / scale).astype(np.int64)
    hashes = get_row_hashes(boxes)
    cnts = get_box_tallies(boxes, hashes)
    non_zero = cnts[cnts > 0]
    box_cnt = non_zero.shape[0]
    return box_cnt


def get_bounded_idxs(
        data,
        scales,
        lower_bound: int | None = None,
        upper_bound: int | None = None) -> \
            tuple[int | None, int | None, int | None, int | None]:
    """
    Locate scale-index bounds based on occupied-box count thresholds.

    Parameters
    ----------
    data : np.ndarray
        Point array of shape ``(n, p)``.
    scales : np.ndarray
        One-dimensional scale grid. The occupied-box count is assumed to vary
        monotonically along this grid.
    lower_bound : int or None, default=None
        Lower accepted occupied-box count. The returned interval ends at the
        first scale whose box count is greater than or equal to this threshold.
    upper_bound : int or None, default=None
        Upper accepted occupied-box count. The returned interval starts at the
        first scale whose box count is less than or equal to this threshold.
    
    Returns
    -------
    tuple[int | None, int | None, int | None, int | None]
        Inclusive ``(start_idx, end_idx)`` of ``scales`` and
        ``(start_box_count, end_box_count)`` that satisfy the criteria. If no
        entries satisfy the criteria, a ``None`` tuple is returned.

    Notes
    -----
    When both bounds are supplied, the function recursively computes the lower
    and upper index limits and combines them. This is used to trim saturated
    regions where the box count is near one or near the number of points.

    If no bounds are provided, the original scales are returned.
    """
    
    k = scales.shape[0]
    blank_return = None, None, None, None

    if lower_bound is None and upper_bound is None:
        return 0, k - 1, None, None
    
    if lower_bound is not None and upper_bound is not None:

        if lower_bound > upper_bound:
            return blank_return

        lower_start_idx, lower_end_idx, _, end_box_cnt = \
            get_bounded_idxs(data, scales, lower_bound=lower_bound)
        upper_start_idx, upper_end_idx, start_box_cnt, _ = \
            get_bounded_idxs(data, scales, upper_bound=upper_bound)

        if upper_start_idx is None or lower_end_idx is None:
            return blank_return

        if abs(upper_start_idx - lower_end_idx) <= 1:
            if start_box_cnt < lower_bound:
                return blank_return
            
            if end_box_cnt > upper_bound:
                return blank_return
            
        if start_box_cnt == end_box_cnt:
            return min(lower_start_idx, upper_start_idx), \
                max(lower_end_idx, upper_end_idx), \
                start_box_cnt, start_box_cnt
        
        return upper_start_idx, lower_end_idx, start_box_cnt, end_box_cnt
        
    start_idx = 0
    end_idx = k - 1
    first_box_cnt = get_box_count(data, scales[start_idx])
    final_box_cnt = get_box_count(data, scales[end_idx])
    start_box_cnt = None
    end_box_cnt = None

    while end_idx > start_idx + 1:
        middle = int((end_idx - start_idx) / 2) + start_idx
        box_cnt = get_box_count(data, scales[middle])
            
        # Lower bound search.
        if lower_bound is not None:
            if box_cnt > lower_bound:
                start_idx = middle
                start_box_cnt = box_cnt
            else:
                end_idx = middle
                end_box_cnt = box_cnt

            if box_cnt == lower_bound:
                end_box_cnt = box_cnt

        # Upper bound search.
        # NOTE: Elif condition used to assist numba with int vs. int? comparison.
        elif upper_bound is not None:
            if box_cnt < upper_bound:
                end_idx = middle
                end_box_cnt = box_cnt
            else:
                start_idx = middle
                start_box_cnt = box_cnt

            if box_cnt == upper_bound:
                start_box_cnt = box_cnt

    if lower_bound is not None:
        
        first_box_cnt = get_box_count(data, scales[0])
        
        if end_idx == 1:
            if first_box_cnt == lower_bound:
                return 0, 0, first_box_cnt, first_box_cnt
            if end_box_cnt >= lower_bound:
                return 0, 1, first_box_cnt, end_box_cnt
            else:
                return blank_return
            
        if end_idx == k - 1:
            final_box_cnt = get_box_count(data, scales[k - 1])

            if final_box_cnt >= lower_bound:
                return 0, k - 1, first_box_cnt, final_box_cnt
            else:
                return 0, k - 2, first_box_cnt, start_box_cnt
            
        else:
            if end_box_cnt < lower_bound:
                end_idx -= 1
                end_box_cnt = get_box_count(data, scales[end_idx])

            return 0, end_idx, first_box_cnt, end_box_cnt

    # NOTE: Assists numba with int vs. int? comparison.
    elif upper_bound is not None:
        
        final_box_cnt = get_box_count(data, scales[k - 1])
        
        if start_idx == k - 2:
            if final_box_cnt == upper_bound:
                return k - 1, k - 1, final_box_cnt, final_box_cnt
            if start_box_cnt <= upper_bound:
                return k - 2, k - 1, start_box_cnt, final_box_cnt
            else:
                return blank_return

        if start_idx == 0:
            first_box_cnt = get_box_count(data, scales[0])
            
            if first_box_cnt <= upper_bound:
                return 0, k - 1, first_box_cnt, final_box_cnt
            else:
                return 1, k - 1, end_box_cnt, final_box_cnt
            
        else:
            if start_box_cnt > upper_bound:
                start_idx += 1
                start_box_cnt = get_box_count(data, scales[start_idx])
                
            return start_idx, k - 1, start_box_cnt, final_box_cnt
        
    else:
        return blank_return


def is_nearly_monotonic(
    y: np.ndarray,
    *,
    direction: str = "decreasing",
    tolerance: float = 0.02) -> bool:
    """
    Check whether y is approximately monotonic, allowing small local violations.

    Parameters
    ----------
    y:
        One-dimensional data.
    direction:
        Either "decreasing" or "increasing".
    tolerance:
        Maximum allowed fraction of pairwise steps violating monotonicity.

    Returns
    -------
    bool
        True if the sequence is mostly monotonic.
    """
    y = np.asarray(y, dtype=np.float64)
    dy = np.diff(y)

    if direction == "decreasing":
        violations = dy > 0
    elif direction == "increasing":
        violations = dy < 0
    else:
        raise ValueError("direction must be 'decreasing' or 'increasing'.")

    return bool(np.mean(violations) <= tolerance)


def compute_deshmukh_slope_estimate(
        trimmed_scales: np.ndarray,
        trimmed_H: np.ndarray,
        deshmukh_reg_proportion: float,
        error_eps: float = 1e-6) -> float | None:
    
    """
    Computes the Deshmukh estimate of the scaling curve slope using Numba implementation.
    
    Computes the distribution of slopes over all possible scaling curve line
    segments of sufficient length. The required segment length is defined as
    greater than or equal to the Deshmukh regularization proportion of the total
    scaling curve length. The final slope estimate is the mean of the resulting
    distribution.

    For more information on the method, see [1]

    Arguments
    ----------


    References
    ----------

    ..[1] V. Deshmukh, E. Bradley, J. Garland, and J. D. Meiss, “Toward automated
        extraction and characterization of scaling regions in dynamical systems,” Chaos
        31, 123102 (2021).
    
    """
    
    # Get length of scaling curve
    N = trimmed_scales.shape[0]

    if N == 0 or N == 1:
        return

    # Compute the minimum length of a line segment to be regressed.
    n = int(max(deshmukh_reg_proportion * N, 1))

    # Compute the total number of line segments.
    S = int((N - n - 1) * (N - n) / 2)
    
    # Initialise arrays storing required information.
    slopes = np.zeros(shape=S)
    log_errors = np.zeros(shape=S)
    log_lengths = np.zeros(shape=S)
    i = 0

    # Iterate over all line segments.
    for lhs in range(N - n):
        for rhs in range(lhs + n + 1, N):
            
            # Get the segment x-axis length.
            reg_length = rhs - lhs

            # Get the segment x and y values.
            xs = trimmed_scales[lhs:rhs]
            ys = trimmed_H[lhs:rhs]

            # Perform linear regression.
            #m, c, _, _, _ = stats.linregress(
                #xs, ys)
            m, c = simple_linear_regression(xs, ys)
            
            # Grab the slope, fitting error and segment length
            slopes[i] = m
            log_errors[i] = 0.5 * (np.log(max(((ys - m * xs - c)**2).sum(), error_eps))
                                   - np.log(reg_length))
            tot_x = abs(xs[-1] - xs[0])
            log_lengths[i] = 0.5 * np.log(1 + m**2) + np.log(tot_x)
            i += 1

    # Weigh the slopes by the line length divided by the fitting error.
    log_weights = log_lengths - log_errors
    weights = np.exp(log_weights)

    # Normalise weights.
    weights = weights / weights.sum()

    # Compute weighted slopes.
    samples = slopes * weights

    # Return the weighted average.
    return samples.sum()


def simple_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fast 1D OLS line fitter.
    Fits: y = mx + c
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    
    denominator = (n * sum_xx) - (sum_x * sum_x)
    
    # Safety check for flat horizontal scaling regions
    if denominator == 0:
        return 0.0, 0.0 # Slope, Intercept fallback
        
    slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
    intercept = (sum_y - (slope * sum_x)) / n
    
    return slope, intercept

# ---------------------------------------------------------------------
# If this is the private Numba copy, compile this module.
# ---------------------------------------------------------------------

if not globals().get("__skip_jit_compile__", False):
    # Compilation options are captured when jit_module() runs, so resolve one
    # consistent settings snapshot before passing them to Numba.
    current_settings = settings.current
    jit_module(
        nopython=True,
        cache=current_settings.numba_caching,
        boundscheck=current_settings.numba_boundschecking,
        error_model=current_settings.numba_error_model,
        fastmath=current_settings.numba_fastmath
    )

# ---------------------------------------------------------------------
# Public module bootstrapping
# ---------------------------------------------------------------------

if not globals().get("__internal_copy__", False):
   install_numba_funcs(sys.modules[__name__])

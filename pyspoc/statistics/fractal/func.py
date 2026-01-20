import numpy as np
import piecewise_regression
import warnings

from scipy.interpolate import interp1d
from scipy import stats
from typing import Optional

from .utils import find_ranges_pct, find_ranges_ls, get_reflex


def _get_boxes(points, scales) -> np.ndarray:
    """
    Compute unique grid boxes for points scaled by `scale`.

    Parameters:
        points (ndarray): Array of points (n, 2).
        scale (float): Scale factor.

    Returns:
        list: Unique grid boxes.
    """
    points = np.expand_dims(points, axis=2)
    results = (points / scales).astype(int)
    unique_counts: np.ndarray = np.empty(shape=results.shape[2], dtype=np.int32)
    
    for k in range(unique_counts.shape[0]):
        unique_count: int = np.unique(results[:,:,k], axis=0).shape[0]
        unique_counts[k] = unique_count

    return unique_counts


def _get_generalized(points, scale):
    """
    Compute generalized grid boxes and their weights.

    Parameters:
        points (ndarray): Array of points (n, 2).
        scale (float): Scale factor.

    Returns:
        dict: Grid boxes and their weights.
    """
    result = {}
    for i in range(len(points) - 1):
        segment = points[i : i + 2]
        boxes = [tuple(s) for s in segment]

        # Traverse x direction
        x_sorted = np.sort(segment[:, 0])
        x_indices = np.arange(
            np.ceil(x_sorted[0] / scale), np.floor(x_sorted[1] / scale) + 1
        )
        if len(x_indices):
            y_interp = interp1d(segment[:, 0], segment[:, 1], fill_value="extrapolate")(
                x_indices * scale
            )
            boxes.extend(zip(x_indices * scale, y_interp))

        # Traverse y direction
        y_sorted = np.sort(segment[:, 1])
        y_indices = np.arange(
            np.ceil(y_sorted[0] / scale), np.floor(y_sorted[1] / scale) + 1
        )
        if len(y_indices):
            x_interp = interp1d(segment[:, 1], segment[:, 0], fill_value="extrapolate")(
                y_indices * scale
            )
            boxes.extend(zip(x_interp, y_indices * scale))

        boxes = sorted(set(boxes), key=lambda x: (x[0], x[1]))

        # Compute weights
        for j in range(len(boxes) - 1):
            mid_x = (boxes[j][0] + boxes[j + 1][0]) * 0.5 / scale
            mid_y = (boxes[j][1] + boxes[j + 1][1]) * 0.5 / scale
            weight = np.linalg.norm(np.array(boxes[j]) - np.array(boxes[j + 1]))

            key = (int(mid_x), int(mid_y))
            result[key] = result.get(key, 0) + weight

    return result


def box_counting(
    points, 
    scales, 
    max_p=None,
    min_r_sq=None,
    return_data=False,
    suppress_warnings=False):
    """
    Compute fractal dimension using the box counting method.

    Parameters:
        points (ndarray): Array of points (n, 2).
        scales (list): List of scales.
        method (str): Method to compute boxes ("original", "oversample", "exact").
        oversample_rate (int): Oversampling rate for "oversample" method.
        return_boxes (bool): Whether to return boxes at each scale.

    Returns:
        dict: Results with fractal dimension and fit statistics.
    """
    
    is_r_sq_required = min_r_sq is not None
    return_inner_data = is_r_sq_required or return_data
    boxes = _get_boxes(points, scales)
    extracted = _extract_fractal_dimension(scales, boxes, return_data=return_inner_data)
    fd = extracted["fd"]
    result = {
        "fd": fd
    }
    
    if return_inner_data:
        boxes = extracted["filtered_boxes"]
        scales = extracted["filtered_scales"]
        
    if return_data:
        result["boxes"] = boxes
        result["scales"] = scales
        result["const"] = extracted["const"]

    if is_r_sq_required:
        x = -np.log(scales)
        y = np.log(boxes)
        y_hat = extracted["const"] + fd * x
        rss = ((y - y_hat) ** 2).sum()
        tss = y.var() * y.shape[0]
        result["rss"] = rss
        result["tss"] = tss
        r_sq = 1 - rss / tss
        result["r_squared"] = r_sq

        if r_sq < min_r_sq and not suppress_warnings:
            warnings.warn(f"R-squared value of {r_sq} was below minimum requirement of {min_r_sq} specified.")

    if max_p is not None:
        p_value = extracted["p_value"]
        result["p_value"] = p_value

        if p_value > max_p and not suppress_warnings:
            warnings.warn(f"R-squared value of {r_sq} was below minimum requirement of {min_r_sq} specified.")

    return result


def box_counting_generalized(points, scales, q=2, return_boxes=False):
    """
    Compute generalized fractal dimension using exact box counting method.

    Parameters:
        points (ndarray): Array of points (n, 2).
        scales (list): List of scales.
        q (float): Order of the generalized fractal dimension.
        return_boxes (bool): Whether to return boxes at each scale.

    Returns:
        dict: Results with fractal dimension and fit statistics.
    """
    results = []
    for scale in scales:
        grid_weights = _get_generalized(points, scale)
        weights = np.array(list(grid_weights.values()))
        weights = weights / np.sum(weights)

        if q == 1:
            result = np.sum(weights * np.log(weights))
        elif q == 0:
            result = -np.log(np.sum(weights > 0))
        else:
            result = np.log(np.sum(np.power(weights, q))) / (q - 1)

        results.append([scale, result])

    results = np.array(results)

    slope, _, r_value, p_value, _ = stats.linregress(
        np.log(results[:, 0]), results[:, 1]
    )
    fd = slope

    output = {
        "fd": fd,
        "r_squared": r_value**2,
        "p_value": p_value,
    }
    if return_boxes:
        output["boxes"] = results

    return output


def _extract_fractal_dimension(scales, boxes, return_data=False) -> dict:
    neg_log_scales = -np.log(scales)
    log_boxes = np.log(boxes)
    log_boxes_diff = np.diff(log_boxes)
    has_starting_elbow = log_boxes_diff[0].round(4) == 0
    max_breakpoints = 2 if has_starting_elbow else 1
    ms = piecewise_regression.ModelSelection(neg_log_scales, log_boxes, max_breakpoints=max_breakpoints)
    curr_summary = ms.model_summaries[0]
    curr_bic = curr_summary["bic"]
    fd = curr_summary["estimates"]["alpha1"]    
    lower_breakpoint = neg_log_scales.min() - 1
    upper_breakpoint = neg_log_scales.max() + 1
    result = {
        "fd": fd        
    }

    for summary in ms.model_summaries[1:]:
        if not summary["converged"]:
            continue

        bic = summary["bic"]
        
        if bic < (curr_bic - 2 * abs(curr_bic)):
            curr_bic = bic
            estimates = summary["estimates"]
            result["fd"] = estimates["alpha2"]["estimate"] if has_starting_elbow else estimates["alpha1"]["estimates"]
            result["p_value"] = estimates["alpha2"]["p_t"] if has_starting_elbow else estimates["alpha1"]["p_t"]
            
            if return_data:
                result["const"] = estimates["const"]["estimate"]

                if has_starting_elbow:
                    lower_breakpoint = estimates["breakpoint1"]["estimate"]
                    upper_breakpoint = estimates["breakpoint2"]["estimate"]
                else:
                    upper_breakpoint = estimates["breakpoint1"]["estimate"]

    if return_data:
        scale_filter = (neg_log_scales > lower_breakpoint) & (neg_log_scales < upper_breakpoint)
        result["filtered_scales"] = scales[scale_filter]
        result["filtered_boxes"] = boxes[scale_filter]

    return result

def temporal_sampling(
    points, min_step=1, max_step=2, q=1, start_index=0, return_boxes=False
):
    """
    Compute fractal dimension using the temporal sampling method.

    Parameters:
        points (ndarray): Array of points (n, 2).
        min_step (int): Minimum step size.
        max_step (int): Maximum step size.
        q (float): Order of the generalized fractal dimension.
        start_index (int): Starting index for sampling.
        return_boxes (bool): Whether to return computed points.

    Returns:
        dict: Results with fractal dimension and fit statistics.
    """
    results = []
    step_sizes = range(min_step, max_step + 1)

    for step in step_sizes:
        indices = range(start_index, len(points), step)
        sampled_points = points[indices]
        distances = np.linalg.norm(np.diff(sampled_points, axis=0), axis=1)
        avg_distance = np.mean(np.power(distances, q))
        results.append(avg_distance)

    results = np.array([step_sizes, results]).T

    slope, _, r_value, p_value, _ = stats.linregress(
        np.log(results[:, 0]), np.log(results[:, 1]) / q
    )
    fd = 1 + (1 - slope) * min(2, 1 / slope)

    output = {
        "fd": fd,
        "r_squared": r_value**2,
        "p_value": p_value,
    }
    if return_boxes:
        output["boxes"] = results

    return output


def corr_sum(points, scales, min_gap=1, return_boxes=False):
    """
    Compute the correlation sum for fractal dimension estimation.

    Parameters:
        points (ndarray): Array of points (n, 2).
        scales (list): List of scales.
        min_gap (int): Minimum separation of points.
        return_boxes (bool): Whether to return computed points.

    Returns:
        dict: Results with fractal dimension and fit statistics.
    """
    dx = np.subtract.outer(points[:, 0], points[:, 0])
    dy = np.subtract.outer(points[:, 1], points[:, 1])
    distances = np.sqrt(dx**2 + dy**2)

    for i in range(min_gap):
        np.fill_diagonal(distances[i:], np.nan)
        np.fill_diagonal(distances[:, i:], np.nan)

    results = []
    N = distances.shape[0]

    for scale in scales:
        count = np.sum(distances < scale) / ((N - min_gap) * (N + 1 - min_gap))
        results.append([scale, count])

    results = np.array(results)

    slope, _, r_value, p_value, _ = stats.linregress(
        np.log(results[:, 0]), np.log(results[:, 1])
    )
    fd = slope

    output = {
        "fd": fd,
        "r_squared": r_value**2,
        "p_value": p_value,
    }
    if return_boxes:
        output["boxes"] = results

    return output


def corr_sum_takens(points, min_gap=1, scale=None):
    """
    Compute the correlation sum using Takens' method.

    Parameters:
        points (ndarray): Array of points (n, 2).
        min_gap (int): Minimum separation of points.
        scale (float): Threshold scale for distances.

    Returns:
        float: Fractal dimension using Takens' method.
    """
    dx = np.subtract.outer(points[:, 0], points[:, 0])
    dy = np.subtract.outer(points[:, 1], points[:, 1])
    distances = np.sqrt(dx**2 + dy**2)

    for i in range(min_gap):
        np.fill_diagonal(distances[i:], np.nan)
        np.fill_diagonal(distances[:, i:], np.nan)

    if scale is None:
        # scale = np.std(points, axis=0).mean() / 4
        scale = np.sqrt(np.mean(np.var(points, axis=0))) / 4

    valid_distances = distances[(distances < scale) & (distances > 0)]
    count = len(valid_distances)
    fd = -(count - 1) / np.sum(np.log(valid_distances / scale))

    return fd


def find_elbow_scale(
    xs,
    init_scales=None,
    method="threshold",
    pct_th=0.01,
    init_scale_config=None) -> dict:
    """
    Compute the elbow scale.

    Parameters:
        xs (ndarray): Array of points (n, 2).
        init_scales (list or None): Initial scales for computation. Defaults to None.
        method (str): Method to find elbow ("threshold" or "regression").
            Defaults to "threshold".
        pct_th (float): Threshold percentage for the "threshold" method.
            Defaults to 0.01.
        return_boxes (bool): Whether to return computed boxes. Defaults to False.
        init_scale_config (dict or None): Configuration for scale initialization.
            Keys can include:
            - "ref_scale": Reference scale for initialization. Defaults to mean distance.
            - "step_size": Step size for logarithmic scaling.
            - "num_scales": Total number of scales.
            - "num_scales_right": Number of scales to the right of reference scale.

    Returns:
        dict: A dictionary containing the results with indices, scales, and
            the minimum index.
    """
    if init_scale_config is None:
        init_scale_config = {}

    # Default scale initialization if not provided
    if init_scales is None:
        init_scales = get_default_init_scale(xs, init_scale_config)

    # Perform box counting
    boxes = _get_boxes(xs, init_scales)

    # Select method for range computation
    if method == "threshold":
        xt, st, idx0, idx1 = find_ranges_pct(boxes, init_scales, pct=pct_th)
    elif method == "regression":
        xt, st, idx0, idx1 = find_ranges_ls(boxes, init_scales)
    else:
        raise ValueError(f"Unknown method: {method}")

    rr = np.array([st, xt]).T
    
    # Compute log-transformed values for reflex detection
    log_rr = np.log10(rr)
    log_rr[:, 0] = -log_rr[:, 0]

    # Find reflex point
    reflex_result = get_reflex(log_rr)
    reflex_idx = reflex_result["min_idx"]

    # Prepare results
    result = {
        "idx0": idx0,
        "idx1": idx1,
        "min_idx": idx0 + reflex_idx,        
        "elbow": st[reflex_idx],
        "elbow_scales": st[reflex_idx:]
    }

    return result


def get_default_init_scale(xs: np.ndarray, 
                           init_scale_config: Optional[dict] = None) -> np.ndarray:

    # Default scale initialization if not provided
    if not init_scale_config:
        scale_config = dict()
    else:
        scale_config = init_scale_config

    eps = 1e-20
    mean_distance = np.mean(np.linalg.norm(xs[:-1, :] - xs[1:, :], axis=1)) + eps
    ref_scale = scale_config.get("ref_scale", mean_distance)
    step_size = scale_config.get("step_size", 0.025)
    num_scales = scale_config.get("num_scales", 400)
    num_scales_right = scale_config.get("num_scales_right", 200)
    num_scales_left = num_scales - num_scales_right

    init_scales = ref_scale * np.power(
        10,
        step_size * np.linspace(num_scales_left, -num_scales_right, num_scales + 1),
    )

    return init_scales
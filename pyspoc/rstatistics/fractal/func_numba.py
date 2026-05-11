import numpy as np

from typing import Optional
from numba import njit

@njit
def _sort_data(data: np.ndarray,
               force_copy: bool = True) -> np.ndarray:
    
    if force_copy:
        data = data.copy()

    p = data.shape[1]

    for j in range(p - 1,-1,-1):
    
        if j < p - 1:
            data = data[data[:,-j].argsort(kind="mergesort")]
        else:
            data = data[data[:,-j].argsort()]

    return data

@njit
def _get_renyi_entropy(q: float,
                       data: np.ndarray,
                       scales: np.ndarray,
                       shannon_entropy_tol: float = 1e-6) -> np.ndarray:
    
    n_scales = scales.shape[0]
    n = data.shape[0]
    p = data.shape[1]
    H = np.empty(shape=n_scales, dtype=np.float64)

    for k in range(n_scales):
        cnts = np.zeros((n, p+1), dtype=np.int32)
        s = scales[k]
        box_ids = np.floor(data / s).astype(np.int32)
        box_ids = _sort_data(box_ids)
        prev_box_id = np.zeros(shape=p).astype(np.int32)
        j = 0

        for i in range(n):
            box_id = box_ids[i]
            
            if np.any(prev_box_id != box_id):
                j += 1
                prev_box_id = box_id
                cnts[j,:-1] = box_id
            
            cnts[j,-1] += 1
                
        cnts = cnts[cnts[:,-1] > 0]

        if q == 0:
            H[k] = np.log(cnts.shape[0])
            continue

        probs = cnts[:,-1] / cnts[:,-1].sum()

        if abs(q-1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H

@njit
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

@njit
def _get_default_init_scale(xs: np.ndarray,
                            step_size: float = 0.025,
                            num_scales: int = 400,
                            num_scales_right: int = 200,
                            ref_scale: Optional[float] = None) -> np.ndarray:

    if not ref_scale:
        eps = 1e-20
        ref_scale = np.mean(np.sqrt(np.sum((xs[:-1, :] - xs[1:, :]) ** 2, axis=1))) + eps

    num_scales_left = num_scales - num_scales_right
    init_scales = ref_scale * np.power(
       10,
       step_size * np.linspace(num_scales_left, -num_scales_right, num_scales + 1),
    )
    return init_scales

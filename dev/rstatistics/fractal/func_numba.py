# %%

# %%
import pyspoc.rstatistics.fractal._funcs_py as fpy
import pyspoc.rstatistics.fractal._funcs_numba as fnb
import pyspoc.rstatistics.fractal.renyi as renyi
import pyspoc.data.generators.fractal as fgen
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

data = np.random.standard_normal(size=(10000,10))
scales = fpy.get_adaptive_scales(data, method="datseries", k=1000)
scales = scales[::-1]
#data = fgen.mandelbrot()
# %%
renyi_entr =  renyi.RenyiEntropy(scale_length=500)
renyi_entr.compute(data)
#fpy._sort_data(data, debug_numba="raise")

# %%
#reload(fnb)
densities = renyi._get_renyi_entropy(0, data, scales)
neg_log_scales = -np.log(scales)
slope, adj_r2 = fpy.compute_ols_results(neg_log_scales, densities)
# %%
elbow_idxs = fpy.find_elbow_idx(
    neg_log_scales,
    densities,
    direction="increasing",
    polyorder=3,
    monotonic_tolerance=0.1)

if elbow_idxs:
    elbow_filter = slice(*elbow_idxs)
    filtered_scales = neg_log_scales[elbow_filter]
    filtered_densities = densities[elbow_filter]

plt.plot(neg_log_scales, densities)
plt.show()
plt.plot(filtered_scales, filtered_densities)
plt.show()
#%%

slope_samples = _get_deshmukh_slope_dist(
    filtered_scales,
    filtered_densities,
    0.25)

#%%
fd = _get_deshmukh_dist_statistic(slope_samples, "mode")
print(fd)

# %%
scales = fpy.get_datseries_scales(data, k=1000)
# %%

fnb._get_bounded_idxs(data, scales, 1, 100, debug_numba=["warn"])
#%%
renyi._get_renyi_entropy(0, data, scales)
# %%
reload(fnb)
#%%
densities = renyi._get_renyi_entropy(0, data, scales)

# %%
import inspect

sig = inspect.signature(fnb._install_numba_funcs)
sig
#%%
fnb._get_bounded_idxs(data, scales, 1, 100)
#%%
scales = fpy.get_adaptive_scales(data, method="datseries", k=1000)
#%%
from typing import Literal, Iterable
from scipy.stats import gaussian_kde
from functools import cache

import pyspoc._numba as ph

def _get_deshmukh_slope_dist(
        neg_log_scales: np.ndarray,
        densities: np.ndarray,
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
    N = neg_log_scales.shape[0]

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
            xs = neg_log_scales[lhs:rhs]
            ys = densities[lhs:rhs]

            # Perform linear regression.
            #m, c, _, _, _ = stats.linregress(
                #xs, ys)
            m, c = _get_ols_parameters(xs, ys)
            
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
    samples = slopes * weights * slopes.shape[0]

    # Return the weighted average.
    return samples


def _get_ols_parameters(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
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


def _get_deshmukh_dist_statistic(
        samples: np.ndarray,
        statistic: Literal["mean", "median", "mode", "modalmean", "all"] | \
            Iterable[Literal["mean", "median", "mode", "modalmean"]]) -> np.ndarray:
        
    # Expand "all" option.
    if statistic == "all":
        statistics = ["mean", "median", "mode", "modalmean"]

    # Or ensure list object if singleton provided.
    elif isinstance(statistic, str):
        statistics = [statistic]

    n_stats = len(statistics)
    fds = np.empty(shape=n_stats, dtype=np.float64)
    
    for i, s in enumerate(statistics):
   
        match s:

            case "mean":
                fds[i] = np.mean(samples)

            case "median":
                fds[i] = np.median(samples)
            
            case opt if opt in ["mode", "modalmean"]:
                sample_axis, sample_dens = _get_sample_kde_distribution(samples)

                if opt == "mode":
                    fds[i] = sample_axis[np.argmax(sample_dens)]
                
                if opt == "modalmean":
                    fds[i] = _get_modal_mean(sample_axis, sample_dens)

    return fds

        
def _get_sample_kde_distribution(
        samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    samples_key = ph._get_unordered_array_key(samples)
    results = _get_sample_kde_distribution_from_key(samples_key)
    return results


@cache
def _get_sample_kde_distribution_from_key(
        samples_key: tuple[str, int, bytes]) -> tuple[np.ndarray, np.ndarray]:
    
    samples = ph._get_array_from_key(samples_key)
    
    sample_axis = np.linspace(
        samples.min(),
        samples.max(),
        samples.shape[0] * 2)
    
    kernel = gaussian_kde(samples)
    sample_dens = kernel(sample_axis)
    return sample_axis, sample_dens


def _get_modal_mean(
        sample_axis: np.ndarray,
        sample_dens: np.ndarray) -> float:
    
    d_sample_dens = np.diff(sample_dens)
    pve_grad = d_sample_dens > 0
    nve_grad = np.hstack([d_sample_dens[1:] < 0, False])
    mode_mask = np.hstack([False, pve_grad & nve_grad])

    if not np.all(mode_mask):
        return sample_axis[np.argmax(sample_dens)]

    weights = sample_dens[mode_mask]
    modes = sample_axis[mode_mask]
    weighted_modes = modes * weights / weights.sum()
    return weighted_modes.sum()

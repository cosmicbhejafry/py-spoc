# %%
from pyspoc.rstatistics.fractal import (
    _funcs_py as fpy,
    _funcs_numba as fnb,
    renyi,
)

from pyspoc.data.generators import (
    fractal as pgf
)

import numpy as np

#data = pgf.mandelbrot()
data = np.random.random(size=(10000,50))
#%%
scales = fpy.get_adaptive_scales(
    data,
    k=200,
    method="datseries")

scales = scales[::-1]
# %%
       
# Estimate density at each box scale.
densities = renyi._get_renyi_entropy_numba(q=0, data=data, scales=scales)
# Log transform for linear relationship.
neg_log_scales = -np.log(scales)
# %%
import matplotlib.pyplot as plt

plt.plot(neg_log_scales, densities)
plt.show()

# %%
 # Perform OLS regression of the scaling curve as starting point.
adj_r2_thresh = 1.0
slope, adj_r2 = fpy.compute_ols_results(neg_log_scales, densities)

if adj_r2 < adj_r2_thresh:

    elbow_idxs = fpy.find_elbow_idx(
        neg_log_scales,
        densities,
        direction="increasing",
        polyorder=3,
        monotonic_tolerance=0.1)

    if elbow_idxs:
        elbow_filter = slice(*elbow_idxs)
        neg_log_scales = neg_log_scales[elbow_filter]
        densities = densities[elbow_filter]

    slope, adj_r2 = fpy.compute_ols_results(neg_log_scales, densities)

#%%
plt.plot(neg_log_scales, densities)
#%%
fd = fnb.compute_deshmukh_slope_estimate(
    neg_log_scales,
    densities,
    0.25)

if fd:
    # Return fractal dimension estimate if available.
    print(fd)

# Return OLS slope if Deshmukh method fails.
if slope:
    print(slope)

#%%
re = renyi.RenyiEntropy(q=0, minimum_scaling_region=0.1, slope_estimation_method="hybrid")
result = re.compute(data)
print(result)
# %%

result = renyi.RenyiEntropy(q=0, slope_estimation_method="ols").compute(data)
print(result)

result = renyi.RenyiEntropy(q=0, slope_estimation_method="deshmukh").compute(data)
print(result)
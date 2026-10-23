#%%
import numpy as np
import time

import pyspoc.statistics.causal as causal
from pyspoc.statistics.base import Statistic

data = np.random.normal(size=(1000,50))

def time_run(stat: Statistic, data: np.ndarray):
    start = time.perf_counter()
    result = stat._compute(data)
    elapsed = time.perf_counter() - start
    print("Time:", elapsed)
    return result, elapsed

# %% CDS
# Rejected - slow (42s)
cds = causal.ConditionalDistributionSimilarity()
result, elapsed = time_run(cds, data)
result

#%% RECI
# Accepted - average (4s)
reci = causal.RegressionErrorCausalInference()
result, elapsed = time_run(reci, data)
result
#%% IGCI
# Accepted - average (4s)
igci = causal.InformationGeometricConditionalIndependence()
result, elapsed = time_run(igci, data)
result
#%% ANM
# Rejected - very slow (>40m)
# Likely the GP fitting - this will need to optimised.
anm = causal.AdditiveNoiseModel()
result, elapsed = time_run(anm, data)
result
#%% GP testing
# Bottleneck is definitely GP fitting. Roughly 2.2s per pair.
# For a submatrix with 50 variates, thats 50 * 51 / 2 = 1275
# runs, which comes out at almost 47 minutes. Not viable.
# Needs heavy parallelisation, kernel memoisation, and perhaps
# even shared GP parameters across pairs.
from sklearn.gaussian_process import GaussianProcessRegressor

start = time.perf_counter()
x_2d = data[:,0].reshape(-1,1)
y = data[:,1]
gp = GaussianProcessRegressor(random_state=0).fit(x_2d,y)
gp.predict(x_2d).reshape(-1,1)
elapsed = time.perf_counter() - start
print(elapsed)
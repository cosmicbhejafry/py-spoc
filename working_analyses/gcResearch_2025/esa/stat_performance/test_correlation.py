#%%
import numpy as np
import time

import pyspoc.statistics.association.covariance as covariance
from pyspoc.statistics.base import Statistic

data = np.random.normal(size=(1000,50))

def time_run(stat: Statistic, data: np.ndarray):
    start = time.perf_counter()
    result = stat._compute(data)
    elapsed = time.perf_counter() - start
    print("Time:", elapsed)
    return result, elapsed

#%% Elliptic Envelope Covariance
# Accepted - average (2s)
ee = covariance.Covariance("EllipticEnvelope")
result, elapsed = time_run(ee, data)
result
#%% Empirical Covariance
# Accepted - v fast (0.002s)
emp_cov = covariance.Covariance("EmpiricalCovariance")
result, elapsed = time_run(emp_cov, data)
result
#%% Graphical Lasso Cross-Validated Covariance
# Accepted - fast (0.4s)
glcv = covariance.Covariance("GraphicalLassoCV")
result, elapsed = time_run(glcv, data)
result
#%% Minimum Determinant Covariance
# Rejected - average (2s), but has high failure rate on multi-modal data
mcd = covariance.Covariance("MinCovDet")
result, elapsed = time_run(mcd, data)
result
#%% Oracle Approximating Shrinkage Covariance
# Accepted - v fast (0.002s)
oas = covariance.Covariance("OAS")
result, elapsed = time_run(oas, data)
result
#%% Ledoit-Wolf Shrinkage Covariance
# Accepted - v fast (0.008s)
lw = covariance.Covariance("LedoitWolf")
result, elapsed = time_run(lw, data)
result
#%% Shrunk Covariance
# Accepted - v fast (0.003s)
sc = covariance.Covariance("ShrunkCovariance")
result, elapsed = time_run(sc, data)
result

#%% Kendall Tau Ordinal Correlation
# Accepted - fast (0.43s)
sc = covariance.KendallTau(nan_policy="omit")
result, elapsed = time_run(sc, data)
result
#%% Spearman R Correlation
# Accepted - v fast (0.008s)
sr = covariance.SpearmanR(squared=False)
result, elapsed = time_run(sr, data)
result

#%% Power Envelope Correlation
# Accepted - v fast (0.018s)
pec = covariance.PowerEnvelopeCorrelation(orth=True)
result, elapsed = time_run(pec, data)
result
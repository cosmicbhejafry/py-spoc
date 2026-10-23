#%%
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from itertools import product

import pyspoc.statistics.errors.regressors as ereg
from pyspoc.statistics.base import Statistic

data = np.random.normal(size=(1000,50))

def time_run(stat: Statistic, data: np.ndarray):
    start = time.perf_counter()
    result = stat._compute(data)
    elapsed = time.perf_counter() - start
    return result, elapsed

#%% All linear regressors
regressors = ereg.LinearRegressorError.available_estimators()
errors = ereg.LinearRegressorError.available_errors()
run_results = pd.DataFrame(
    data = np.full(shape=(len(regressors), len(errors)),
                   fill_value=np.nan),
    index = pd.Index(regressors, name="Regressor"),
    columns = pd.Index(errors, name="Error"),
    dtype=pd.Float32Dtype()
)

regressor_error_grid = list(product(regressors, errors))
#%%
progress = tqdm(regressor_error_grid)

for regressor, error in progress:
    progress.set_description(
        f"Regressor {regressor} running on error {error}")

    try:
        reg = ereg.LinearRegressorError(
            model=regressor,
            error=error,
            force_positive_tweedie=True)
        result, elapsed = time_run(reg, data)

        if result is None:
            tqdm.write(regressor, "with error", error, "returned no results")
        else:
            run_results.loc[regressor, error] = elapsed
    except Exception as e:
        print(e)
##%
run_results

# NOTE:
# Reject (too slow):
# LassoCV,
# ElasticNetCV
# QuantileRegressor
# RANSACRegressor
# TheilSenRegressor
# GammaRegressor
# PoissonRegressor
# InverseGaussianRegressor

# Reject (other reasons):
# OrthogonalMatchingPursuitCV (needs multiple regressors)
#
# Iffy: LarsCV, LassoLarsCV (but they are CV methods after all)
#%% GPRegressor
# Rejected - glacial (>5m)
reg = ereg.GPRegressorError()
result, elapsed = time_run(reg, data)
elapsed
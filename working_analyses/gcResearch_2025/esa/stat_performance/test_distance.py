#%%
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from itertools import product

import pyspoc.statistics.distance as dist
from pyspoc.statistics.base import Statistic

data = np.random.normal(size=(1000,50))

def time_run(stat: Statistic, data: np.ndarray):
    start = time.perf_counter()
    result = stat._compute(data)
    elapsed = time.perf_counter() - start
    return result, elapsed

#%% All standard distance metrics
metrics = dist.PairwiseDistance.available_distances()
exclude_metrics = [
    "haversine", # Only valid on 2d dimensional inputs.
]
metrics = [m for m in metrics if m not in exclude_metrics]
stats = (dist.PairwiseDistance,)

run_results = pd.DataFrame(
    data = np.full(shape=(len(metrics), len(stats)),
                   fill_value=np.nan),
    index = pd.Index(metrics, name="Regressor"),
    columns = pd.Index([s.__qualname__ for s in stats],
                       name="Error"),
    dtype=pd.Float32Dtype()
)

stat_metric_grid = list(product(stats, metrics))
#%%
progress = tqdm(stat_metric_grid)

for stat_cls, metric in progress:
    stat_name = stat_cls.__qualname__
    progress.set_description(
        f"Statistic {stat_name} running on metric {metric}")

    try:
        stat = stat_cls(metric=metric)
        result, elapsed = time_run(stat, data)

        if result is None:
            tqdm.write(stat_name, "with metric", metric, "returned no results")
        else:
            run_results.loc[metric, stat_name] = elapsed
    except Exception as e:
        #tqdm.write(str(e))
        raise e

# Rejected (too slow - these are all implemented by the hyppo library, which was slow for HSIC too):
# DistanceCorrelation
# MultiscaleGraphCorrelation
# HellerHellerGorfine

# Accepted:
# PairwiseDistance vfast (0.2s average)

#%% GWT Distance
# Accepted: vfast (0.14s)
gwt = dist.GromovWassersteinTau()
result, elapsed = time_run(gwt, data)
print(elapsed)
#%% Hilbert-Schmidt Independence Criterion
# Accepted: fast (1s) - need to test different kernels though
hsic = dist.HilbertSchmidtIndependenceCriterion()
result, elapsed = time_run(hsic, data)
print(elapsed)
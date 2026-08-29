#%%
from time import perf_counter

import numpy as np

from pyspoc.statistics.distance.hsic import HilbertSchmidtIndependenceCriterion
from pyspoc.profiling import profile_component_cpu
from pyspoc.settings import settings

from hyppo.independence import Hsic

data_size = (10000,20)
rng = np.random.default_rng(0)
data = rng.normal(size=data_size)
# %%
kt = HilbertSchmidtIndependenceCriterion(dim="p", biased=False, downsample=True)

with settings.override(max_worker_threads=8):
    report = profile_component_cpu(
        kt,
        data,
        warmup_runs=1,
        measured_runs=5,
    )

    print(report)
#%%
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    report = profile_component_cpu(
        kt,
        data,
        warmup_runs=1,
        measured_runs=5,
    )

    print(report)

#%%
hsic = Hsic()
p = data.shape[1]
H = np.zeros(shape=(p,p))
start = perf_counter()

for i in range(p):
    for j in range(i, p):
        x,y = data[:,i], data[:,j]
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        result = hsic.statistic(x,y)
        H[i,j] = result

        if i != j:
            H[j,i] = result

elapsed = perf_counter() - start
print(f"hyppo HSIC elapsed time: {elapsed:.6f} seconds")

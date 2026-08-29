#%%
import numpy as np

from pyspoc.statistics.cov import KendallTau
from pyspoc.profiling import profile_component_cpu
from pyspoc.settings import Settings

data_size = (10000,50)
kt = KendallTau(squared=False)
rng = np.random.default_rng(0)
data = rng.normal(size=data_size)

local_settings = Settings()

with local_settings.override(max_worker_threads=1):
    report = profile_component_cpu(
        kt,
        data,
        warmup_runs=1,
        measured_runs=5,
    )

    print(report)

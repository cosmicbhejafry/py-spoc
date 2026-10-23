#%%
import pyspoc
import numpy as np

from pathlib import Path
from importlib import reload
from tqdm import tqdm

from results import (
    plotting as p)

from results.execution import (
    luo_files as luo,
    cleaning as clean,
    pipeline as pipe)

#%%

###############################################
# Main results run.
###############################################

cfg = pyspoc.Config.from_yaml_file(name="luo_run", yaml_file_path="cfg.yml")
sampling_seed = 0
base_data_path = Path("D:/OneDrive/OneDrive - Imperial College London/Data/")
luo_dir = base_data_path / "Luo" / "pyspoc"
luo_results_prefix = "luo"
include_target = False
batch_n = 5000
batch_p = 50
max_batches = 1

luo_datasets = luo.get_annotated_datasets(
    luo_folder=luo_dir,
    refresh=False
)
luo_datasets = list(luo_datasets)

#%%
results, short_results, timings, short_timings = pipe.run_pyspoc_loop(
    cfg=cfg,
    annotated_datasets=[luo_datasets[0]],
    batch_n=batch_n,
    batch_p=batch_p,
    use_batching=True,
    max_batches=max_batches,
    results_path=luo_dir,
    sampling_seed=sampling_seed,
    output_files_prefix=luo_results_prefix,
    save_frequency=1
)

#%%
luo_dataset = luo_datasets[0]
data = luo_dataset.get_data()
dataset_name = luo_dataset.get_name()
dataset = pyspoc.Dataset(data, name=dataset_name)
batches = dataset.get_batches(
    batch_shape=(batch_n, batch_p),
    sampling_type="cyclical",
    seed=sampling_seed,
    max_n_batches=max_batches)
batch = batches[0]
from pyspoc.statistics.causal import RegressionErrorCausalInference

reci = RegressionErrorCausalInference()
reci._compute(batch)

#%%
import sys
from tqdm.auto import tqdm

with tqdm(total=3, desc="Outer", position=0, file=sys.stdout) as main_bar:

    for i in range(3):
        with tqdm(total=10000000, mininterval=0.5, desc="Inner", position=1, leave=False, file=sys.stdout) as sub_bar:

            for j in range(10000000):
                sub_bar.update(1)

        main_bar.update(1)

###############################################
# Results cleaning.
###############################################
#%%
results = short_results

#%% Remove based on infinite or NaN values.
finite_col_mask = np.all(np.isfinite(results), axis=0)
results = results.loc[:,finite_col_mask]

#%% Remove based on coefficient of variation
# NOTE: Better than stdev as very small scaled features but with high variability are preserved.
# Also better than normalisation and then using stdev as this can create variability in flat columns
# due to floating point inaccuracies.
results = clean.remove_low_cv_features(results, epsilon=1e-6)

#%% Remove based on high collinearlity.
results = clean.remove_colinear_features(results, timings, r=.95)

#%% Remove based on VIF scoring.
results, vifs = clean.remove_features_by_vif(results, 100, 5, nan_policy="col-omit")
print("Highest VIF:", vifs.max())

#%% Remove redundant scheme names.
results = results.rename(columns=clean.reduce_stat_name, level="Statistic")
###############################################
# Basic plotting.
###############################################

# %%
reload(p)
fig, ax, hm = p.plot_heatmap(results)

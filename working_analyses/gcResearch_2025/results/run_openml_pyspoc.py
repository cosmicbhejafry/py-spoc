#%%
import pyspoc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from importlib import reload

from results import (
    plotting as p,
    clustering as c,
    similarity as s)

from results.execution import (
    openml_files as opf,
    cleaning as clean,
    pipeline as pipe,
    analysis as ana)


###############################################
# Main results run.
###############################################

cfg = pyspoc.Config.from_yaml_file(name="openml_run", yaml_file_path="main.cfg")
sampling_seed = 0
base_data_path = Path("D:/OneDrive/OneDrive - Imperial College London/Data/")
openml_dir = base_data_path / "OpenML"
openml_registry_path = openml_dir / "openml_dataset_catalogue.csv"
openml_extension = ".parquet"
openml_results_prefix = "openml_"
include_target = False
batch_n = 5000
batch_p = 50
max_batches = 10


openml_datasets = opf.get_annotated_datasets(
    openml_registry_path=openml_registry_path,
    openml_folder=openml_dir,
    extension=openml_extension,
    include_target=include_target
)

results, short_results, timings = pipe.run_pyspoc_loop(
    cfg=cfg,
    annotated_datasets=openml_datasets,
    batch_n=batch_n,
    batch_p=batch_p,
    max_batches=max_batches,
    results_path=openml_dir,
    sampling_seed=sampling_seed,
    output_files_prefix=openml_results_prefix
)

###############################################
# Results cleaning.
###############################################

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

#%% Remove based on VIF scoring
results, vifs = clean.remove_features_by_vif(results, 10, 100, nan_policy="col-omit")
print("Highest VIF:", vifs.max())

###############################################
# Basic plotting.
###############################################

# %%
reload(p)
fig, ax, hm = p.plot_heatmap(results)
# %%
fig, ax, pcat = p.plot_pca_tsne(results, perplexity=50, nan_policy="col-omit")

# %%
fig, ax, pca = p.plot_pca(results, nan_policy="col-omit")

###############################################
# Embedded space region estimation
###############################################

# %% Gaussian group assignments grid
groups = {
    name: group
    for name, group in pcat.groupby("Domain")
}

gaussians = []
group_names = []
group_points = []

for group_name, group_data in groups.items():
    grid = ana.evaluate_gaussian_density_grid(
        group_data.values,
        noise=0.5,
        grid_step=0.01,
        xlim=(-4,2),
        ylim=(-2,5.5))
    gaussians.append(grid)
    group_names.append(group_name)
    group_points.append(group_data)

n = len(group_names)
cmap = plt.get_cmap("rainbow", n)
group_colours = [cmap(i) for i in range(n)]
#%% Gaussian density group assignments (dominant density)
reload(p)
fig, ax, blended = p.plot_gaussian_density_blend(
    density_grids=gaussians,
    colours=group_colours,
    original_points=group_points,
    label_name="Domain",
    title="Density-weighted Gaussian groups (blended density)",
    labels=group_names,
    alpha_power=0.01,
    overlap_mode="blend"
)
ax.set_xlim(-3,1.5)
ax.set_ylim(-0.5,4)
#%% Gaussian density group assignments (dominant density)
reload(p)
fig, ax, blended = p.plot_gaussian_density_blend(
    density_grids=gaussians,
    colours=group_colours,
    original_points=group_points,
    label_name="Domain",
    title="Density-weighted Gaussian groups (dominant density)",
    labels=group_names,
    gaussian_alpha=0.8,
    overlap_mode="dominant"
)
ax.set_xlim(-3,1.5)
ax.set_ylim(-0.5,4)

#%%
reload(c)
#%% 1NN group assignments
grid, assignments = c.nearest_group_grid(groups, grid_step=0.01)
group_idxs = [assignments[key] - 1 for key in group_names]
new_colours = [group_colours[i] for i in group_idxs]
p.plot_nearest_group_regions(
    grid,
    assignments,
    new_colours,
    original_points=group_points,
    point_alpha=1,
    alpha=0.5,
    label_name="Domain",
    title="Nearest-point data domain group regions")
#%% KNN group assignments
grid, assignments = c.knn_group_grid(
    groups,
    grid_step=0.01
)
group_idxs = [assignments[key] - 1 for key in group_names]
new_colours = [group_colours[i] for i in group_idxs]
fig, ax, grids = p.plot_nearest_group_regions(
    grid,
    assignments,
    new_colours,
    original_points=group_points,
    point_alpha=1,
    alpha=0.5,
    label_name="Domain",
    title="5-NN data domain group regions")
#%%
grid, cluster_ids = c.kmeans_group_grid(
    groups,
    grid_step=0.05,
    n_runs=10,
    seed=42,
)

# %% Gaussian group assignments grid
groups = {
    name: group
    for name, group in pcat.groupby("Dataset Name")
}

gaussians = []
pdfs = []
ds_group_names = []
ds_group_points = []

for group_name, group_data in groups.items():
    grid, pdf = ana.evaluate_gaussian_density_grid(
        group_data.values,
        noise=0.5,
        grid_step=0.01,
        xlim=(-4,2),
        ylim=(-2,5.5))
    gaussians.append(grid)
    pdfs.append(pdf)
    ds_group_names.append(group_name)
    ds_group_points.append(group_data)

mapped_group_names = [p.index.get_level_values("Domain").unique().item()
                      for p in ds_group_points]
mapped_group_colours = [group_colours[group_names.index(g)]
                        for g in mapped_group_names]
#%% Gaussian density group assignments (dominant density)
fig, ax, blended = p.plot_gaussian_density_blend(
    density_grids=gaussians,
    colours=mapped_group_colours,
    original_points=ds_group_points,
    label_name="Domain",
    title="Density-weighted Gaussian groups (blended density)",
    labels=mapped_group_names,
    alpha_power=0.01,
    overlap_mode="blend"
)
ax.set_xlim(-3,1.5)
ax.set_ylim(-0.5,4)
#%%
###############################################
# Group similarity
###############################################

ds_names = results.index.get_level_values("Dataset Name").to_list()
modality_names = results.index.get_level_values("Modality").to_list()
domain_names = results.index.get_level_values("Domain").to_list()
ds_name_set = set(ds_names)
ds_name_idxs = [ds_names.index(x) for x in ds_name_set]
modality_index = [modality_names[i] for i in ds_name_idxs]
domain_index = [domain_names[i] for i in ds_name_idxs]
S_matrices = []

for d in range(2,11):
    pca_tsne_results = p.compute_pca_tsne(
        results,
        n_components=d,
        n_pca_components=50 + 5*d)

    groups = {
        name: group
        for name, group in pca_tsne_results.groupby("Dataset Name")
    }

    pdfs = []
    sim_group_names = []
    sim_group_points = []

    for group_name, group_data in groups.items():
        pdf = ana.fit_multivariate_gaussian_pdf(
            group_data.values,
            noise=0.5)
        pdfs.append(pdf)
        sim_group_names.append(group_name)
        sim_group_points.append(group_data)

    S = s.gaussian_group_similarity(
        pdfs,
        sim_group_points,
        sim_group_names,
        log=True
    )
    new_S_index = pd.MultiIndex.from_arrays([
            range(10),
            S.index.to_list(),
            modality_index,
            domain_index
        ],
        names = ["ID","Dataset Name", "Modality", "Domain"]
)
    S.index = new_S_index
    S_matrices.append(S)
# %%
max_similarities = []

for S in S_matrices:
    n_S = len(S.index)
    max_S = pd.DataFrame(
        data = list(zip(S.index.get_level_values("ID").to_list(), [-np.inf] * n_S)),
        index=S.index,
        columns=["Similar ID", "Log Similarity Index"])

    for i in range(n):
        best_sim = max_S.iloc[i,1]
        best_sim_idx = -1
        sim_group = S.index[i][0]

        for j in range(n):

            if i == j:
                continue

            sim = S.iloc[i,j]

            if sim > best_sim:
                best_sim = sim
                best_sim_idx = j
                best_sim_group = S.index[j][0]

        if best_sim > -np.inf:
            max_S.iloc[i] = (best_sim_group, best_sim)

    max_similarities.append(max_S)

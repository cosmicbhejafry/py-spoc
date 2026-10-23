#%%
import pyspoc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from typing import Literal, Generator, Any
from importlib import reload
from functools import partial
from working_analyses.gcResearch_2025.results.execution.cleaning import run_analysis

import plotting as p
import clustering as c

#%%

#%% Import configuration.
cfg = pyspoc.Config.from_yaml_file("esa", "cfg.yml")

#%% Immutable config params
base_data_path = Path("D:/OneDrive/OneDrive - Imperial College London/Data/")
openml_format = ".parquet"
openml_path = base_data_path / "OpenML"
openml_files = find_files(openml_path, openml_format)
openml_registry_path = openml_path / "openml_dataset_catalogue.csv"
openml_registry = get_file_registry(openml_registry_path)
result_index_list = list()
include_target = False
sampling_seed = 0
output_file_name = "openml_results.pq"
short_output_file_name = "openml_short_results.pq"
timings_file_name = "openml_timings.pq"
#%% Current run config params
batch_n = 5000
batch_p = 50
max_batches = 10
sampling_seed = 0

#%% Test run config params
# n_files_to_process = 2
#%%
run_analysis(openml_files=openml_files,
             openml_registry=openml_registry,
             cfg=cfg,
             batch_n=batch_n,
             batch_p=batch_p,
             max_batches=max_batches,
             include_target=False,
             sampling_seed=sampling_seed,
             output_file_name=output_file_name,
             short_output_file_name=short_output_file_name,
             timings_file_name=timings_file_name,
             save_frequency=3,
             raise_errors=False)

#%%
results = pd.read_parquet(output_file_name)
short_results = pd.read_parquet(short_output_file_name)
timings = pd.read_parquet(timings_file_name)
#%% Remove based on infinite or NaN values.
finite_col_mask = np.all(np.isfinite(results), axis=0)
results = results.loc[:,finite_col_mask]

#%% Remove based on coefficient of variation
# NOTE: Better than stdev as very small scaled features but with high variability are preserved.
# Also better than normalisation and then using stdev as this can create variability in flat columns
# due to floating point inaccuracies.
epsilon = 1e-6
non_zero_abs_cv = (np.abs(results.std() / results.mean()) >= epsilon)
results = results.loc[:,non_zero_abs_cv]

#%% Remove based on high collinearlity.
removal_mask = get_colinear_feature_removal_mask(results, timings)
results = results.loc[:,removal_mask]

#%% Remove based on VIF scoring
results, vifs = remove_features_by_vif(results, 10, 100, nan_policy="col-omit")
print("Highest VIF:", vifs.max())

# %%
reload(p)
fig, ax, hm = p.plot_heatmap(results)
# %%
fig, ax, pcat = p.plot_pca_tsne(results, perplexity=50, nan_policy="col-omit")

# %%
fig, ax, pca = p.plot_pca(results, nan_policy="col-omit")
# %% Gaussian group assignments grid
groups = {
    name: group
    for name, group in pcat.groupby("Domain")
}

gaussians = []
group_names = []
group_points = []

for group_name, group_data in groups.items():
    grid = evaluate_gaussian_density_grid(
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
    grid, pdf = evaluate_gaussian_density_grid(
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
import similarity as s
reload(s)

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
        pdf = fit_multivariate_gaussian_pdf(
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
    max_S = pd.DataFrame(data = list(zip(S.index.get_level_values("ID").to_list(), [-np.inf] * n_S)),
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

#%%
base_data_path
#bio_files = [base_data_path / f"Patient{i}_RNA_GEX.pq" for i in range(1,18)]
n_files_to_process = 4
k = 0
raise_errors = True
save_frequency = 1
output_file_name = "gene_expression_results2.pq"
short_output_file_name = "gene_expression_short_results2.pq"
timings_file_name = "gene_expression_timings2.pq"
max_batches = 10
#%%
result_index_list = []
gene_results = None
short_gene_results = None

for i in range(n_files_to_process):

    try:
        file = bio_files[i]
        dataset_name = file.stem
        dataset_modality = "Gene"
        dataset_domain = "Bio"
        data = pd.read_parquet(file)
        dataset = pyspoc.Dataset(data)
        batches = dataset.get_batches(
            batch_shape=(batch_n, batch_p),
            sampling_type="cyclical",
            seed=sampling_seed,
            max_n_batches=max_batches)
        
        print(f"Calculating stats for dataset {dataset_name} [{i+1}/{n_files_to_process}].")
        print("-" * 100)
        
        n_batches = len(batches)

        for j in range(n_batches):

            if n_batches > 1:
                print()
                print(f"Processing batch {j+1}/{n_batches}.")
                print("-" * 100)

            batch = batches[j]
            batch_name = dataset_name + f"_{j}"
            calc = pyspoc.Calculator(batch, name=batch_name, normalise=True)
            calc.compute(cfg)
            result = calc.get_results("full")
            short_result = calc.get_results("short")
            result_index_list.append((dataset_name, batch_name, dataset_modality, dataset_domain))

            # Collect results.
            if gene_results is None:
                gene_results = result
                short_gene_results = short_result

            else:
                gene_results = pd.concat((gene_results, result))
                short_gene_results = pd.concat((short_gene_results, short_result))

            # Collect timings.
            k += 1
            timing_results = sorted(calc._times.items(), key=lambda x: x[0], reverse=False)
            timing_idx, timing_data = tuple(zip(*timing_results))
            timing_df = pd.DataFrame(
                index=pd.Index(data=timing_idx, name="Statistic"),
                data=timing_data,
                columns=[k])
            #timings = pd.concat((timings, timing_df), axis=1)

    except Exception as e:
        if raise_errors:
            raise e
        
        print(e)

    finally:
        if i % save_frequency == 0:
            if gene_results is not None:
                results_index = pd.MultiIndex.from_tuples(
                    result_index_list,
                    names=["Dataset Name","Batch Name", "Modality", "Domain"])

                gene_results.index = results_index
                gene_results.to_parquet(output_file_name, engine="pyarrow")
                short_gene_results.to_parquet(short_output_file_name, engine="pyarrow")

            #timings.to_parquet(timings_file_name, engine="pyarrow")

if gene_results is not None:
    results_index = pd.MultiIndex.from_tuples(
        result_index_list,
        names=["Dataset Name","Batch Name", "Modality", "Domain"])
    
    gene_results.index = results_index
    gene_results.to_parquet(output_file_name, engine="pyarrow")
    short_gene_results.to_parquet(short_output_file_name, engine="pyarrow")

#timings.to_parquet(timings_file_name, engine="pyarrow")
#%%
gene_results = pd.read_parquet("gene_expression_results.pq")
gene_results2 = pd.read_parquet("gene_expression_results2.pq")
gene_results = pd.concat((gene_results, gene_results2))
#%%
finite_col_mask = np.all(np.isfinite(gene_results), axis=0)
gene_results = gene_results.loc[:,finite_col_mask]
#%% Remove based on coefficient of variation
# NOTE: Better than stdev as very small scaled features but with high variability are preserved.
# Also better than normalisation and then using stdev as this can create variability in flat columns
# due to floating point inaccuracies.
epsilon = 1e-6
non_zero_abs_cv = (np.abs(gene_results.std() / gene_results.mean()) >= epsilon)
gene_results = gene_results.loc[:,non_zero_abs_cv]
#%%
gene_results.groupby(level="Dataset Name").mean().to_parquet("avg_gene_expression_results.pq")
#%% Remove based on high collinearlity.
removal_mask = get_colinear_feature_removal_mask(gene_results, timings)
gene_results = gene_results.loc[:,removal_mask]

#%% Remove based on VIF scoring
gene_results, vifs = remove_features_by_vif(gene_results, 10, 100, nan_policy="col-omit")
print("Highest VIF:", vifs.max())

#%%
missing_idxs = [x for x in results.columns.to_list() if x not in gene_results.columns.to_list()]
accepted_cols = [x for x in results.columns if x not in missing_idxs]
results = results[accepted_cols]
full_results = pd.concat((results, gene_results))
#%%
finite_col_mask = np.all(np.isfinite(full_results), axis=0)
full_results = full_results.loc[:,finite_col_mask]
#%% Remove based on coefficient of variation
# NOTE: Better than stdev as very small scaled features but with high variability are preserved.
# Also better than normalisation and then using stdev as this can create variability in flat columns
# due to floating point inaccuracies.
epsilon = 1e-6
non_zero_abs_cv = (np.abs(full_results.std() / full_results.mean()) >= epsilon)
full_results = full_results.loc[:,non_zero_abs_cv]
#%% Remove based on high collinearlity.
removal_mask = get_colinear_feature_removal_mask(full_results, timings)
full_results = full_results.loc[:,removal_mask]
#%% Remove based on VIF scoring
full_results, vifs = remove_features_by_vif(full_results, 10, 100, nan_policy="col-omit")
print("Highest VIF:", vifs.max())

#%%
reload(p)
short_results = pd.read_parquet("openml_short_results.pq")
short_gene_results = pd.read_parquet("gene_expression_short_results.pq")
short_gene_results2 = pd.read_parquet("gene_expression_short_results2.pq")
# %%
short_gene_results = pd.concat((short_gene_results, short_gene_results2))
missing_idxs = [x for x in short_results.columns.to_list() if x not in short_gene_results.columns.to_list()]
accepted_cols = [x for x in short_results.columns if x not in missing_idxs]
short_results = short_results[accepted_cols]
full_short_results = pd.concat((short_results, short_gene_results))
full_short_results.index = full_results.index

finite_col_mask = np.all(np.isfinite(full_short_results), axis=0)
full_short_results = full_short_results.loc[:,finite_col_mask]

non_zero_abs_cv = (np.abs(full_short_results.std() / full_short_results.mean()) >= epsilon)
full_short_results = full_short_results.loc[:,non_zero_abs_cv]
removal_mask = get_colinear_feature_removal_mask(full_short_results, short_timings, r=0.5)
heatmap_results = full_short_results.loc[:,removal_mask]
print(heatmap_results.shape)
# %%
reload(p)
idxs = np.random.choice(a=range(heatmap_results.shape[1]), size=50, replace=False)
sampled_results = heatmap_results.groupby(level="Dataset Name").mean().iloc[:,idxs].sort_index()
fig, ax, hm = p.plot_heatmap(sampled_results, figsize=(25,20))
# %%
fig, ax, pcat = p.plot_pca_tsne(full_results, perplexity=40, nan_policy="col-omit")
# %%
fig, ax, pca = p.plot_pca(full_results, nan_policy="col-omit")
# %% Gaussian group assignments grid
pca_tsne_2 = p.compute_pca_tsne(
    full_results)

groups = {
    name: group
    for name, group in pca_tsne_2.groupby("Domain")
}

gaussians = []
group_names = []
group_points = []

for group_name, group_data in groups.items():
    grid, pdf = evaluate_gaussian_density_grid(
        group_data.values,
        noise=0.5,
        grid_step=0.01,
        xlim=(-55,45),
        ylim=(-15,10))
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
    alpha_power=0.1,
    figsize = (20,5),
    overlap_mode="blend"
)
ax.set_xlim(-45,42)
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
ax.set_xlim(-30,55)
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
    grid, pdf = evaluate_gaussian_density_grid(
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
import similarity as s
reload(s)

ds_names = full_results.index.get_level_values("Dataset Name").to_list()
modality_names = full_results.index.get_level_values("Modality").to_list()
domain_names = full_results.index.get_level_values("Domain").to_list()
ds_name_set = set(ds_names)
ds_name_idxs = [ds_names.index(x) for x in ds_name_set]
modality_index = [modality_names[i] for i in ds_name_idxs]
domain_index = [domain_names[i] for i in ds_name_idxs]
S_matrices = []

for d in range(2,11):
    pca_tsne_results = p.compute_pca_tsne(
        full_results,
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
        pdf = fit_multivariate_gaussian_pdf(
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
            range(39),
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
    max_S = pd.DataFrame(data = list(zip(S.index.get_level_values("ID").to_list(), [-np.inf] * n_S)),
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
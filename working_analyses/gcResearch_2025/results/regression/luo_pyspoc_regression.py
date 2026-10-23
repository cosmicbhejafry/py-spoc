#%%
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import re
import lasso as ls
import ols
import plotting as p

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

###############################################
# Data loading and processing
###############################################
#%% Set data paths
reload_required = False # Set this to true to perform the data loading and processing again.
data_dir = "D:/OneDrive/OneDrive - Imperial College London/Data"
mut_file = "Luo_PBMC_RNA_T_dfMutationalLoads.pq"
exp_file = "Luo_PBMC_RNA_GEX_only.h5ad"
data_path = Path(data_dir)
data_path = data_path / "Luo"
#%% Get mutation load data.
mut_file_path = data_path / mut_file
muts = pd.read_parquet(mut_file_path)
#%% Split expression data by patient.

if reload_required:
    # Get gene expression matrix
    exp_file_path = data_path / exp_file
    adata = ad.read_h5ad(exp_file_path)

    # Filter on T cells (largest cohort of cells).
    T_cell_mask = adata.obs["manual_annot_leiden_res0.21"] == "T"
    T_cells_df = adata.to_df().loc[T_cell_mask] # pyright: ignore[reportCallIssue]
    T_cells_donors = adata.obs["donor_num"].loc[T_cell_mask] # pyright: ignore[reportCallIssue]
    T_cells_df.index = pd.MultiIndex.from_arrays(
        [T_cells_df.index.to_list(),
        T_cells_donors.to_list()],
        names=["Sequence", "DonorNum"])

    # Save to parquet
    T_cells_df.to_parquet(data_path / "Luo_PBMC_RNA_GEX_only.pq")
    # Load parquet (only needed when resuming).
    T_cells_df = pd.read_parquet(data_path / "Luo_PBMC_RNA_GEX_only.pq")
    donor_dfs = {name : group for name, group in T_cells_df.groupby("DonorNum")}

    # Split the parquet by donor number
    for num, donor_df in donor_dfs.items():
        donor_df.to_parquet(data_path / f"Patient{num}_RNA_GEX.pq")

#%% Iterate over donor numbers and save individual parquets.
patient_pattern = re.compile(r".*Patient\d+_RNA_GEX.pq$")
patient_files = [d for d in data_path.glob("Patient*_RNA_GEX.pq")
                 if patient_pattern.match(str(d))]

mean_exps = []

for i, file_name in enumerate(patient_files):
    file_path = data_path / file_name
    patient_df = pd.read_parquet(file_path)
    patient_mean_exp_data = patient_df.mean()
    mean_exps.append(patient_mean_exp_data)

#%% Save as our predictors and target.
X_df = pd.concat(mean_exps, axis=1).T
y_clp = muts["HHLP_mean"]
y_age = muts["age"]

# %%
fig, ax = plt.subplots(figsize=(8, 6))
p.plot_scatter(y_age, y_clp, ax=ax)
slope, intercept = p.get_simple_regression_line(y_age, y_clp)
xlim = ax.get_xlim()
x_line = np.linspace(xlim[0], xlim[1], 200)
y_line = slope * x_line + intercept

ax.plot(
    x_line,
    y_line,
    alpha=0.5,
    color="tab:red",
    linewidth=2,
    label="Regression fit"
)
ax.grid(alpha=0.3)
ax.set_xlabel("Patient age")
ax.set_ylabel("Patient CLP load")
ax.set_title("Simple regression of patient ages onto CLP load")
ax.legend()

# ax.plot(
#     x_line,
#     y_line,
#     alpha=0.5,
#     color="tab:red",
#     linewidth=2,
#     label="Regression fit")

# ax.set_xlabel("Patient age")
# ax.set_ylabel("Patient CLP load")
# ax.set_title("Simple regression of patient ages onto CLP load")
# ax.legend()

y_clp_pred = slope * y_age + intercept
print(
    "MAE:", mean_absolute_error(y_clp_pred, y_clp), "\n" +
    "RMSE:", root_mean_squared_error(y_clp_pred, y_clp), "\n" +
    "R2:", r2_score(y_clp_pred, y_clp))

###############################################
# Mean Expression Regression
###############################################

#%% LASSO regression on average expressions
# We run it over the whole bunch because we're investigating information retention,
# hence whether a linear relationship can be found rather than whether the model
# generalises (only 16 patients).
alphas = [0.01, 0.1, 0.2]
results = dict()
add_age = True

for alpha in alphas:
    model, X_scaled, scaler = ls.run_lasso(X_df, y_clp, alpha=alpha)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    abs_coefs = np.abs(model.coef_)
    sort_idx = np.argsort(abs_coefs, descending=True)
    predictors = X_df.columns[sort_idx]
    sorted_coefs = abs_coefs[sort_idx]
    zero_idx = np.argmin(sorted_coefs)
    selected_predictors = predictors[:zero_idx].to_list()
    X_selected = X_df[selected_predictors]

    if add_age:
        X_selected = pd.concat((X_selected, y_age), axis=1)

    model, X_scaled, scaler = ols.run_ols(X_selected, y_clp)
    y_pred = model.predict(X_scaled)
    title = "CLP load vs. OLS predicted CLP load from LASSO selected average expression levels"

    if add_age:
        title += " and age."

    # Plot the predictions.
    p.plot_model_scatter(
        model,
        X_scaled,
        y_clp,
        x_label="CLPs",
        y_label="Predicted CLPs",
        title=title,
        subtitle=f"LASSO $\\alpha = {alpha}$",
        show_line=True)

    plt.show()

    results[alpha] = {
             "MAE":mean_absolute_error(y_clp, y_pred),
             "RMSE": root_mean_squared_error(y_clp, y_pred),
             "R2": r2_score(y_clp, y_pred),
             "Selected": selected_predictors
        }

    print("Alpha:", alpha, "\n" +
        "MAE:", results[alpha]["MAE"], "\n" +
        "RMSE:", results[alpha]["RMSE"], "\n" +
        "R2:", results[alpha]["R2"], "\n" +
        "Selected Genes:", "\n\t" + "\n\t".join(selected_predictors))

#%% Permutation testing.
alpha = 0.2
base_r2 = results[alpha]["R2"]
n_perm = 100

fig, ax, maes, rmses, r2s, avg_in_common = p.plot_model_permutations(
    X_df,
    y_clp,
    n_perm=n_perm,
    lasso_alpha=alpha,
    use_ols=True,
    x_label="Permuted CLPs",
    y_label="Predicted CLPs",
    title="Permutated fits for CLP load vs. OLS predicted CLP load from selected average expression levels",
    subtitle=f"LASSO $\\alpha = {alpha}$ (1000 Permutated Fits)",
    point_alpha=0.5,
    line_alpha=0.05,
    lim_buffer=0.1,
    seed=0
)
plt.show()
print("Average MAE:", np.array(maes).mean(), "\n" +
      "Average RMSE:", np.array(rmses).mean(), "\n" +
      "Average R2:", np.array(r2s).mean(), "\n" +
      "Permuted R2 greater than real R2 p-value:",
        (1 + (np.array(r2s) > base_r2).sum()) / (n_perm + 1), "\n" +
      "Average features in common:", avg_in_common)


###############################################
# Mean Expression PCA LASSO
###############################################

#%% LASSO regression on average expressions 10 principal components.
num_pcs = 10
add_age = False
pca = PCA(n_components=num_pcs, random_state=0)
X = X_df if not add_age else pd.concat((X_df, y_age), axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
model, X_scaled, scaler = ols.run_ols(X_pca, y_clp)
y_pred = model.predict(X_scaled)

# Plot the predictions.
p.plot_model_scatter(
    model,
    X_scaled,
    y_clp,
    x_label="CLPs",
    y_label="Predicted CLPs",
    title=f"CLPs vs. OLS predicted CLPs from average expression level PCA ({num_pcs} PCs)",
    show_line=True)
plt.show()
base_r2 = r2_score(y_clp, y_pred)

print("MAE:", mean_absolute_error(y_clp, y_pred), "\n" +
      "RMSE:", root_mean_squared_error(y_clp, y_pred), "\n" +
      "R2:", r2_score(y_clp, y_pred))

#%% Check the predictor importance based on PCA weighting.
weights = pca.components_

feature_importance = (
    weights**2 * pca.explained_variance_ratio_[:, np.newaxis]
).sum(axis=0)

# Normalize retained-feature importance to sum to one.
feature_importance /= feature_importance.sum()

importance = pd.Series(
    feature_importance,
    index=X.columns,
    name="PCA importance",
).sort_values(ascending=False)

sorted_importance = importance.sort_values(ascending=False)[:num_pcs]

print("|Feature|Importance|")
print("|---|---|")
for i in range(len(sorted_importance)):
    print("|",sorted_importance.index[i] , "|", round(sorted_importance.iloc[i],12), "|")

#%% Check the predictor importance based on model coefficients.
original_coefficients = pca.components_.T @ model.coef_

regression_importance = pd.Series(
    np.abs(original_coefficients),
    index=X.columns,
    name="PCA regression importance",
).sort_values(ascending=False)

sorted_regression_importance = regression_importance.sort_values(ascending=False)[:num_pcs]

print("|Feature|Importance|")
print("|---|---|")
for i in range(len(sorted_regression_importance)):
    print("|",sorted_regression_importance.index[i] , "|",
          round(sorted_regression_importance.iloc[i],6), "|")

#%% Permutation testing.
n_perm = 1000
fig, ax, maes, rmses, r2s, avg_in_common = p.plot_model_permutations(
    X_pca,
    y_clp,
    n_perm=n_perm,
    lasso_alpha=0,
    x_label="Permuted CLPs",
    y_label="Predicted CLPs",
    title=f"Permutated fits for CLP load vs. CLP load from average expression level PCA ({num_pcs} PCs)",
    subtitle="(1000 Permutated Fits)",
    point_alpha=0.5,
    line_alpha=0.05,
    lim_buffer=0.5,
    seed=0
)
plt.show()
print("Average MAE:", np.array(maes).mean(), "\n" +
      "Average RMSE:", np.array(rmses).mean(), "\n" +
      "Average R2:", np.array(r2s).mean(), "\n" +
      "Permuted R2 less than real R2 p-value:",
        (1 + (np.array(r2s) > base_r2).sum()) / (n_perm + 1))

###############################################
# pySPoC Expression Features LASSO
###############################################

#%% LASSO regression on pySPoC expression features.
X_pyspoc_df = (pd
    .read_parquet(data_path / "luo_short_results_cleaned.pq")
    .groupby(level="Dataset Name")
    .mean())
X_pyspoc_df.index = pd.MultiIndex.from_arrays(
    [
        [int(x[7:x.find("_")]) for x in X_pyspoc_df.index.to_list()],
        X_pyspoc_df.index.to_list()
    ], names=["DonorNum", "Dataset Name"])

X_pyspoc_df = X_pyspoc_df.sort_index(level=0)

alphas = [0.01, 0.1, 0.2]
results = dict()
add_age = False

for alpha in alphas:
    model, X_scaled, scaler = ls.run_lasso(X_pyspoc_df, y_clp, alpha=alpha)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected feature idxs:", np.flatnonzero(model.coef_))

    abs_coefs = np.abs(model.coef_)
    sort_idx = np.argsort(abs_coefs, descending=True)
    predictors = X_pyspoc_df.columns[sort_idx]
    sorted_coefs = abs_coefs[sort_idx]
    zero_idx = np.argmin(sorted_coefs)
    selected_predictors = predictors[:zero_idx].to_list()
    X_selected = X_pyspoc_df[selected_predictors]
    selected_predictors = [x[0] +"=>" + x[1] for x in selected_predictors]

    if add_age:
        X_selected = pd.concat((X_selected,
                                pd.DataFrame(y_age.values,
                                             index=X_selected.index,
                                             columns=["Age"])),
                               axis=1)

    model, X_scaled, scaler = ols.run_ols(X_selected, y_clp)
    y_pred = model.predict(X_scaled)
    title = "CLP load vs. OLS predicted CLP load from LASSO selected pySPoC expression features"

    if add_age:
        title += " and age."

    fig, ax = p.plot_model_scatter(
        model,
        X_scaled,
        y_clp,
        x_label="CLPs",
        y_label="Predicted CLPs",
        title=title,
        subtitle=f"LASSO $\\alpha = {alpha}$",
        show_line=True)

    plt.show()

    results[alpha] = {
             "MAE":mean_absolute_error(y_clp, y_pred),
             "RMSE": root_mean_squared_error(y_clp, y_pred),
             "R2": r2_score(y_clp, y_pred),
             "Selected": selected_predictors
        }

    print("Alpha:", alpha, "\n" +
        "MAE:", results[alpha]["MAE"], "\n" +
        "RMSE:", results[alpha]["RMSE"], "\n" +
        "R2:", results[alpha]["R2"], "\n" +
        "Selected Features:", "\n\t" + "\n\t".join(selected_predictors))

#%% Permutation testing.
alpha = 0.2
base_r2 = results[alpha]["R2"]
n_perm = 1000
fig, ax, maes, rmses, r2s, avg_in_common = p.plot_model_permutations(
    X_pyspoc_df,
    y_clp,
    n_perm=n_perm,
    lasso_alpha=alpha,
    x_label="Permuted CLPs",
    y_label="Predicted CLPs",
    title="Permutated fits for CLP load vs. OLS predicted CLP load from selected pySPoC expression features",
    subtitle=f"LASSO $\\alpha = {alpha}$ (1000 Permutated Fits)",
    point_alpha=0.5,
    line_alpha=0.05,
    lim_buffer=0.1,
    seed=0
)
plt.show()
print("Average MAE:", np.array(maes).mean(), "\n" +
      "Average RMSE:", np.array(rmses).mean(), "\n" +
      "Average R2:", np.array(r2s).mean(), "\n" +
      "Permuted R2 less than real R2 p-value:",
        (1 + (np.array(r2s) > base_r2).sum()) / (n_perm + 1), "\n" +
      "Average features in common:", avg_in_common)

#%% LASSO regression on pySPoC expression features on 10 principal components.
num_pcs = 10
add_age = False
pca = PCA(n_components=num_pcs, random_state=0)
X = X_pyspoc_df if not add_age else pd.concat((X_pyspoc_df,
    pd.DataFrame(y_age.values,
        index=X_selected.index,
        columns=["Age"])),
    axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
X_pyspoc_pca = pca.fit_transform(X_scaled)
model, X_scaled, scaler = ols.run_ols(X_pyspoc_pca, y_clp)
y_pred = model.predict(X_scaled)

fig, ax = p.plot_model_scatter(
    model,
    X_scaled,
    y_clp,
    x_label="CLPs",
    y_label="Predicted CLPs",
    title=f"CLPs vs. OLS predicted CLPs from pySPoC expression features PCA ({num_pcs} PCs)",
    show_line=True)

plt.show()
print("MAE:", mean_absolute_error(y_clp, y_pred), "\n" +
      "RMSE:", root_mean_squared_error(y_clp, y_pred), "\n" +
      "R2:", r2_score(y_clp, y_pred))
#%% Check the predictor importance based on PCA weighting.
weights = pca.components_

feature_importance = (
    weights**2 * pca.explained_variance_ratio_[:, np.newaxis]
).sum(axis=0)

# Normalize retained-feature importance to sum to one.
feature_importance /= feature_importance.sum()

importance = pd.Series(
    feature_importance,
    index=X.columns,
    name="PCA importance",
).sort_values(ascending=False)

sorted_importance = importance.sort_values(ascending=False)[:num_pcs]
print("|Statistic|Reduction|Importance|")
print("|---|---|---|")
for i in range(len(sorted_importance)):
    print("|",sorted_importance.index[i][0] , "|",
          sorted_importance.index[i][1] , "|",
          round(sorted_importance.iloc[i],10), "|")

#%% Check the predictor importance based on model coefficients.
original_coefficients = pca.components_.T @ model.coef_

regression_importance = pd.Series(
    np.abs(original_coefficients),
    index=X.columns,
    name="PCA regression importance",
).sort_values(ascending=False)

sorted_regression_importance = regression_importance.sort_values(ascending=False)[:num_pcs]

print("|Statistic|Reduction|Importance|")
print("|---|---|---|")
for i in range(len(sorted_regression_importance)):
    print("|",sorted_regression_importance.index[i][0] , "|",
          sorted_regression_importance.index[i][1] , "|",
          round(sorted_regression_importance.iloc[i],10), "|")

#%% Permutation testing.
n_perm = 1000
fig, ax, maes, rmses, r2s, avg_in_common = p.plot_model_permutations(
    pd.DataFrame(X_pyspoc_pca),
    y_clp,
    n_perm=n_perm,
    lasso_alpha=0,
    x_label="Permuted CLPs",
    y_label="Predicted CLPs",
    title=f"Permutated fits for CLP load vs. CLP load from pySPoC expression features PCA ({num_pcs} PCs)",
    subtitle="(1000 Permutated Fits)",
    point_alpha=0.5,
    line_alpha=0.05,
    lim_buffer=0.1,
    seed=0
)
plt.show()
print("Average MAE:", np.array(maes).mean(), "\n" +
      "Average RMSE:", np.array(rmses).mean(), "\n" +
      "Average R2:", np.array(r2s).mean(), "\n" +
      "Permuted R2 less than real R2 p-value:",
        (1 + (np.array(r2s) > base_r2).sum()) / (n_perm + 1))

#%% Expression averages alpha sweep.
expr_results_dict = ls.run_lasso_alpha_sweep(
    X_df,
    y_clp,
    alphas=np.linspace(0.01,0.2,num=11))
expr_results_dict
# pySPoC features alpha sweep.
pyspoc_results_dict = ls.run_lasso_alpha_sweep(
    X_pyspoc_df,
    y_clp,
    alphas=np.linspace(0.01,0.2,num=11))
pyspoc_results_dict
# Plot sweep
expr_results_metrics = expr_results_dict["Metrics"]
pyspoc_results_metrics = pyspoc_results_dict["Metrics"]
p.plot_model_alpha_sweep(
    expr_results_metrics, pyspoc_results_metrics, include_plots=["RMSE", "R2"])

#%% Expression averages alpha sweep.
expr_results_dict = ls.run_lasso_alpha_sweep(
    X_pca,
    y_clp,
    alphas=np.linspace(0.01,0.2,num=11))
expr_results_dict
# pySPoC features alpha sweep.
pyspoc_results_dict = ls.run_lasso_alpha_sweep(
    X_pyspoc_pca,
    y_clp,
    alphas=np.linspace(0.01,0.2,num=11))
pyspoc_results_dict
# Plot sweep
expr_results_metrics = expr_results_dict["Metrics"]
pyspoc_results_metrics = pyspoc_results_dict["Metrics"]
p.plot_model_alpha_sweep(
    pyspoc_results_metrics, expr_results_metrics, include_plots=["RMSE", "R2"], is_pca=True)

# NOTE:
# Results are good.
# - pySPoC features provide roughly equal amount of information to the average
#   expression values, allowing for near-perfect fitting on the training data.
# - pySPoC features perform BETTER when restricted to 10 principal components.
#   Suggests greater density of information capture.
# - Most important pySPoC features by far are the third moments of the Canberra
#   pairwise distance matrix, based on regression coefficients.
# - Manhattan distance third moments also provide some information.

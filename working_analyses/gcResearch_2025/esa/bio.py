#%%
from pathlib import Path
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
def run_lasso_cv(X, y, cv):
    X = np.asarray(X)
    y = np.asarray(y)
    cv_splits = list(cv.split(X, y))

    # Lasso is sensitive to feature scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(
        alphas=100,          # Number of automatically selected alpha values
        cv=cv_splits,        # Reusable observation splits
        max_iter=10_000,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_scaled, y)

    print("Selected alpha:", model.alpha_)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    return model, X_scaled, scaler

def run_lasso(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    # Lasso is sensitive to feature scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(
        alpha = 0.02,
        max_iter=10_000,
        random_state=42,
    )

    model.fit(X_scaled, y)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Selected features:", np.flatnonzero(model.coef_))

    return model, X_scaled, scaler

def plot_scatter(
        x_values,
        y_values,
        x_label=None,
        y_label=None,
        title=None,
        ):
    x = np.asarray(x_values)
    y = np.asarray(y_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x,
        y,
        s=30,             # Point size
        alpha=0.7,        # Point transparency
        color="tab:blue",
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_xlabel(x_label or "X variable")
    ax.set_ylabel(y_label or "Y variable")
    ax.set_title(title or "Scatter plot")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, ax

#%%

data_path = Path("D:/OneDrive/OneDrive - Imperial College London/Data")
#%%
mut_file = "Luo_PBMC_RNA_T_dfMutationalLoads.pq"
mut_file_path = data_path / mut_file
muts = pd.read_parquet(mut_file_path)
muts

#%%
import anndata as ad
exp_file = "Luo_PBMC_RNA_GEX_only.h5ad"
exp_file_path = data_path / exp_file
adata = ad.read_h5ad(exp_file_path)
#%%
print(adata)
#%%
print(type(adata.X))
import numpy as np
from scipy import sparse

matrix = adata.X

if sparse.issparse(matrix):
    matrix = matrix.toarray()
else:
    matrix = np.asarray(matrix)
#%%
pd.DataFrame(
    data = adata.layers["counts"].toarray(),
    columns = adata.var.index,
    index = adata.obs.index
).to_parquet(data_path / "Luo_PBMC_RNA_GEX_only-counts.pq")
# %%
gene_exp_processed = pd.read_parquet(data_path / "Luo_PBMC_RNA_GEX_only.pq")
gene_exp_processed.index = pd.MultiIndex.from_arrays(
    [gene_exp_processed.index.to_list(),
    adata.obs["donor_num"].to_list(),])
# %%
matrices = {name : group for name, group in gene_exp_processed.groupby(level=1)}
# %%
for name, group_data in matrices.items():
    group_data.to_parquet(data_path / f"Patient{name}_RNA_GEX.pq")

#%%
muts = pd.read_parquet(data_path / "Luo_PBMC_RNA_T_dfMutationalLoads.pq")
#%%
gene_exp_processed = pd.read_parquet(data_path / "Patient1_RNA_GEX.pq")
gene_exp_processed
#%%
avg_expression = gene_exp_processed.mean()
#%%
gene_exp_file_names = [f"Patient{i}_RNA_GEX.pq" for i in [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]]
mean_exps = []

for i, file_name in enumerate(gene_exp_file_names):
    file_path = data_path / file_name
    exp_data = pd.read_parquet(file_path)
    mean_exp_data = exp_data.mean()
    mean_exps.append(mean_exp_data)

#%%
# Generate and store the splits once.
cv = KFold(
    n_splits=2,
    shuffle=True,
    random_state=42,
)

X_df = pd.concat(mean_exps, axis=1).T
X_train = X_df.iloc[:10]
X_test = X_df.iloc[10:]
y_train = muts["HHLP_mean"].iloc[:10]
y_test = muts["HHLP_mean"].iloc[10:]
model, X_scaled, scaler = run_lasso(X_train, y_train)
X_test_scaled = scaler.transform(np.asarray(X_test))
X_all_scaled = np.vstack((X_scaled, X_test_scaled))
y_all = np.hstack((y_train, y_test))
y_pred = model.predict(X_all_scaled)
plot_scatter(y_all,
             y_pred,
             title="HHLP load vs. LASSO Predicted load from average expression levels",
             x_label="HHLP load",
             y_label="Predicted load")
#%%
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

print(mean_absolute_error(y_all, y_pred),
      root_mean_squared_error(y_all, y_pred),
      r2_score(y_all, y_pred))

#%%
X_pyspoc_df = pd.read_parquet("avg_gene_expression_results.pq")
X_pyspoc_df.index = pd.MultiIndex.from_arrays(
    [
        [int(x[7:x.find("_")]) for x in X_pyspoc_df.index.to_list()],
        X_pyspoc_df.index.to_list()
    ]
)
X_pyspoc_df = X_pyspoc_df.sort_index(level=0)
X_pyspoc_train = X_pyspoc_df.iloc[:8]
X_pyspoc_test = X_pyspoc_df.iloc[8:]
y_train = muts["HHLP_mean"].iloc[:8]
y_test = muts["HHLP_mean"].iloc[8:10]
model, X_scaled, scaler = run_lasso(X_pyspoc_train, y_train)
X_test_scaled = scaler.transform(np.asarray(X_pyspoc_test))
X_all_scaled = np.vstack((X_scaled, X_test_scaled))
y_all = np.hstack((y_train, y_test))
y_pred = model.predict(X_all_scaled)

fig, ax = plot_scatter(y_all,
             y_pred,
             title="HHLP load vs. LASSO Predicted load from pySPoC expression features",
             x_label="HHLP load",
             y_label="Predicted load")

ax.set_ylim(0.05, 0.85)
plt.show()
# %%
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

print(mean_absolute_error(y_all, y_pred),
      root_mean_squared_error(y_all, y_pred),
      r2_score(y_all, y_pred))

# %%

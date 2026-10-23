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

import plotting as p
import clustering as c

#%%
def remove_features_by_vif(
        data: pd.DataFrame,
        threshold: float,
        min_columns: int,
        nan_policy: Literal[
            "report",
            "col-omit",
            "row-omit"] = "report") -> tuple[pd.DataFrame, pd.Series]:
    """Iteratively remove features with excessive variance inflation.

    At each iteration, the variance inflation factor (VIF) is calculated for
    every remaining column and the column with the largest VIF is removed. The
    process stops when every VIF is at or below ``threshold`` or removing
    another column would reduce the result below ``min_columns``.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric feature matrix containing finite values.
    threshold : float
        Maximum permitted VIF. A common choice is 5 or 10.
    min_columns : int
        Minimum number of columns to retain.

    Returns
    -------
    trimmed_data : pandas.DataFrame
        Copy of ``data`` containing the retained columns. The input frame is
        not modified.
    vif_scores : pandas.Series
        Final VIF of each retained column, indexed by its original label.

    Raises
    ------
    TypeError
        If the arguments have incompatible types or the frame contains
        non-numeric columns.
    ValueError
        If ``threshold`` is not finite and positive, ``min_columns`` is outside
        the available column range, or the data contain missing or infinite
        values.

    Notes
    -----
    A VIF of one is assigned when only one feature remains because there are
    no other features capable of explaining its variance.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be a real number.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and greater than zero.")
    if isinstance(min_columns, bool) or not isinstance(
            min_columns, (int, np.integer)):
        raise TypeError("min_columns must be an integer.")
    if min_columns < 1 or min_columns > data.shape[1]:
        raise ValueError(
            "min_columns must be between one and the number of data columns."
        )

    non_numeric = data.select_dtypes(exclude=np.number).columns
    if len(non_numeric) > 0:
        raise TypeError(
            "VIF feature removal requires numeric columns; received "
            f"non-numeric columns {list(non_numeric)!r}."
        )

    # Work on an independent frame so both column removal and any future
    # preprocessing added here cannot affect the caller's data.
    trimmed_data = data.copy(deep=True)
    values = trimmed_data.to_numpy(dtype=float, copy=False)
    is_finite_array = np.isfinite(values)

    if not is_finite_array.all():
        if nan_policy == "report":
            raise ValueError("VIF feature removal requires finite, non-missing data.")

        if nan_policy == "col-omit":
            finite_mask = np.all(is_finite_array, axis=0)
            trimmed_data = trimmed_data.loc[:,finite_mask]

        if nan_policy == "row-omit":
            finite_mask = np.all(is_finite_array, axis=1)
            trimmed_data = trimmed_data.loc[finite_mask]

    def calculate_vif(frame: pd.DataFrame) -> pd.Series:
        """Calculate VIF values while retaining the frame's column index."""
        if frame.shape[1] == 1:
            return pd.Series([1.0], index=frame.columns, name="VIF")

        matrix = frame.to_numpy(dtype=float, copy=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = []
            progress = tqdm(range(matrix.shape[1]))

            for column_index in progress:
                name = frame.columns[column_index]
                progress.set_description(str(name))
                score = variance_inflation_factor(matrix, column_index)
                scores.append(score)

        return pd.Series(scores, index=frame.columns, dtype=float, name="VIF")

    print("Computing VIF scores...")
    vif_scores = calculate_vif(trimmed_data)

    while trimmed_data.shape[1] > min_columns:
        # An infinite VIF correctly wins this comparison. Undefined VIFs are
        # ignored; if every score is undefined there is no defensible column
        # choice, so return the current frame and scores unchanged.
        comparable_scores = vif_scores.to_numpy(copy=True)
        comparable_scores[np.isnan(comparable_scores)] = -np.inf

        if np.all(np.isneginf(comparable_scores)):
            break

        removal_position = int(np.argmax(comparable_scores))
        maximum_vif = comparable_scores[removal_position]

        if maximum_vif <= threshold:
            break
        
        print(trimmed_data.columns[removal_position], "was removed with VIF", maximum_vif)
        keep_positions = np.arange(trimmed_data.shape[1]) != removal_position
        trimmed_data = trimmed_data.iloc[:, keep_positions]
        vif_scores = calculate_vif(trimmed_data)

    return trimmed_data, vif_scores

def find_files(folder: str | Path, extension: str) -> list[tuple[Path, Path]]:
    folder = Path(folder)
    extension = extension.lstrip(".")
    files = list(folder.rglob(f"*.{extension}"))
    paired_files = list()
    n_files = len(files)

    while files:
        file = files.pop()
        n_files -= 1
        file_name = file.stem
        file_parts = file_name.split("_")

        if len(file_parts) == 1:
            paired_files.append((file,))
            continue

        if len(file_parts) == 2:
            file_type, file_id = file_parts
            paired_type = "y" if file_type == "X" else "X"
            paired_file_name = paired_type + "_" + file_id

            for i in range(n_files-1,-1,-1):
                alt_file = files[i]
                alt_file_name = alt_file.stem

                if alt_file_name != paired_file_name:
                    continue

                if paired_type == "y":
                    paired_files.append((file, alt_file))
                else:
                    paired_files.append((alt_file, file))

                files.remove(alt_file)
                n_files -= 1
                break

    return paired_files


def get_file_registry(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path} does not exist.")

    if not file_path.is_file():
        raise IsADirectoryError(f"Path {file_path} is not a file.")

    if not file_path.suffix != "csv":
        raise ValueError(f"Path {file_path} is not a CSV file.")

    return pd.read_csv(file_path, index_col="DatasetID")

def get_random_consistent_subset(
        required_size: int,
        axis: Literal[0, 1],
        samples: int,
        data: np.ndarray | pd.DataFrame) -> Generator[np.ndarray]:

    data_np = data.values if isinstance(data, pd.DataFrame) else data
    actual_size = data.shape[axis]
    choices = list(range(0,actual_size))
    np.random.seed(0)
    choices_perm = np.random.choice(choices, size=actual_size, replace=False)
    choices_perm_copy = list(choices_perm)

    for i in range(samples):
        vars = choices_perm_copy[:required_size]
        current_size = len(vars)

        while current_size < required_size:
            remaining_size = required_size - current_size
            choices_perm_copy = list(choices_perm)
            vars.extend(choices_perm_copy[:remaining_size])
            choices_perm_copy = choices_perm_copy[remaining_size:]
        else:
            choices_perm_copy = choices_perm_copy[required_size:]

        if axis == 0:
            yield data_np[vars, :]
        else:
            yield data_np[:, vars]

def run_analysis(
        openml_files: list[tuple[Path, Path]],
        openml_registry: pd.DataFrame,
        cfg: pyspoc.Config,
        batch_n: int,
        batch_p: int,
        sampling_seed: int,
        max_batches: int,
        n_files_to_process: int | None = None,
        include_target: bool = True,
        output_file_name="openml_results.pq",
        short_output_file_name="openml_short_results.pq",
        timings_file_name="openml_timings.pq",
        save_frequency: int = 10,
        raise_errors: bool = True):

    results = None
    timings = pd.DataFrame(index=pd.Index(data=cfg.statistics.keys(), name="Statistic"))
    k = 0
    result_index_list = []
    n_openml_files = len(openml_files)
    n_files_to_process = (n_openml_files
        if n_files_to_process is None
        else min(n_files_to_process, n_openml_files))

    for i in range(n_files_to_process):

        try:
            X_file, y_file = openml_files[i]

            if not X_file:
                print(f"Skipping dataset {i+1}/{n_files_to_process} due to lack of design matrix.")
                continue

            file_name = X_file.stem
            dataset_id = int(file_name.split("_")[1])
            dataset_info = openml_registry.loc[dataset_id]
            dataset_name = dataset_info.Name
            dataset_modality = dataset_info.Modality
            dataset_domain = dataset_info.Domain
            X_data = pd.read_parquet(X_file)
            y_data = pd.read_parquet(y_file)

            if include_target:
                data = pd.concat((X_data, y_data), axis=1)
            else:
                data = X_data

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
                if results is None:
                    results = result
                    short_results = short_result

                else:
                    results = pd.concat((results, result))
                    short_results = pd.concat((short_results, short_result))

                # Collect timings.
                k += 1
                timing_results = sorted(calc._times.items(), key=lambda x: x[0], reverse=False)
                timing_idx, timing_data = tuple(zip(*timing_results))
                timing_df = pd.DataFrame(
                    index=pd.Index(data=timing_idx, name="Statistic"),
                    data=timing_data,
                    columns=[k])
                timings = pd.concat((timings, timing_df), axis=1)

        except Exception as e:
            if raise_errors:
                raise e
            
            print(e)

        finally:
            if i % save_frequency == 0:
                if results is not None:
                    results_index = pd.MultiIndex.from_tuples(
                        result_index_list,
                        names=["Dataset Name","Batch Name", "Modality", "Domain"])

                    results.index = results_index
                    results.to_parquet(output_file_name, engine="pyarrow")
                    short_results.to_parquet(short_output_file_name, engine="pyarrow")

                timings.to_parquet(timings_file_name, engine="pyarrow")

    if results is not None:
        results_index = pd.MultiIndex.from_tuples(
            result_index_list,
            names=["Dataset Name","Batch Name", "Modality", "Domain"])
        
        results.index = results_index
        results.to_parquet(output_file_name, engine="pyarrow")
        short_results.to_parquet(short_output_file_name, engine="pyarrow")

    timings.to_parquet(timings_file_name, engine="pyarrow")

def get_colinear_feature_removal_mask(
        results: pd.DataFrame,
        timings: pd.DataFrame,
        r: float = 0.95) -> np.ndarray:

    mean_timings = timings.mean(axis=1)
    results_corr = results.corr().values
    results_corr = np.array(results_corr)
    results_corr[np.tril_indices_from(results_corr, k=0)] = 0
    abs_results_corr = np.abs(results_corr)
    n_vars = abs_results_corr.shape[0]
    i = 0
    mask = np.full(shape=n_vars, fill_value=True)

    while i < n_vars:
        var_corr = abs_results_corr[i]

        if np.all(np.isnan(var_corr)):
            i += 1
            continue

        max_corr_idx = np.nanargmax(var_corr)
        max_corr = var_corr[max_corr_idx]

        if max_corr < r:
            print(f"No high correlation found for variable {i}. Continuing.")
            i += 1
            continue

        var_name = results.columns[i]
        var_timing = mean_timings.loc[var_name[0]]
        other_var_name = results.columns[max_corr_idx]
        other_var_timing = mean_timings.loc[other_var_name[0]]
        removal_idx = i if var_timing > other_var_timing else max_corr_idx
        mask[removal_idx] = False
        #abs_results_corr = np.delete(abs_results_corr, removal_idx, axis=0)
        #abs_results_corr = np.delete(abs_results_corr, removal_idx, axis=1)
        abs_results_corr[removal_idx,:] = np.nan
        abs_results_corr[:,removal_idx] = np.nan
        print(f"High correlation found for variable {i}. "
            f"Removing variable {removal_idx} due to slower timing.")

    return mask

def fit_multivariate_gaussian_pdf(
        data: np.ndarray,
        noise: float) -> partial[Any]:
    """Fit an arbitrary-dimensional multivariate Gaussian PDF.

    Parameters
    ----------
    data : numpy.ndarray
        Finite numeric array with shape
        ``(n_observations, n_dimensions)``. At least two observations and one
        dimension are required.
    noise : float
        Finite non-negative value added to every diagonal entry of the sample
        covariance matrix.

    Returns
    -------
    functools.partial
        PDF callable accepting one point or a compatible array of points. Its
        fixed ``mean`` and ``cov`` keyword arguments contain the fitted
        Gaussian parameters.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array, values are non-numeric, or ``noise``
        is not a real number.
    ValueError
        If the input rank or size is invalid, values are non-finite, or
        ``noise`` is negative or non-finite.

    Notes
    -----
    The returned partial is directly compatible with
    ``similarity.gaussian_group_similarity``. Singular empirical covariance
    matrices are permitted, which is particularly useful when dimensions
    outnumber observations or some features are linearly dependent.
    """
    from scipy.stats import multivariate_normal

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if isinstance(noise, bool) or not isinstance(
            noise, (int, float, np.integer, np.floating)):
        raise TypeError("noise must be a real number.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and non-negative.")
    if data.ndim != 2:
        raise ValueError(
            "data must have shape (n_observations, n_dimensions)."
        )
    if data.shape[0] < 2:
        raise ValueError("data must contain at least two observations.")
    if data.shape[1] < 1:
        raise ValueError("data must contain at least one dimension.")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must contain numeric values.")

    matrix = np.asarray(data, dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError("data must contain only finite values.")

    dimensionality = matrix.shape[1]
    mean = matrix.mean(axis=0)
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    fitted_covariance = (
        covariance
        + float(noise) * np.eye(dimensionality)
    )
    return partial(
        multivariate_normal.pdf,
        mean=mean,
        cov=fitted_covariance,
        allow_singular=True,
    )


def evaluate_gaussian_density_grid(
        data: np.ndarray,
        noise: float,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        grid_step: float = 0.05) -> tuple[np.ndarray, partial[Any]]:
    """Fit a bivariate Gaussian and evaluate its density on a regular grid.

    The Gaussian mean is the sample centroid. Its covariance is the sample
    covariance regularized by adding ``noise * I``. Grid limits span three
    marginal standard deviations either side of the fitted mean. Each bound is
    rounded outwards to the nearest multiple of ``grid_step`` so independently
    fitted Gaussians lie on the same coordinate lattice.

    Parameters
    ----------
    data : numpy.ndarray
        Finite numeric array with shape ``(n_samples, 2)``. Rows represent
        observations and columns represent the x and y variables.
    noise : float
        Finite, non-negative diagonal covariance regularization.
    xlim : tuple of float or None, optional
        Explicit x-axis limits. When omitted, use the fitted mean plus or minus
        three marginal standard deviations.
    ylim : tuple of float or None, optional
        Explicit y-axis limits. When omitted, use the fitted mean plus or minus
        three marginal standard deviations.
    grid_step : float, default=0.05
        Positive spacing between adjacent x and y coordinates. Grid coordinates
        are integer multiples of this value.

    Returns
    -------
    grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 3)``. ``grid[..., 0]`` contains the x
        coordinates, ``grid[..., 1]`` contains the y coordinates, and
        ``grid[..., 2]`` contains the fitted Gaussian density. Grid sizes depend
        on the fitted spread and ``grid_step``.
    pdf : functools.partial
        Fitted bivariate Gaussian PDF, also suitable for direct point-density
        evaluation and similarity computation.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array or a numeric argument is not a real
        number.
    ValueError
        If the data do not have shape ``(n_samples, 2)``, contain fewer than
        two observations, contain non-finite values, or ``noise`` is negative
        or non-finite, or ``grid_step`` is not finite and positive.

    Notes
    -----
    The standard deviations used for the grid limits include the supplied
    noise because they are derived from the regularized fitted covariance.
    ``allow_singular=True`` permits density evaluation when ``noise`` is zero
    and the empirical covariance is singular.
    """
    pdf = fit_multivariate_gaussian_pdf(data, noise)

    if isinstance(grid_step, bool) or not isinstance(
            grid_step, (int, float, np.integer, np.floating)):
        raise TypeError("grid_step must be a real number.")
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be finite and positive.")
    if data.shape[1] != 2:
        raise ValueError("data must have shape (n_samples, 2).")
    mean = np.asarray(pdf.keywords["mean"], dtype=float)
    fitted_covariance = np.asarray(pdf.keywords["cov"], dtype=float)
    standard_deviations = np.sqrt(np.diag(fitted_covariance))

    def make_discretized_axis(
            centroid: float,
            standard_deviation: float,
            lim: tuple[float, float] | None) -> np.ndarray:
        """Return outward-rounded three-sigma coordinates on one lattice."""
        step = float(grid_step)

        if lim is not None:
            lower_scaled, upper_scaled = lim[0] / step, lim[1] / step
        else:
            lower_scaled = (centroid - 3 * standard_deviation) / step
            upper_scaled = (centroid + 3 * standard_deviation) / step

        # The tolerance prevents an exact mathematical lattice point from
        # being moved by one extra step solely because of floating-point noise.
        lower_tolerance = 1e-12 * max(1.0, abs(lower_scaled))
        upper_tolerance = 1e-12 * max(1.0, abs(upper_scaled))
        lower_index = int(np.floor(lower_scaled + lower_tolerance))
        upper_index = int(np.ceil(upper_scaled - upper_tolerance))

        # A zero-variance axis otherwise has only one coordinate, which cannot
        # form image cells or establish grid spacing for downstream blending.
        if lower_index == upper_index:
            lower_index -= 1
            upper_index += 1

        return np.arange(lower_index, upper_index + 1, dtype=float) * step

    x_points = make_discretized_axis(mean[0], standard_deviations[0], xlim)
    y_points = make_discretized_axis(mean[1], standard_deviations[1], ylim)
    x_grid, y_grid = np.meshgrid(x_points, y_points, indexing="ij")
    grid_points = np.stack((x_grid, y_grid), axis=-1)

    density = pdf(x=grid_points)
    return np.concatenate((grid_points, density[..., np.newaxis]), axis=-1), pdf

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
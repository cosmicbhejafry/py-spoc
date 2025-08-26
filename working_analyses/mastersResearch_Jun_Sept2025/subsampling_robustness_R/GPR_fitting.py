# Importing necessary libraries for data processing
import numpy as np
import pandas as pd
import scipy.stats
from typing import List, Dict, Union, Callable, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from tqdm.notebook import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from Functions_for_Robustness import *


def bootstrap_sample_rows_and_columns(
    df: pd.DataFrame,
    row_fraction: float = 0.9,
    col_fraction: float = 0.9,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Bootstrap sample both rows and columns of a dataframe.
    Rows are sampled with replacement, columns without replacement.
    """
    rng = np.random.default_rng(seed=random_seed)
    # Sample rows with replacement
    n_rows = max(1, int(len(df) * row_fraction))
    sampled_rows = rng.choice(df.index, size=n_rows, replace=True)
    # Sample columns without replacement
    n_cols = max(1, int(len(df.columns) * col_fraction))
    sampled_cols = rng.choice(df.columns, size=n_cols, replace=False)
    return df.loc[sampled_rows, sampled_cols].copy()

def pyspi_calc_bootstrap_rows_and_columns(
    df: pd.DataFrame,
    config_name: str,
    yaml,
    row_fraction: float = 0.9,
    col_fraction: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None
):
    """
    Compute statistics on a bootstrap sample of both rows and columns.
    """
    df = load_data(df)
    bootstrap_df = bootstrap_sample_rows_and_columns(
        df, row_fraction=row_fraction, col_fraction=col_fraction, random_seed=random_seed
    )
    if normalize is not None:
        bootstrap_df = normalize_data(bootstrap_df, method=normalize)
    cfg = Config.from_yaml(config_name, yaml)
    calc = Calculator(bootstrap_df, normalise=False)
    calc.compute(cfg)
    return calc.results

def bootstrap_test_rows_and_columns(
    df: pd.DataFrame,
    config_name,
    yaml,
    calc,
    n_bootstrap: int = 10,
    row_fraction: float = 0.9,
    col_fraction: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_bootstrap_data: bool = True,
    orig_calc_table: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform joint row and column bootstrap sampling test to assess variability of statistics.
    """
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)
    if orig_calc_table is None:
        orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)
    orig_results = {
        reducer: orig_calc_table[reducer].values[0] for reducer in REDUCERS
    }
    bootstrap_values = {reducer: [] for reducer in REDUCERS}
    iter_range = tqdm(range(n_bootstrap), desc="Joint row/col bootstrap sampling")
    for i in iter_range:
        bootstrap_seed = None if random_seed is None else random_seed + i + 1
        bootstrap_calc_table = pyspi_calc_bootstrap_rows_and_columns(
            df, config_name, yaml,
            row_fraction=row_fraction,
            col_fraction=col_fraction,
            normalize=normalize,
            random_seed=bootstrap_seed
        )
        for reducer_name in REDUCERS:
            try:
                bootstrap_value = bootstrap_calc_table[reducer_name].values[0]
                bootstrap_values[reducer_name].append(bootstrap_value)
            except Exception:
                continue
    final_results = {}
    for reducer_name in REDUCERS:
        orig_value = orig_results[reducer_name]
        result = {
            'original_value': orig_value,
            'n_bootstrap': n_bootstrap,
            'row_fraction': row_fraction,
            'col_fraction': col_fraction
        }
        if save_bootstrap_data and bootstrap_values[reducer_name]:
            bootstrap_data = bootstrap_values[reducer_name]
            ci_lower = np.percentile(bootstrap_data, 2.5)
            ci_upper = np.percentile(bootstrap_data, 97.5)
            result.update({
                'bootstrap_mean': np.mean(bootstrap_data),
                'bootstrap_median': np.median(bootstrap_data),
                'bootstrap_std': np.std(bootstrap_data),
                'bootstrap_variance': np.var(bootstrap_data),
                'bootstrap_std_err': np.std(bootstrap_data) / np.sqrt(len(bootstrap_data)),
                'bootstrap_min': np.min(bootstrap_data),
                'bootstrap_max': np.max(bootstrap_data),
                'bootstrap_ci_lower': ci_lower,
                'bootstrap_ci_upper': ci_upper,
                'bootstrap_distribution': bootstrap_data if save_bootstrap_data else None,
                'original_in_ci': ci_lower <= orig_value <= ci_upper,
                'bias': np.mean(bootstrap_data) - orig_value,
                'coefficient_of_variation': np.std(bootstrap_data) / np.abs(np.mean(bootstrap_data)) * 100 if np.mean(bootstrap_data) != 0 else np.nan
            })
        final_results[reducer_name] = result
    return final_results

def variable_scale_array(start=0.001, end=1, n_points=10, log_weight=0.5):
    """
    Generate an array that varies on both logarithmic and linear scales.
    
    Parameters:
    - start: Starting value (default: 0)
    - end: Ending value (default: 1) 
    - n_points: Number of points to generate (default: 100)
    - log_weight: Weight for logarithmic vs linear spacing (0=linear, 1=log, default: 0.5)
    
    Returns:
    - numpy array with variable spacing
    """
    # Create logarithmic spacing (avoiding log(0) by starting from a small value)
    log_start = 1e-6 if start == 0 else start
    log_space = np.logspace(np.log10(log_start), np.log10(end), n_points)
    
    # Create linear spacing
    lin_space = np.linspace(start, end, n_points)
    
    # Combine logarithmic and linear spacing with weighted average
    combined = log_weight * log_space + (1 - log_weight) * lin_space
    
    # Ensure start and end values are exact
    combined[0] = start
    combined[-1] = end
    
    return combined

def n_p_plane_bootstrap_gpr(
    df,
    config_name,
    yaml,
    n_grid=np.linspace(0.5, 1.0, 3),
    p_grid=np.linspace(0.5, 1.0, 3),
    n_bootstrap=10,
    normalize='z-score',
    random_seed=42,
    median_only=False
):
    """
    For each reducer, computes the bootstrap distribution (or median) at each (n, p) grid point,
    then fits and plots a Gaussian Process Regression surface.
    Stores all plots in a dictionary and returns them.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml: YAML string for configuration
    n_grid: np.ndarray, optional
        Grid of row fractions (default: np.linspace(0.5, 1.0, 3))
    p_grid: np.ndarray, optional
        Grid of column fractions (default: np.linspace(0.5, 1.0, 3))
    n_bootstrap: int, optional
        Number of bootstrap samples (default: 10)
    normalize: str, optional
        Normalization method (default: 'z-score')
    random_seed: int, optional
        Random seed for reproducibility (default: 42)
    median_only: bool, optional
        If True, only fit GPR on medians instead of full bootstrap distributions (default: False)

    Returns:
    --------
    Dict[str, Dict[str, Any]], Dict[str, Figure]]
        A dictionary containing results for each reducer and a dictionary of plots.
        Each reducer's results include means, GPR model, score, and grid information.
        The plots dictionary contains 3D surface plots for each reducer.
    """
    calc = pyspi_calc(df, config_name, yaml, normalize=normalize)
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)
    results = {}
    plots = {}

    for reducer_name in REDUCERS:
        print(f"Processing reducer: {reducer_name}")
        means = np.zeros((len(n_grid), len(p_grid)))
        all_bootstrap = [[[] for _ in range(len(p_grid))] for _ in range(len(n_grid))] if not median_only else None

        for i, n_frac in enumerate(n_grid):
            for j, p_frac in enumerate(p_grid):
                bootstrap_result = bootstrap_test_rows_and_columns(
                    df, config_name, yaml, calc=calc,
                    n_bootstrap=n_bootstrap,
                    row_fraction=n_frac,
                    col_fraction=p_frac,
                    normalize=normalize,
                    random_seed=random_seed,
                    save_bootstrap_data=True
                )
                if reducer_name in bootstrap_result and 'bootstrap_distribution' in bootstrap_result[reducer_name]:
                    all_bootstrap[i][j] = bootstrap_result[reducer_name]['bootstrap_distribution']
                    means[i, j] = np.median(all_bootstrap[i][j]) if all_bootstrap[i][j] else np.nan
                else:
                    all_bootstrap[i][j] = []
                    means[i, j] = np.nan

        # GPR fit and plot
        N, P = np.meshgrid(n_grid, p_grid, indexing='ij')
        X = np.column_stack([N.ravel(), P.ravel()])
        y = means.ravel()
        kernel = C(1.0, (1e-3, 1e3)) * RBF([0.1, 0.1], (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True)
        gpr.fit(X, y)
        score = gpr.score(X, y)
        n_fine = np.linspace(n_grid[0], n_grid[-1], 50)
        p_fine = np.linspace(p_grid[0], p_grid[-1], 50)
        N_fine, P_fine = np.meshgrid(n_fine, p_fine, indexing='ij')
        X_fine = np.column_stack([N_fine.ravel(), P_fine.ravel()])
        y_pred, y_std = gpr.predict(X_fine, return_std=True)
        Y_pred = y_pred.reshape(N_fine.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(N_fine, P_fine, Y_pred, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Row fraction (n)')
        ax.set_ylabel('Col fraction (p)')
        ax.set_zlabel(f'GPR value')
        ax.set_title(f'GPR fit for {reducer_name} in (n, p) space')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Predicted value')
        # plt.show()  

        plots[reducer_name] = fig  # Store figure

        results[reducer_name] = {
            'means': means,
            'gpr': gpr,
            'score': score,
            'n_grid': n_grid,
            'p_grid': p_grid
        }
    return results, plots
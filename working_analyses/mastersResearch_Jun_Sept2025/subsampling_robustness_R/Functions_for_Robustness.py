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
from pyspoc import Calculator, Config

def strip_common_prefix(data):
    # prefix = "C:/Rishi/Maths/Research Project/py-spoc/working_analyses/hjResearch_2025/summary_stats/"
    # result = []
    # for x in data:
    #     if isinstance(x, tuple):
    #         # Remove prefix from each element in the tuple
    #         result.append(tuple(
    #             xi.replace(prefix, '') if isinstance(xi, str) else xi
    #             for xi in x
    #         ))
    #     elif isinstance(x, str):
    #         result.append(x.replace(prefix, ''))
    #     else:
    #         result.append(x)
    return data

def load_data(df) -> pd.DataFrame:
    """
    Drop non-numeric columns and convert to numeric.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change.
    Returns:
    --------
    pd.DataFrame
        The loaded data.
    """
    if type(df) is not pd.DataFrame:
        df = pd.DataFrame(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]  
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    return df


def normalize_data(data: pd.DataFrame, method: str = 'z-score') -> pd.DataFrame:
    """
    Normalize the data using the specified method.

    Parameters:
    -----------
    data : pd.DataFrame
        The data to normalize.
    method : str, optional
        The method to use for scaling. Options are:
        - 'z-score': Z-Score Normalization (StandardScaler)
        - 'min-max': Min-Max Normalization (MinMaxScaler)
        - 'robust': Robust Scaling using median and IQR (RobustScaler)
        - 'none': No scaling, returns data unchanged.

    Returns:
    --------
    pd.DataFrame
        The normalized (or unchanged) data.
    """
    if method == 'none':
        return data.copy()

    normalized = data.copy()

    # Select the appropriate scaler 
    if method == 'z-score':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Choose from 'z-score', 'min-max', 'robust', or 'none'.")

    # Apply the scaler to the data
    normalized_values = scaler.fit_transform(normalized)
    normalized = pd.DataFrame(normalized_values,
                              index=normalized.index,
                              columns=normalized.columns)

    return normalized



def shuffle_data(df: pd.DataFrame, random_seed=None) -> pd.DataFrame:
    """
    Returns a new DataFrame with each column's elements shuffled independently.

    Parameters:
    - df: The input pandas DataFrame.
    - random_seed: Optional seed for reproducibility.

    Returns:
    - A new DataFrame with columns shuffled.
    """
    rng = np.random.default_rng(seed=random_seed)
    return pd.DataFrame(
        [rng.permutation(df[col]) for col in df.columns],
        index=df.columns,
        columns=df.index
    ).T

def perturb_dataframe(df, scale_factor=0.1, random_seed=None):
    """    
    Perturb the values in a DataFrame by adding Gaussian noise scaled by the column variances.
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing numeric values
    scale_factor : float, default=0.1
        Scaling factor for the perturbation
    random_seed : int, optional
        Random seed for reproducible results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with perturbed values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate column variances
    col_variances = df.var(axis=0, ddof=1)
    col_variances = np.maximum(col_variances, 1e-6)  
    # Calculate perturbation standard deviations
    perturbation_stds = scale_factor * np.sqrt(col_variances)
    
    # Create perturbation matrix
    perturbations = np.random.normal(0, 1, size=df.shape)
    
    # Scale perturbations 
    for j in range(df.shape[1]):
        perturbations[:, j] *= perturbation_stds.iloc[j]
    
    # Add perturbations to original data
    perturbed_df = df + perturbations
    
    return perturbed_df

def apply_reducer(calc, reducer) -> float:
    """
    Apply a specific reduced statistic to a matrix and get a value, useful for iterating
    Parameters:
    -----------
    calc : Calculator
        The calculator instance.
    reducer : gets reducer value from calc.results

    Returns:
    --------
    float:
        The reduced value.
    """
    return calc.results[reducer].values[0]


def pyspi_calc(df,config_name, yaml, normalize=None, shuffle=False, perturb=False,scale_factor=0.1,random_seed=None):
    """
    Parameters:
    - df: DataFrame containing the data to change
    - config_name: Name of the configuration file
    - yaml: YAML string for configuration
    - normalize: Method to normalize data (default: None)
    - shuffle: Whether to randomly shuffle rows (default: False)
    - random_seed: Seed for reproducible shuffling (default: None)

    Returns:
    - Calc results table
    """
    # Load data 
    df = load_data(df)

    # If shuffle is requested, shuffle the data
    if shuffle:
        df = shuffle_data(df, random_seed=random_seed)

    if perturb:
        df = perturb_dataframe(df, scale_factor=scale_factor, random_seed=random_seed)

    if normalize is not None:
        df = normalize_data(df, method=normalize)
        
    cfg = Config.from_yaml(config_name, yaml)
    # Use Calculator class to compute the statistics
    # pass the dataframe or numpy array
    # constructs a 'Calculator' instance
    calc = Calculator(df, normalise=False)
    # run the compute function - applies functions in the config (i.e. in Statistic and Reducer) to the Data
    calc.compute(cfg)
    return calc.results

def permutation_test(
    df: pd.DataFrame,
    config_name,
    yaml,
    calc,
    n_permutations: int = 1000,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_permutation_data: bool = True,
    orig_calc_table: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform a permutation test to assess statistical significance and calculate variance.
    Parameters:
    -----------
    df : pd.DataFrame
    df: DataFrame containing the data to change
    config_name: Name of the configuration file
    yaml: YAML string for configuration
    n_permutations: number of permutations in our test (default: 1000)
    normalize: Method to normalize data (default: None)
    random_seed: Seed for reproducible shuffling (default: None)
    save_permutation_data: Whether to save permutation data (default: True)
    orig_calc_table: Check to see if orig_calc_table exists
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        A dictionary containing the original values, p-values, and permutation statistics for each reducer.
    """
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)
    # Use provided orig_calc_table or compute
    if orig_calc_table is None:
        orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)

    orig_results = {
        reducer: orig_calc_table[reducer].values[0] for reducer in REDUCERS
    }
    
    perm_values = {reducer: [] for reducer in REDUCERS}

    # Run permutations with progress bar
    iter_range = tqdm(range(n_permutations))
    for i in iter_range:
        perm_seed = None if random_seed is None else random_seed + i + 1
        perm_calc_table = pyspi_calc(df, config_name,yaml, normalize=normalize, shuffle=True, random_seed=perm_seed)

        for reducer in REDUCERS:
            perm_value = perm_calc_table[reducer].values[0]
            perm_values[reducer].append(perm_value)

    # Calculate p-values, variances, and add to results
    perm_data = pd.DataFrame(perm_values)
    final_results = {}
    for reducer in REDUCERS:
        # Access the specific reducer's data from the perm_data DataFrame
        perm_data_series = perm_data[reducer]
        orig_value = orig_results[reducer]
        n = len(perm_data_series) # Use the length of the collected data
        
        # Calculate permutation percentile for p-value
        p_left_adj = (np.sum(perm_data_series <= orig_value) + 1) / (n + 1)
        p_right_adj = (np.sum(perm_data_series >= orig_value) + 1) / (n + 1)
        p_value = 2 * min(p_left_adj, p_right_adj)
        
        result = {'value': orig_value,
              'p_value': p_value}

        # Update results with permutation statistics
        result.update({
            'permutation_mean': np.mean(perm_data_series),
            'permutation_median': np.median(perm_data_series),
            'permutation_std': np.std(perm_data_series),
            'permutation_std_err': np.std(perm_data_series) / np.sqrt(n),
            'permutation_variance': np.var(perm_data_series),
            'permutation_min': np.min(perm_data_series),
            'permutation_max': np.max(perm_data_series),
            'permutation_distribution': perm_data_series.tolist() if save_permutation_data else None, # Store as list if needed
            'permutation_percentile_rank': p_value
        })

        final_results[reducer] = result
    return final_results


def plot_permutation_distribution(results: Dict, reducer: str,
                                 figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Plot the distribution of permutation test values

    Parameters:
    -----------
    results : Dict
        Dictionary containing permutation test results.
    reducer : str
        Name of the reducer
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (10, 6)

    Returns:
    --------
    Figure
        Matplotlib figure object.
    """

    # Extract data
    result = results[reducer]
    perm_values = result['permutation_distribution']
    orig_value = result['value']
    p_value = result['p_value']

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram of permutation values
    n, bins, patches = ax.hist(perm_values, bins=30, alpha=0.7, color='skyblue',
                              edgecolor='black', density=True)

    # Add kernel density estimate
    sns.kdeplot(perm_values, color='navy', ax=ax, linewidth=2)

    # Add vertical line for original value
    ax.axvline(x=orig_value, color='red', linestyle='--', linewidth=2,
              label=f'Original value: {orig_value:.4f}')

    # Add permutation statistics
    stats_text = (
        f"Permutation Stats:\n"
        f"Mean: {result['permutation_mean']:.4f}\n"
        f"Median: {result['permutation_median']:.4f}\n"
        f"Std Dev: {result['permutation_std']:.4f}\n"
        f"Std Err: {result['permutation_std_err']:.4f}\n"
        f"p-value: {p_value:.4f}"
    )

    # Add the stats text as an annotation
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax.set_xlabel(f'{reducer} Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Permutation Distribution for {reducer}')

    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def plot_all_permutation_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
    """
    Plot permutation distributions for all reducers in the results dictionary.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing permutation test results for multiple reducers.
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (12, 8)
    Returns:
    --------
    Dict[str, Figure]
        Dictionary mapping reducer names to their corresponding matplotlib Figure objects.
    """

    figures = {}
    for reducer_name, reducer_results in results.items():
        # Convert reducer_name to string for safe dict key
        key = str(reducer_name)
        if (isinstance(reducer_results, dict) and
            'permutation_distribution' in reducer_results and
            'value' in reducer_results):
            try:
                fig = plot_permutation_distribution(results, reducer_name, figsize=figsize)
                figures[key] = fig
            except Exception as e:
                print(f"Error plotting {reducer_name}: {e}")
    return figures

def bootstrap_sample_data(df: pd.DataFrame, sample_fraction: float = 0.9, random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create a bootstrap sample from the dataframe by sampling a fraction of rows with replacement.

    Parameters:
    -----------
    df : pd.DataFrame
        The original dataframe
    sample_fraction : float, optional
        Fraction of rows to sample
    random_seed : Optional[int], optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Bootstrap sampled dataframe (rows)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate number of rows to sample
    n_samples = max(1, int(len(df) * sample_fraction))

    # Sample row indices with replacement
    sampled_indices = np.random.choice(df.index, size=n_samples, replace=True)

    # Create bootstrap sample with selected rows
    bootstrap_df = df.loc[sampled_indices].copy()
    return bootstrap_df


def pyspi_calc_bootstrap(df: pd.DataFrame,config_name: str, yaml, normalize: Optional[str] = None,
                        sample_fraction: float = 0.9, random_seed: Optional[int] = None):
    """
    Perform bootstrap sampling and calculate statistics using pyspoc
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml : YAML string for configuration
    normalize : Optional[str], optional
        Method to normalize data (default: None)
    sample_fraction : float, optional
        Fraction of rows to sample for bootstrap (default: 0.9)
    random_seed : Optional[int], optional
        Seed for reproducible sampling (default: None)

    Returns:
    --------
    Dictionary of calculated statistics
    """
    # Load data 
    df = load_data(df)

    # Create bootstrap sample
    bootstrap_df = bootstrap_sample_data(df, sample_fraction=sample_fraction, random_seed=random_seed)
    if normalize is not None:
        bootstrap_df = normalize_data(bootstrap_df, method=normalize)
    # Use Calculator class to compute the statistics
    cfg = Config.from_yaml(config_name, yaml)
    calc = Calculator(bootstrap_df, normalise=False)
    calc.compute(cfg)
    return calc.results


def bootstrap_test(
    df: pd.DataFrame,
    config_name,
    yaml,
    calc,
    n_bootstrap: int = 10,
    sample_fraction: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_bootstrap_data: bool = True,
    orig_calc_table: Optional[Dict] = None,  
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Perform bootstrap sampling test to assess variability of statistics.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml: YAML string for configuration
    calc: Calculator instance with original results
    n_bootstrap: int, optional
        Number of bootstrap samples (default: 10)
    sample_fraction: float, optional
        Fraction of rows to sample for bootstrap (default: 0.9)
    normalize: Optional[str], optional
        Method to normalize data (default: None)
    random_seed: Optional[int], optional
        Seed for reproducible sampling (default: None)
    save_bootstrap_data: bool, optional
        Whether to save bootstrap data (default: True)
    orig_calc_table: Optional[Dict], optional
        Original calculation table if available

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        A dictionary containing the original values, bootstrap statistics, and p-values for each reducer.
    """
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)
    if orig_calc_table is None:
        orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)
    orig_results = {
        reducer: orig_calc_table[reducer].values[0] for reducer in REDUCERS
    }
    bootstrap_values = {reducer: [] for reducer in REDUCERS}
    iter_range = tqdm(range(n_bootstrap), desc="Bootstrap sampling")
    for i in iter_range:
        bootstrap_seed = None if random_seed is None else random_seed + i + 1
        bootstrap_calc_table = pyspi_calc_bootstrap(df, config_name, yaml, normalize=normalize,
                                                   sample_fraction=sample_fraction, random_seed=bootstrap_seed)
        for reducer_name in REDUCERS:
            try:
                bootstrap_value = bootstrap_calc_table[reducer_name].values[0]
                bootstrap_values[reducer_name].append(bootstrap_value)
            except:
                continue
    final_results = {}
    for reducer_name in REDUCERS:
        orig_value = orig_results[reducer_name]
        result = {
            'original_value': orig_value,
            'n_bootstrap': n_bootstrap,
            'sample_fraction': sample_fraction
        }
        if save_bootstrap_data and bootstrap_values[reducer_name]:
            bootstrap_data = bootstrap_values[reducer_name]
            ci_lower = np.percentile(bootstrap_data, 2.5)
            ci_upper = np.percentile(bootstrap_data, 97.5)
            
            n_bootstrap_samples = len(bootstrap_data)
            p_left_adj = (np.sum(bootstrap_data <= orig_value) + 1) / (n_bootstrap_samples + 1)
            p_right_adj = (np.sum(bootstrap_data >= orig_value) + 1) / (n_bootstrap_samples + 1)
            p_val = 2 * min(p_left_adj, p_right_adj)
            result.update({
                'bootstrap_mean': np.mean(bootstrap_data),
                'bootstrap_median': np.median(bootstrap_data),
                'bootstrap_std': np.std(bootstrap_data, ddof=1),
                'bootstrap_variance': np.var(bootstrap_data, ddof=1),
                'bootstrap_std_err': np.std(bootstrap_data, ddof=1) / np.sqrt(len(bootstrap_data)),
                'bootstrap_min': np.min(bootstrap_data),
                'bootstrap_max': np.max(bootstrap_data),
                'bootstrap_ci_lower': ci_lower,
                'bootstrap_ci_upper': ci_upper,
                'bootstrap_distribution': bootstrap_data if save_bootstrap_data else None,
                'original_in_ci': ci_lower <= orig_value <= ci_upper,
                'bias': np.mean(bootstrap_data) - orig_value,
                'coefficient_of_variation': np.std(bootstrap_data, ddof=1) / np.abs(np.mean(bootstrap_data)) * 100
                    if np.mean(bootstrap_data) != 0 else np.nan,
                'bootstrap_p_value': p_val  
            })
        final_results[reducer_name] = result
    return final_results


def plot_bootstrap_distribution(results: Dict, reducer,
                               figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Plot the distribution of bootstrap test values

    Parameters:
    -----------
    results : Dict
        Dictionary containing bootstrap test results.
    reducer : str
        Name of the reducer
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (10, 6)

    Returns:
    --------
    Figure
        Matplotlib figure object.
    """
    # Extract data
    result = results[reducer]
    bootstrap_values = result['bootstrap_distribution']
    orig_value = result['original_value']
    ci_lower = result['bootstrap_ci_lower']
    ci_upper = result['bootstrap_ci_upper']

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram of bootstrap values
    n, bins, patches = ax.hist(bootstrap_values, bins=30, alpha=0.7, color='lightgreen',
                              edgecolor='black', density=True)

    # Add kernel density
    sns.kdeplot(bootstrap_values, color='darkgreen', ax=ax, linewidth=2)

    # Add vertical line for original value
    ax.axvline(x=orig_value, color='red', linestyle='--', linewidth=2,
              label=f'Original value: {orig_value:.4f}')

    # Add confidence interval lines
    ax.axvline(x=ci_lower, color='orange', linestyle=':', linewidth=2,
              label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(x=ci_upper, color='orange', linestyle=':', linewidth=2)

    # Fill confidence interval area
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange')

    # Add bootstrap statistics
    stats_text = (
        f"Bootstrap Stats:\n"
        f"Mean: {result['bootstrap_mean']:.4f}\n"
        f"Median: {result['bootstrap_median']:.4f}\n"
        f"Std Dev: {result['bootstrap_std']:.4f}\n"
        f"Bias: {result['bias']:.4f}\n"
        f"Std Err: {result['bootstrap_std_err']:.4f}\n"
        f"CV: {result['coefficient_of_variation']:.2f}%\n"
        f"Original in CI: {result['original_in_ci']}\n"
        f"p-value: {result.get('bootstrap_p_value', float('nan')):.4f}"
    )

    # Add the stats text as an annotation
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax.set_xlabel(f'{reducer} Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Row Bootstrap Distribution for {reducer}')

    # Add subtitle with sample info
    subtitle = f'Row Bootstrap samples: {result["n_bootstrap"]}, Sample fraction: {result["sample_fraction"]:.1%}'
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='center', style='italic')

    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def plot_all_bootstrap_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
    """
    Plot bootstrap distributions for all reducers in the results dictionary.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing bootstrap test results for multiple reducers.
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (12, 8)

    Returns:
    --------
    Dict[str, Figure]
        Dictionary mapping reducer names to their corresponding matplotlib Figure objects.
    """
    figures = {}
    for reducer_name, _ in results.items():
        key = str(reducer_name)
        try:
            fig = plot_bootstrap_distribution(results, reducer_name, figsize=figsize)
            figures[key] = fig
        except Exception as e:
            print(f"Error plotting bootstrap {reducer_name}: {e}")
    return figures


def bootstrap_sample_columns(df: pd.DataFrame, sample_fraction: float = 0.9, random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create a bootstrap sample from the dataframe by sampling a fraction of columns without replacement.
    Parameters:
    -----------
    df : pd.DataFrame
        The original dataframe
    sample_fraction : float, optional
        Fraction of columns to sample (default: 0.9)
    random_seed : Optional[int], optional
        Seed for reproducible sampling (default: None)

    Returns:
    --------
    pd.DataFrame
        Bootstrap sampled dataframe (columns)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = max(1, int(len(df.columns) * sample_fraction))
    sampled_columns = np.random.choice(df.columns, size=n_samples, replace=False)
    bootstrap_df = df.loc[:, sampled_columns].copy()
    return bootstrap_df

def pyspi_calc_bootstrap_columns(df: pd.DataFrame,config_name: str, yaml, normalize: Optional[str] = None,
                        sample_fraction: float = 0.9, random_seed: Optional[int] = None):
    """
    Perform bootstrap sampling on columns and calculate statistics using pyspoc.
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml : YAML string for configuration
    normalize : Optional[str], optional
        Method to normalize data (default: None)
    sample_fraction : float, optional
        Fraction of columns to sample for bootstrap (default: 0.9)
    random_seed : Optional[int], optional
        Seed for reproducible sampling (default: None)

    Returns:
    --------
    Dictionary of calculated statistics
    """
    # Load data with optional normalization
    df = load_data(df)

    # Create bootstrap sample
    bootstrap_df = bootstrap_sample_columns(df, sample_fraction=sample_fraction, random_seed=random_seed)
    if normalize is not None:
        bootstrap_df = normalize_data(bootstrap_df, method=normalize)
    # Use Calculator class to compute the statistics
    cfg = Config.from_yaml(config_name, yaml)
    calc = Calculator(bootstrap_df, normalise=False)
    calc.compute(cfg)
    return calc.results


def bootstrap_test_columns(
    df: pd.DataFrame,
    config_name,
    yaml,
    calc,
    n_bootstrap_columns: int = 10,
    sample_fraction_columns: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_bootstrap_data: bool = True,
    orig_calc_table: Optional[Dict] = None,  
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Perform bootstrap sampling test to assess variability of statistics.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml: YAML string for configuration
    calc: Calculator instance with original results
    n_bootstrap_columns: int, optional
        Number of bootstrap samples (default: 10)
    sample_fraction_columns: float, optional
        Fraction of columns to sample for bootstrap (default: 0.9)
    normalize: Optional[str], optional
        Method to normalize data (default: None)
    random_seed: Optional[int], optional
        Seed for reproducible sampling (default: None)
    save_bootstrap_data: bool, optional
        Whether to save bootstrap data (default: True)
    orig_calc_table: Optional[Dict], optional
        Original calculation table if available

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        A dictionary containing the original values, bootstrap statistics, and p-values for each reducer.
    """
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)

    # Use provided orig_calc_table or compute if not given
    if orig_calc_table is None:
        orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)
    
    orig_results = {
        reducer: orig_calc_table[reducer].values[0] for reducer in REDUCERS
    }
    
    # Store bootstrap values for variance calculation and plotting
    bootstrap_values = {reducer: [] for reducer in REDUCERS}

    # Run bootstrap samples with progress bar
    iter_range = tqdm(range(n_bootstrap_columns), desc="Bootstrap sampling")
    for i in iter_range:
        bootstrap_seed = None if random_seed is None else random_seed + i + 1
        bootstrap_calc_table = pyspi_calc_bootstrap_columns(
            df, config_name, yaml,
            normalize=normalize,
            sample_fraction=sample_fraction_columns,
            random_seed=bootstrap_seed
        )

        for reducer_name in REDUCERS:
            try:
                bootstrap_value = bootstrap_calc_table[reducer_name].values[0]
                bootstrap_values[reducer_name].append(bootstrap_value)
            except:
                continue
            
    # Calculate bootstrap statistics and add to results
    final_results = {}
    for reducer_name in REDUCERS:
        orig_value = orig_results[reducer_name]
        result = {
            'original_value': orig_value,
            'n_bootstrap': n_bootstrap_columns,
            'sample_fraction': sample_fraction_columns
        }

        if save_bootstrap_data and bootstrap_values[reducer_name]:
            bootstrap_data = np.array(bootstrap_values[reducer_name])
            ci_lower = np.percentile(bootstrap_data, 2.5)
            ci_upper = np.percentile(bootstrap_data, 97.5)

            n_bootstrap_samples = len(bootstrap_data)
            p_left_adj = (np.sum(bootstrap_data <= orig_value) + 1) / (n_bootstrap_samples + 1)
            p_right_adj = (np.sum(bootstrap_data >= orig_value) + 1) / (n_bootstrap_samples + 1)
            p_val = 2 * min(p_left_adj, p_right_adj)

            result.update({
                'bootstrap_mean': np.mean(bootstrap_data),
                'bootstrap_median': np.median(bootstrap_data),
                'bootstrap_std': np.std(bootstrap_data, ddof=1),
                'bootstrap_variance': np.var(bootstrap_data, ddof=1),
                'bootstrap_std_err': np.std(bootstrap_data, ddof=1) / np.sqrt(len(bootstrap_data)),
                'bootstrap_min': np.min(bootstrap_data),
                'bootstrap_max': np.max(bootstrap_data),
                'bootstrap_ci_lower': ci_lower,
                'bootstrap_ci_upper': ci_upper,
                'bootstrap_distribution': bootstrap_data if save_bootstrap_data else None,
                'original_in_ci': ci_lower <= orig_value <= ci_upper,
                'bias': np.mean(bootstrap_data) - orig_value,
                'coefficient_of_variation': np.std(bootstrap_data, ddof=1) / np.abs(np.mean(bootstrap_data)) * 100
                    if np.mean(bootstrap_data) != 0 else np.nan,
                'bootstrap_p_value': p_val
            })

        final_results[reducer_name] = result

    return final_results


def plot_bootstrap_distribution_columns(results: Dict, reducer,
                               figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Plot the distribution of bootstrap test values for columns

    Parameters:
    -----------
    results : Dict
        Dictionary containing bootstrap test results.
    reducer : str
        Name of the reducer
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (10, 6)

    Returns:
    --------
    Figure
        Matplotlib figure object.
    """
    # Extract data
    result = results[reducer]
    bootstrap_values = result['bootstrap_distribution']
    orig_value = result['original_value']
    ci_lower = result['bootstrap_ci_lower']
    ci_upper = result['bootstrap_ci_upper']

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram of bootstrap values
    n, bins, patches = ax.hist(bootstrap_values, bins=30, alpha=0.7, color='lightgreen',
                              edgecolor='black', density=True)

    # Add kernel density
    sns.kdeplot(bootstrap_values, color='darkgreen', ax=ax, linewidth=2)

    # Add vertical line for original value
    ax.axvline(x=orig_value, color='red', linestyle='--', linewidth=2,
              label=f'Original value: {orig_value:.4f}')

    # Add confidence interval lines
    ax.axvline(x=ci_lower, color='orange', linestyle=':', linewidth=2,
              label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(x=ci_upper, color='orange', linestyle=':', linewidth=2)

    # Fill confidence interval area
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange')

    # Add bootstrap statistics
    stats_text = (
        f"Bootstrap Stats:\n"
        f"Mean: {result['bootstrap_mean']:.4f}\n"
        f"Median: {result['bootstrap_median']:.4f}\n"
        f"Std Dev: {result['bootstrap_std']:.4f}\n"
        f"Bias: {result['bias']:.4f}\n"
        f"Std Err: {result['bootstrap_std_err']:.4f}\n"
        f"CV: {result['coefficient_of_variation']:.2f}%\n"
        f"Original in CI: {result['original_in_ci']}\n"
        f"p-value: {result.get('bootstrap_p_value', float('nan')):.4f}"
    )

    # Add the stats text as an annotation
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax.set_xlabel(f'{reducer} Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Column Bootstrap Distribution for {reducer}')

    # Add subtitle with sample info
    subtitle = f'Column Bootstrap samples: {result["n_bootstrap"]}, Sample fraction: {result["sample_fraction"]:.1%}'
    ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='center', style='italic')

    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def plot_all_bootstrap_distributions_columns(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
    """
    Plot bootstrap distributions for all reducers in the results dictionary (for columns).
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing bootstrap test results for multiple reducers.
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (12, 8)

    Returns:
    --------
    Dict[str, Figure]
        Dictionary mapping reducer names to their corresponding matplotlib Figure objects.
    """
    figures = {}
    for reducer_name, reducer_results in results.items():
        key = str(reducer_name)
        try:
            fig = plot_bootstrap_distribution_columns(results, reducer_name, figsize=figsize)
            figures[key] = fig
        except Exception as e:
            print(f"Error plotting bootstrap {reducer_name}: {e}")
    return figures


def perturbation_test(
    df: pd.DataFrame,
    config_name,
    yaml,
    calc,
    n_perturbations: int = 1000,
    scale_factor: float = 0.1,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_perturbation_data: bool = True,
    orig_calc_table: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform a perturbation test to assess statistical significance and calculate variance.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml: YAML string for configuration
    calc: Calculator instance with original results
    n_perturbations: int, optional
        Number of perturbation samples (default: 1000)
    scale_factor: float, optional
        Scale factor for perturbation (default: 0.1)
    normalize: Optional[str], optional
        Method to normalize data (default: None)
    random_seed: Optional[int], optional
        Seed for reproducible sampling (default: None)
    save_perturbation_data: bool, optional
        Whether to save perturbation data (default: True)
    orig_calc_table: Optional[Dict], optional
        Original calculation table if available

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        A dictionary containing the original values, perturbation statistics, and p-values for each reducer
    """
    REDUCERS = list(calc.columns.values)
    REDUCERS = strip_common_prefix(REDUCERS)

    # Use provided orig_calc_table or compute
    if orig_calc_table is None:
        orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)

    orig_results = {
        reducer: orig_calc_table[reducer].values[0] for reducer in REDUCERS
    }
    
    perturb_values = {reducer: [] for reducer in REDUCERS}

    # Run perturbations with progress bar
    iter_range = tqdm(range(n_perturbations), desc="Perturbation sampling")
    for i in iter_range:
        perturb_seed = None if random_seed is None else random_seed + i + 1
        perturb_calc_table = pyspi_calc(
            df, config_name, yaml,
            normalize=normalize,
            perturb=True,
            scale_factor=scale_factor,
            random_seed=perturb_seed
        )

        for reducer in REDUCERS:
            perturb_value = perturb_calc_table[reducer].values[0]
            perturb_values[reducer].append(perturb_value)

    # Create the DataFrame of perturbation results
    perturb_data = pd.DataFrame(perturb_values)

    # Calculate stats for each reducer
    final_results = {}
    for reducer in REDUCERS:
        perturb_series = np.array(perturb_data[reducer])
        orig_value = orig_results[reducer]
        n = len(perturb_series)

        diffs = np.abs(perturb_series - np.mean(perturb_series))
        orig_dist = np.abs(orig_value - np.mean(perturb_series))
        p_value = (1 + np.sum(diffs >= orig_dist)) / (n + 1)

        # 95% confidence interval
        ci_lower = np.percentile(perturb_series, 2.5)
        ci_upper = np.percentile(perturb_series, 97.5)

        result = {
            'value': orig_value,
            'p_value': p_value,
            'perturbation_mean': np.mean(perturb_series),
            'perturbation_median': np.median(perturb_series),
            'perturbation_std': np.std(perturb_series, ddof=1),
            'perturbation_variance': np.var(perturb_series, ddof=1),
            'perturbation_std_err': np.std(perturb_series, ddof=1) / np.sqrt(n),
            'perturbation_min': np.min(perturb_series),
            'perturbation_max': np.max(perturb_series),
            'perturbation_distribution': perturb_series.tolist() if save_perturbation_data else None,
            'perturbation_ci_lower': ci_lower,
            'perturbation_ci_upper': ci_upper,
            'original_in_ci': ci_lower <= orig_value <= ci_upper
        }

        final_results[reducer] = result

    return final_results



def plot_perturbation_distribution(results: Dict, reducer: str,
                                 figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """
    Plot the distribution of perturbation test values, including a 95% confidence interval.
    Parameters:
    -----------
    results : Dict
        Dictionary containing perturbation test results.
    reducer : str
        Name of the reducer
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (10, 6)

    Returns:
    --------
    Figure
        Matplotlib figure object.         
    """
    # Extract data
    result = results[reducer]
    perturb_values = result['perturbation_distribution']
    orig_value = result['value']
    p_value = result['p_value']
    ci_lower = result.get('perturbation_ci_lower', None)
    ci_upper = result.get('perturbation_ci_upper', None)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram of permutation values
    n, bins, patches = ax.hist(perturb_values, bins=30, alpha=0.7, color='skyblue',
                              edgecolor='black', density=True)

    # Add kernel density estimate
    sns.kdeplot(perturb_values, color='navy', ax=ax, linewidth=2)

    # Add vertical line for original value
    ax.axvline(x=orig_value, color='red', linestyle='--', linewidth=2,
              label=f'Original value: {orig_value:.4f}')

    # Add confidence interval lines if available
    if ci_lower is not None and ci_upper is not None:
        ax.axvline(x=ci_lower, color='orange', linestyle=':', linewidth=2,
                  label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(x=ci_upper, color='orange', linestyle=':', linewidth=2)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange')

    # Add perturbation statistics
    stats_text = (
        f"Perturbation Stats:\n"
        f"Mean: {result['perturbation_mean']:.4f}\n"
        f"Median: {result['perturbation_median']:.4f}\n"
        f"Std Dev: {result['perturbation_std']:.4f}\n"
        f"Std Err: {result['perturbation_std_err']:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        f"Original in CI: {result.get('original_in_ci', False)}"
    )

    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax.set_xlabel(f'{reducer} Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Perturbation Distribution for {reducer}')

    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def plot_all_perturbation_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
    """    
    Plot perturbation distributions for all reducers in the results dictionary.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing perturbation test results for multiple reducers.
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (12, 8)

    Returns:
    --------
    Dict[str, Figure]
        Dictionary mapping reducer names to their corresponding matplotlib Figure objects.
    """
    figures = {}
    for reducer_name, reducer_results in results.items():
        # Convert reducer_name to string for safe dict key
        key = str(reducer_name)
        if (isinstance(reducer_results, dict) and
            'perturbation_distribution' in reducer_results and
            'value' in reducer_results):
            try:
                fig = plot_perturbation_distribution(results, reducer_name, figsize=figsize)
                figures[key] = fig
            except Exception as e:
                print(f"Error plotting {reducer_name}: {e}")
    return figures


def run_analysis(
    df: pd.DataFrame,
    config_name,
    yaml,
    n_permutations: int = 1000,
    n_bootstrap_row: int = 1000,
    n_bootstrap_col: int = 1000,
    n_perturbations: int = 1000,
    scale_factor: float = 0.1,
    bootstrap_sample_fraction_row: float = 0.9,
    bootstrap_sample_fraction_col: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    pdf_output: Optional[str] = None
) -> Dict:
    """
    Run a comprehensive permutation and bootstrap analysis with combined visualizations and detailed summaries.
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to change
    config_name: str
        Name of the configuration file
    yaml: YAML string for configuration
    n_permutations: int, optional
        Number of permutations (default: 1000)
    n_bootstrap_row: int, optional
        Number of row bootstrap samples (default: 1000)
    n_bootstrap_col: int, optional
        Number of column bootstrap samples (default: 1000)
    n_perturbations: int, optional
        Number of perturbations (default: 1000)
    scale_factor: float, optional
        Scale factor for perturbations (default: 0.1)
    bootstrap_sample_fraction_row: float, optional
        Fraction of rows to sample for bootstrap (default: 0.9)
    bootstrap_sample_fraction_col: float, optional
        Fraction of columns to sample for bootstrap (default: 0.9)
    normalize: Optional[str], optional
        Method to normalize data (default: None)
    random_seed: Optional[int], optional
        Seed for reproducible sampling (default: None)
    pdf_output: Optional[str], optional
        Path to save the PDF output with plots (default: None)

    Returns:
    --------
    Dict
        A dictionary containing all results, including p-values, plots, and metadata.
    """
    print(f"{n_permutations} permutations, {n_bootstrap_row} row bootstrap samples, {n_bootstrap_col} column bootstrap samples, {n_perturbations} perturbations")

    # Calculate orig_calc_table once
    orig_calc_table = pyspi_calc(df, config_name, yaml, normalize=normalize)

    # Permutation test
    print("Permutation analysis:")
    permutation_results = permutation_test(
        df,
        config_name,
        yaml,
        orig_calc_table,
        n_permutations=n_permutations,
        normalize=normalize,
        random_seed=random_seed,
        save_permutation_data=True,
        orig_calc_table=orig_calc_table,
    )

    # Row bootstrap test
    print("Row bootstrap analysis:")
    bootstrap_results = bootstrap_test(
        df,
        config_name,
        yaml,
        orig_calc_table,
        n_bootstrap=n_bootstrap_row,
        sample_fraction=bootstrap_sample_fraction_row,
        normalize=normalize,
        random_seed=random_seed,
        save_bootstrap_data=True,
        orig_calc_table=orig_calc_table,
    )

    print("Column bootstrap analysis (random columns):")
    column_bootstrap_results_random = bootstrap_test_columns(
        df,
        config_name,
        yaml,
        orig_calc_table,
        n_bootstrap_columns=n_bootstrap_col,
        sample_fraction_columns=bootstrap_sample_fraction_col,
        normalize=normalize,
        random_seed=random_seed,
        save_bootstrap_data=True,
        orig_calc_table=orig_calc_table,
    )

    # Perturbation test
    print("Perturbation analysis:")
    perturbation_results = perturbation_test(
        df,
        config_name,
        yaml,
        orig_calc_table,
        n_perturbations=n_perturbations,
        scale_factor=scale_factor,
        normalize=normalize,
        random_seed=random_seed,
        save_perturbation_data=True,
        orig_calc_table=orig_calc_table,
    )

    # Combine results
    results = {
        'permutation_results': permutation_results,
        'row_bootstrap_results': bootstrap_results,
        'column_bootstrap_results_random': column_bootstrap_results_random,
        'perturbation_results': perturbation_results,
        'metadata': {
            'n_permutations': n_permutations,
            'n_bootstrap_row': n_bootstrap_row,
            'n_bootstrap_col': n_bootstrap_col,
            'n_perturbations': n_perturbations,
            'scale_factor': scale_factor,
            'bootstrap_sample_fraction_row': bootstrap_sample_fraction_row,
            'bootstrap_sample_fraction_col': bootstrap_sample_fraction_col,
            'normalize': normalize,
            'random_seed': random_seed
        }
    }

    print("Generating permutation distribution plots...")
    permutation_distribution_plots = plot_all_permutation_distributions(permutation_results)
    print("Generating row bootstrap distribution plots...")
    row_bootstrap_plots = plot_all_bootstrap_distributions(bootstrap_results)
    print("Generating column bootstrap (random) distribution plots...")
    column_bootstrap_random_plots = plot_all_bootstrap_distributions_columns(column_bootstrap_results_random)
    print("Generating perturbation distribution plots...")
    perturbation_distribution_plots = plot_all_perturbation_distributions(perturbation_results)

    results['permutation_distribution_plots'] = permutation_distribution_plots
    results['row_bootstrap_plots'] = row_bootstrap_plots
    results['column_bootstrap_random_plots'] = column_bootstrap_random_plots
    results['perturbation_distribution_plots'] = perturbation_distribution_plots

    pval_table = pd.DataFrame({
        'Reducer': list(permutation_results.keys()),
        'Permutation p-value': [float(f"{permutation_results[k]['p_value']:.5g}") for k in permutation_results],
        'Row Bootstrap p-value': [float(f"{bootstrap_results[k].get('bootstrap_p_value', float('nan')):.5g}") for k in bootstrap_results],
        'Col Bootstrap p-value': [float(f"{column_bootstrap_results_random[k].get('bootstrap_p_value', float('nan')):.5g}") for k in column_bootstrap_results_random],
        'Perturbation p-value': [float(f"{perturbation_results[k]['p_value']:.5g}") for k in perturbation_results]
    })
    results['pval_table'] = pval_table


    if pdf_output:
        with PdfPages(pdf_output) as pdf:
            fig = plt.figure(figsize=(8, 10))
            plt.axis('off')
            metadata_text = (
                f"Number of permutations: {n_permutations}\n"
                f"Number of bootstrap samples: {n_bootstrap_row}\n"
                f"Row bootstrap sample fraction: {bootstrap_sample_fraction_row:.1%}\n"
                f"Column bootstrap sample fraction: {bootstrap_sample_fraction_row:.1%}\n"
                f"Number of perturbations: {n_perturbations}\n"
                f"Scale factor for perturbations: {scale_factor}\n"
                f"Normalization: {normalize or 'None'}\n"
                f"Random seed: {random_seed or 'None'}\n"
            )
            plt.text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10, transform=fig.transFigure)
            pdf.savefig(fig)
            plt.close(fig)

            # Save permutation distribution plots
            for fig in results['permutation_distribution_plots'].values():
                pdf.savefig(fig)
                plt.close(fig)
            # Save row bootstrap distribution plots
            for fig in results['row_bootstrap_plots'].values():
                pdf.savefig(fig)
                plt.close(fig)
            # Save column bootstrap (random) distribution plots
            for fig in results['column_bootstrap_random_plots'].values():
                pdf.savefig(fig)
                plt.close(fig)
            # Save perturbation distribution plots
            for fig in results['perturbation_distribution_plots'].values():
                pdf.savefig(fig)
                plt.close(fig)
            # Save p-value table as a figure

            fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(pval_table))) 
            ax.axis('off')

            tbl = ax.table(
                cellText=pval_table.values,
                colLabels=pval_table.columns,
                loc='center',
                cellLoc='center',
                colColours=['#f2f2f2'] * len(pval_table.columns)
            )

            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.2, 1.5) 
            for (row, col), cell in tbl.get_celld().items():
                if col == 0:
                    cell.set_width(0.6) 
                else:
                    cell.set_width(0.12) 

            ax.set_title("P-value Table for Each Reducer", fontsize=14, pad=20)
            pdf.savefig(fig, orientation='landscape')
            plt.close(fig)
    return results
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
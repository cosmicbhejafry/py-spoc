# PySPI API Documentation

A comprehensive Python library for statistical analysis with permutation testing, bootstrap sampling, and perturbation analysis.

## Table of Contents

- [Data Processing Functions](#data-processing-functions)
- [Statistical Testing Functions](#statistical-testing-functions)
- [Visualization Functions](#visualization-functions)
- [Advanced Analysis Functions](#advanced-analysis-functions)

## Data Processing Functions

### `load_data(df)`

Drops non-numeric columns and converts to numeric.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change

**Returns:**
- `pd.DataFrame`: The loaded data

```python
def load_data(df) -> pd.DataFrame:
```

### `normalize_data(data, method='z-score')`

Normalizes the data using the specified method.

**Parameters:**
- `data` (pd.DataFrame): The data to normalize
- `method` (str, optional): The method to use for scaling. Options are:
  - `'z-score'`: Z-Score Normalization (StandardScaler)
  - `'min-max'`: Min-Max Normalization (MinMaxScaler)
  - `'robust'`: Robust Scaling using median and IQR (RobustScaler)
  - `'none'`: No scaling, returns data unchanged

**Returns:**
- `pd.DataFrame`: The normalized (or unchanged) data

```python
def normalize_data(data: pd.DataFrame, method: str = 'z-score') -> pd.DataFrame:
```

### `shuffle_data(df, random_seed=None)`

Returns a new DataFrame with each column's elements shuffled independently.

**Parameters:**
- `df` (pd.DataFrame): The input pandas DataFrame
- `random_seed` (int, optional): Optional seed for reproducibility

**Returns:**
- `pd.DataFrame`: A new DataFrame with columns shuffled

```python
def shuffle_data(df: pd.DataFrame, random_seed=None) -> pd.DataFrame:
```

### `perturb_dataframe(df, scale_factor=0.1, random_seed=None)`

Perturb the values in a DataFrame by adding Gaussian noise scaled by the column variances.

**Parameters:**
- `df` (pandas.DataFrame): Input DataFrame containing numeric values
- `scale_factor` (float, default=0.1): Scaling factor for the perturbation
- `random_seed` (int, optional): Random seed for reproducible results

**Returns:**
- `pd.DataFrame`: DataFrame with perturbed values

```python
def perturb_dataframe(df, scale_factor=0.1, random_seed=None):
```

## Statistical Testing Functions

### `apply_reducer(calc, reducer)`

Apply a specific reduced statistic to a matrix and get a value, useful for iterating.

**Parameters:**
- `calc` (Calculator): The calculator instance
- `reducer`: Gets reduced statistic value from calc.results

**Returns:**
- `float`: The reduced statistic value

```python
def apply_reducer(calc, reducer):
```

### `pyspi_calc(df, config_name, yaml, normalize=None, shuffle=False, perturb=False, scale_factor=0.1, random_seed=None)`

Apply perturbation, shuffling and normalisation as appropriate.

**Parameters:**
- `df` (DataFrame): DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `normalize` (str, optional): Method to normalize data (default: None)
- `shuffle` (bool): Whether to randomly shuffle rows (default: False)
- `perturb` (bool): Whether to perturb the data (default: False)
- `scale_factor` (float): Scale factor for perturbation (default: 0.1)
- `random_seed` (int, optional): Seed for reproducible shuffling (default: None)

**Returns:**
- Calc results table

```python
def pyspi_calc(df, config_name, yaml, normalize=None, shuffle=False, 
               perturb=False, scale_factor=0.1, random_seed=None):
```

### `permutation_test(df, config_name, yaml, calc, n_permutations=1000, normalize=None, random_seed=None, save_permutation_data=True, orig_calc_table=None)`

Perform a permutation test to assess statistical significance and calculate variance.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `calc`: Calculator instance with original results
- `n_permutations` (int, optional): Number of permutations in our test (default: 1000)
- `normalize` (str, optional): Method to normalize data (default: None)
- `random_seed` (int, optional): Seed for reproducible shuffling (default: None)
- `save_permutation_data` (bool, optional): Flag to save permutation data (default: True)
- `orig_calc_table` (Dict, optional): Check to see if orig_calc_table exists

**Returns:**
- `Dict[str, Dict[str, Any]]`: A dictionary containing the original values, p-values, and permutation statistics for each reduced statistic

```python
def permutation_test(df: pd.DataFrame, config_name, yaml, calc,
    n_permutations: int = 1000, normalize: Optional[str] = None,
    random_seed: Optional[int] = None, 
    save_permutation_data: bool = True, 
    orig_calc_table: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
```

### `perturbation_test(df, config_name, yaml, calc, n_perturbations=1000, scale_factor=0.1, normalize=None, random_seed=None, save_perturbation_data=True, orig_calc_table=None)`

Perform a perturbation test to assess statistical significance and calculate variance.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `calc`: Calculator instance with original results
- `n_perturbations` (int, optional): Number of perturbation samples (default: 1000)
- `scale_factor` (float, optional): Scale factor for perturbation (default: 0.1)
- `normalize` (str, optional): Method to normalize data (default: None)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)
- `save_perturbation_data` (bool, optional): Whether to save perturbation data (default: True)
- `orig_calc_table` (Dict, optional): Original calculation table if available

**Returns:**
- `Dict[str, Dict[str, Any]]`: A dictionary containing the original values, perturbation statistics, and p-values for each reducer

```python
def perturbation_test(df: pd.DataFrame, config_name, yaml, calc,
    n_perturbations: int = 1000, scale_factor: float = 0.1,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_perturbation_data: bool = True,
    orig_calc_table: Optional[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
```

## Bootstrap Sampling Functions

### `bootstrap_sample_data(df, sample_fraction=0.9, random_seed=None)`

Create a bootstrap sample from the dataframe by sampling a fraction of rows with replacement.

**Parameters:**
- `df` (pd.DataFrame): The original dataframe
- `sample_fraction` (float, optional): Fraction of rows to sample (default: 0.9)
- `random_seed` (int, optional): Random seed for reproducibility

**Returns:**
- `pd.DataFrame`: Bootstrap sampled dataframe (rows)

```python
def bootstrap_sample_data(df: pd.DataFrame, 
sample_fraction: float = 0.9, random_seed: Optional[int] = None) 
-> pd.DataFrame:
```

### `pyspi_calc_bootstrap(df, config_name, yaml, normalize=None, sample_fraction=0.9, random_seed=None)`

Perform bootstrap sampling and calculate statistics using PySPoC.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `normalize` (str, optional): Method to normalize data (default: None)
- `sample_fraction` (float, optional): Fraction of rows to sample for bootstrap (default: 0.9)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)

**Returns:**
- `Dict`: Dictionary of calculated statistics

```python
def pyspi_calc_bootstrap(df: pd.DataFrame, config_name: str, yaml, 
normalize: Optional[str] = None, sample_fraction: float = 0.9, 
random_seed: Optional[int] = None):
```

### `bootstrap_test(df, config_name, yaml, calc, n_bootstrap=10, sample_fraction=0.9, normalize=None, random_seed=None, save_bootstrap_data=True, orig_calc_table=None)`

Perform bootstrap sampling test to assess variability of statistics.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `calc`: Calculator instance with original results
- `n_bootstrap` (int, optional): Number of bootstrap samples (default: 10)
- `sample_fraction` (float, optional): Fraction of rows to sample for bootstrap (default: 0.9)
- `normalize` (str, optional): Method to normalize data (default: None)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)
- `save_bootstrap_data` (bool, optional): Whether to save bootstrap data (default: True)
- `orig_calc_table` (Dict, optional): Original calculation table if available

**Returns:**
- `Dict[str, Dict[str, Any]]`: A dictionary containing the original values, bootstrap statistics, and p-values for each reducer

```python
def bootstrap_test(df: pd.DataFrame, config_name, yaml, calc,
    n_bootstrap: int = 10, sample_fraction: float = 0.9,
    normalize: Optional[str] = None, 
    random_seed: Optional[int] = None,
    save_bootstrap_data: bool = True, 
    orig_calc_table: Optional[Dict] = None,  
) -> Dict[str, Dict[str, Dict[str, Any]]]:
```

### `bootstrap_sample_columns(df, sample_fraction=0.9, random_seed=None)`

Create a bootstrap sample from the dataframe by sampling a fraction of columns without replacement.

**Parameters:**
- `df` (pd.DataFrame): The original dataframe
- `sample_fraction` (float, optional): Fraction of columns to sample (default: 0.9)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)

**Returns:**
- `pd.DataFrame`: Bootstrap sampled dataframe (columns)

```python
def bootstrap_sample_columns(df: pd.DataFrame, 
sample_fraction: float = 0.9, random_seed: Optional[int] = None)
-> pd.DataFrame:
```

### `pyspi_calc_bootstrap_columns(df, config_name, yaml, normalize=None, sample_fraction=0.9, random_seed=None)`

Perform bootstrap sampling on columns and calculate statistics using PySPoC.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `normalize` (str, optional): Method to normalize data (default: None)
- `sample_fraction` (float, optional): Fraction of columns to sample for bootstrap (default: 0.9)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)

**Returns:**
- `Dict`: Dictionary of calculated statistics

```python
def pyspi_calc_bootstrap_columns(df: pd.DataFrame,
config_name: str, yaml, normalize: Optional[str] = None,
sample_fraction: float = 0.9, random_seed: Optional[int] = None):
```

### `bootstrap_test_columns(df, config_name, yaml, calc, n_bootstrap_columns=10, sample_fraction_columns=0.9, normalize=None, random_seed=None, save_bootstrap_data=True, orig_calc_table=None)`

Perform bootstrap sampling test to assess variability of statistics.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `calc`: Calculator instance with original results
- `n_bootstrap_columns` (int, optional): Number of bootstrap samples (default: 10)
- `sample_fraction_columns` (float, optional): Fraction of columns to sample for bootstrap (default: 0.9)
- `normalize` (str, optional): Method to normalize data (default: None)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)
- `save_bootstrap_data` (bool, optional): Whether to save bootstrap data (default: True)
- `orig_calc_table` (Dict, optional): Original calculation table if available

**Returns:**
- `Dict`: Dictionary of calculated statistics

```python
def bootstrap_test_columns(df: pd.DataFrame, config_name, yaml,
    calc, n_bootstrap_columns: int = 10, 
    sample_fraction_columns: float = 0.9,
    normalize: Optional[str] = None,
    random_seed: Optional[int] = None,
    save_bootstrap_data: bool = True,
    orig_calc_table: Optional[Dict] = None,  
) -> Dict[str, Dict[str, Dict[str, Any]]]:
```

## Visualization Functions

### `plot_permutation_distribution(results, reducer, figsize=(10, 6))`

Plot the distribution of permutation test values.

**Parameters:**
- `results` (Dict): Dictionary containing permutation test results
- `reducer` (str): Name of the reducer
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (10, 6)

**Returns:**
- `Figure`: Matplotlib figure object

```python
def plot_permutation_distribution(results: Dict, reducer: str,
figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

### `plot_all_permutation_distributions(results, figsize=(12, 8))`

Plot permutation distributions for all reducers in the results dictionary.

**Parameters:**
- `results` (Dict): Dictionary containing permutation test results for multiple reducers
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (12, 8)

**Returns:**
- `Dict[str, Figure]`: Dictionary mapping reducer names to their corresponding matplotlib Figure objects

```python
def plot_all_permutation_distributions(results: Dict, 
figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

### `plot_bootstrap_distribution(results, reducer, figsize=(10, 6))`

Plot the distribution of bootstrap test values.

**Parameters:**
- `results` (Dict): Dictionary containing bootstrap test results
- `reducer` (str): Name of the reducer
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (10, 6)

**Returns:**
- `Figure`: Matplotlib figure object

```python
def plot_bootstrap_distribution(results: Dict, reducer, 
figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

### `plot_all_bootstrap_distributions(results, figsize=(12, 8))`

Plot bootstrap distributions for all reducers in the results dictionary.

**Parameters:**
- `results` (Dict): Dictionary containing bootstrap test results for multiple reducers
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (12, 8)

**Returns:**
- `Dict[str, Figure]`: Dictionary mapping reducer names to their corresponding matplotlib Figure objects

```python
def plot_all_bootstrap_distributions(results: Dict, 
figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

### `plot_bootstrap_distribution_columns(results, reducer, figsize=(10, 6))`

Plot the distribution of bootstrap test values for columns.

**Parameters:**
- `results` (Dict): Dictionary containing bootstrap test results
- `reducer` (str): Name of the reducer
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (10, 6)

**Returns:**
- `Figure`: Matplotlib figure object

```python
def plot_bootstrap_distribution_columns(results: Dict, reducer,
    figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

### `plot_all_bootstrap_distributions_columns(results, figsize=(12, 8))`

Plot bootstrap distributions for all reducers in the results dictionary (for columns).

**Parameters:**
- `results` (Dict): Dictionary containing bootstrap test results for multiple reducers
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (12, 8)

**Returns:**
- `Dict[str, Figure]`: Dictionary mapping reducer names to their corresponding matplotlib Figure objects

```python
def plot_all_bootstrap_distributions_columns(results: Dict, 
figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

### `plot_perturbation_distribution(results, reducer, figsize=(10, 6))`

Plot the distribution of perturbation test values.

**Parameters:**
- `results` (Dict): Dictionary containing perturbation test results
- `reducer` (str): Name of the reducer
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (10, 6)

**Returns:**
- `Figure`: Matplotlib figure object

```python
def plot_perturbation_distribution(results: Dict, reducer: str,
    figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

### `plot_all_perturbation_distributions(results, figsize=(12, 8))`

Plot perturbation distributions for all reducers in the results dictionary.

**Parameters:**
- `results` (Dict): Dictionary containing perturbation test results for multiple reducers
- `figsize` (Tuple[int, int], optional): Figure size (width, height), by default (12, 8)

**Returns:**
- `Dict[str, Figure]`: Dictionary mapping reducer names to their corresponding matplotlib Figure objects

```python
def plot_all_perturbation_distributions(results: Dict, 
    figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

## Advanced Analysis Functions

### `run_analysis(df, config_name, yaml, n_permutations=1000, n_bootstrap_row=1000, n_bootstrap_col=1000, n_perturbations=1000, scale_factor=0.1, bootstrap_sample_fraction_row=0.9, bootstrap_sample_fraction_col=0.9, normalize=None, random_seed=None, pdf_output=None)`

Run a comprehensive permutation and bootstrap analysis with combined visualizations and detailed summaries.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `n_permutations` (int, optional): Number of permutations (default: 1000)
- `n_bootstrap_row` (int, optional): Number of row bootstrap samples (default: 1000)
- `n_bootstrap_col` (int, optional): Number of column bootstrap samples (default: 1000)
- `n_perturbations` (int, optional): Number of perturbations (default: 1000)
- `scale_factor` (float, optional): Scale factor for perturbations (default: 0.1)
- `bootstrap_sample_fraction_row` (float, optional): Fraction of rows to sample for bootstrap (default: 0.9)
- `bootstrap_sample_fraction_col` (float, optional): Fraction of columns to sample for bootstrap (default: 0.9)
- `normalize` (str, optional): Method to normalize data (default: None)
- `random_seed` (int, optional): Seed for reproducible sampling (default: None)
- `pdf_output` (str, optional): Path to save the PDF output with plots (default: None)

**Returns:**
- `Dict`: A dictionary containing all results, including p-values, plots, and metadata

```python
def run_analysis(df, config_name, yaml, n_permutations=1000, 
    n_bootstrap_row=1000, n_bootstrap_col=1000, n_perturbations=1000, 
    scale_factor=0.1, bootstrap_sample_fraction_row=0.9, 
    bootstrap_sample_fraction_col=0.9, normalize=None, 
    random_seed=None, pdf_output=None):
```

### `n_p_plane_bootstrap_gpr(df, config_name, yaml, n_grid=np.linspace(0.5, 1.0, 3), p_grid=np.linspace(0.5, 1.0, 3), n_bootstrap=10, normalize='z-score', random_seed=42, median_only=False)`

For each reducer, computes the bootstrap distribution (or median) at each (n, p) grid point, then fits and plots a Gaussian Process Regression surface. Stores all plots in a dictionary and returns them.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing the data to change
- `config_name` (str): Name of the configuration file
- `yaml` (str): YAML string for configuration
- `n_grid` (np.ndarray, optional): Grid of row fractions (default: np.linspace(0.5, 1.0, 3))
- `p_grid` (np.ndarray, optional): Grid of column fractions (default: np.linspace(0.5, 1.0, 3))
- `n_bootstrap` (int, optional): Number of bootstrap samples (default: 10)
- `normalize` (str, optional): Normalization method (default: 'z-score')
- `random_seed` (int, optional): Random seed for reproducibility (default: 42)
- `median_only` (bool, optional): If True, only fit GPR on medians instead of full bootstrap distributions (default: False)

**Returns:**
- `Tuple[Dict[str, Dict[str, Any]], Dict[str, Figure]]`: A dictionary containing results for each reducer and a dictionary of plots. Each reducer's results include means, GPR model, score, and grid information. The plots dictionary contains 3D surface plots for each reducer.

```python
def n_p_plane_bootstrap_gpr(df, config_name, yaml,
    n_grid=np.linspace(0.5, 1.0, 3),
    p_grid=np.linspace(0.5, 1.0, 3),
    n_bootstrap=10, normalize='z-score',
    random_seed=42, median_only=False
):
```

## Usage Examples

### Basic Data Processing

```python
import pandas as pd

# Load and normalize data
df = pd.read_csv('your_data.csv')
processed_df = load_data(df)
normalized_df = normalize_data(processed_df, method='z-score')
```

### Running Statistical Tests

```python
# Perform permutation test
perm_results = permutation_test(
    df=normalized_df,
    config_name='my_config',
    yaml=config_yaml,
    calc=calculator_instance,
    n_permutations=1000,
    random_seed=42
)

# Perform bootstrap test
bootstrap_results = bootstrap_test(
    df=normalized_df,
    config_name='my_config',
    yaml=config_yaml,
    calc=calculator_instance,
    n_bootstrap=100,
    random_seed=42
)
```

### Comprehensive Analysis

```python
# Run complete analysis pipeline
all_results = run_analysis(
    df=your_dataframe,
    config_name='analysis_config',
    yaml=yaml_configuration,
    n_permutations=1000,
    n_bootstrap_row=1000,
    n_bootstrap_col=1000,
    normalize='z-score',
    random_seed=42
)
```

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

## Installation

```bash
pip install pyspi
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]
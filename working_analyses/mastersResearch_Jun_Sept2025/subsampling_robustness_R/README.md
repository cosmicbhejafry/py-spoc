# Table of Contents

  - [Function: `load_data`](#function:-`load_data`)
  - [Function: `normalize_data`](#function:-`normalize_data`)
  - [Function: `shuffle_data`](#function:-`shuffle_data`)
  - [Function: `perturb_dataframe`](#function:-`perturb_dataframe`)
  - [Function: `apply_reducer`](#function:-`apply_reducer`)
  - [Function: `pyspi_calc`](#function:-`pyspi_calc`)
  - [Function: `permutation_test`](#function:-`permutation_test`)
  - [Function: `plot_permutation_distribution`](#function:-`plot_permutation_distribution`)
  - [Function: `plot_all_permutation_distributions`](#function:-`plot_all_permutation_distributions`)
  - [Function: `bootstrap_sample_data`](#function:-`bootstrap_sample_data`)
  - [Function: `pyspi_calc_bootstrap`](#function:-`pyspi_calc_bootstrap`)
  - [Function: `bootstrap_test`](#function:-`bootstrap_test`)
  - [Function: `plot_bootstrap_distribution`](#function:-`plot_bootstrap_distribution`)
  - [Function: `plot_all_bootstrap_distributions`](#function:-`plot_all_bootstrap_distributions`)
  - [Function: `bootstrap_sample_columns`](#function:-`bootstrap_sample_columns`)
  - [Function: `pyspi_calc_bootstrap_columns`](#function:-`pyspi_calc_bootstrap_columns`)
  - [Function: `bootstrap_test_columns`](#function:-`bootstrap_test_columns`)
  - [Function: `plot_bootstrap_distribution_columns`](#function:-`plot_bootstrap_distribution_columns`)
  - [Function: `plot_all_bootstrap_distributions_columns`](#function:-`plot_all_bootstrap_distributions_columns`)
  - [Function: `perturbation_test`](#function:-`perturbation_test`)
  - [Function: `plot_perturbation_distribution`](#function:-`plot_perturbation_distribution`)
  - [Function: `plot_all_perturbation_distributions`](#function:-`plot_all_perturbation_distributions`)
  - [Function: `run_analysis`](#function:-`run_analysis`)
  - [Function: `n_p_plane_bootstrap_gpr`](#function:-`n_p_plane_bootstrap_gpr`)

## Function: `load_data`

**Description**
Drops non-numeric columns and converts to numeric

**Syntax**

```python
def load_data(df) -> pd.DataFrame:
```

**Parameters**

```
df : pd.DataFrame
    The DataFrame containing the data to change
```

**Returns**

```
pd.DataFrame: 
    The loaded data
```

---

## Function: `normalize_data`

**Description**
Normalizes the data using the specified method

**Syntax**

```python
def normalize_data(data: pd.DataFrame, method: str = 'z-score') -> pd.DataFrame:
```

**Parameters**

```
data : pd.DataFrame
    The data to normalize.
method : str, optional
    The method to use for scaling. Options are:
    - 'z-score': Z-Score Normalization (StandardScaler)
    - 'min-max': Min-Max Normalization (MinMaxScaler)
    - 'robust': Robust Scaling using median and IQR (RobustScaler)
    - 'none': No scaling, returns data unchanged.
```

**Returns**

```
pd.DataFrame
    The normalized (or unchanged) data.
```

---

## Function: `shuffle_data`

**Description**
Returns a new DataFrame with each column's elements shuffled independently.

**Syntax**

```python
def shuffle_data(df: pd.DataFrame, random_seed=None) -> pd.DataFrame:
```

**Parameters**

```
df: pd.DataFrame
    The input pandas DataFrame.
random_seed: Optional seed for reproducibility.
```

**Returns**

```
pd.DataFrame
    A new DataFrame with columns shuffled.
```

---

## Function: `perturb_dataframe`

**Description**
Perturb the values in a DataFrame by adding Gaussian noise scaled by the column variances

**Syntax**

```python
perturb_dataframe(df, scale_factor=0.1, random_seed=None)
```

**Parameters**

```python
df : pandas.DataFrame
    Input DataFrame containing numeric values
scale_factor : float, default=0.1
    Scaling factor for the perturbation
random_seed : int, optional
    Random seed for reproducible results
```

**Returns**

```
pd.DataFrame
    DataFrame with perturbed values
```

---

## Function: `apply_reducer`

**Description**
Apply a specific reduced statistic to a matrix and get a value, useful for iterating

**Syntax**

```python
def apply_reducer(calc: Calculator, reducer) -> float:
```

**Parameters**

```
calc : Calculator
    The calculator instance.
reducer : gets reduced statistic value from calc.results
```

**Returns**

```python
float:
    The reduced statistic value.
```

---

## Function: `pyspi_calc`

**Description**
Apply perturbation, shuffling and normalisation as appropriate

**Syntax**

```python
def pyspi_calc(df, config_name, yaml, normalize=None, shuffle=False, perturb=False, scale_factor=0.1, random_seed=None):
```

**Parameters**

```python
df: DataFrame containing the data to change  
config_name: Name of the configuration file  
yaml: YAML string for configuration  
normalize: Method to normalize data (default: None)  
shuffle: Whether to randomly shuffle rows (default: False)  
random_seed: Seed for reproducible shuffling (default: None)
```

**Returns**

```python
Calc results table
```

---

## Function: `permutation_test`

**Description**
Perform a permutation test to assess statistical significance and calculate variance

**Syntax**

```python
def permutation_test(df: pd.DataFrame, config_name, yaml, calc,
                     n_permutations: int = 1000,
                     normalize: Optional[str] = None,
                     random_seed: Optional[int] = None,
                     save_permutation_data: bool = True,
                     orig_calc_table: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
```

**Parameters**

```
df: DataFrame containing the data to change  
config_name: Name of the configuration file  
yaml: YAML string for configuration  
n_permutations: number of permutations in our test (default: 1000)  
normalize: Method to normalize data (default: None)  
random_seed: Seed for reproducible shuffling (default: None)  
save_permutation_data: Flag to save permutation data (default: True)  
orig_calc_table: Check to see if orig_calc_table exists
```

**Returns**

```
A dictionary containing the original values, p-values, and 
permutation statistics for each reduced statistic
```

---

## Function: `plot_permutation_distribution`

**Description**
Plot the distribution of permutation test values

**Syntax**

```python
def plot_permutation_distribution(results: Dict, reducer: str, figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

**Parameters**

```python
results : Dict
    Dictionary containing permutation test results.
reducer : str
    Name of the reducer
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (10, 6)
```

**Returns**

```
Figure
    Matplotlib figure object.
```

## Function: `plot_all_permutation_distributions`

**Description**
Plot permutation distributions for all reducers in the results dictionary

**Syntax**

```python
def plot_all_permutation_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

**Parameters**

```
results : Dict
    Dictionary containing permutation test results for multiple reducers.
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (12, 8)
```

**Returns**

```python
Dict[str, Figure]
    Dictionary mapping reducer names to their corresponding matplotlib Figure objects.
```

---

## Function: `bootstrap_sample_data`

**Description**
Create a bootstrap sample from the dataframe by sampling a fraction of rows with replacement

**Syntax**

```python
def bootstrap_sample_data(df: pd.DataFrame, 
                          sample_fraction: float = 0.9, 
                          random_seed: Optional[int] = None) -> pd.DataFrame:
```

**Parameters**

```python
df : pd.DataFrame
    The original dataframe
sample_fraction : float, optional
    Fraction of rows to sample
random_seed : Optional[int], optional
    Random seed for reproducibility
```

**Returns**

```python
pd.DataFrame
    Bootstrap sampled dataframe (rows)
```

---

## Function: `pyspi_calc_bootstrap`

**Description**
Perform bootstrap sampling and calculate statistics using PySPoC

**Syntax**

```python
def pyspi_calc_bootstrap(df: pd.DataFrame, config_name: str, yaml, 
                         normalize: Optional[str] = None, 
                         sample_fraction: float = 0.9, 
                         random_seed: Optional[int] = None):
```

**Parameters**

```python
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
```

**Returns**

```python
Dictionary of calculated statistics
```

---

## Function: `bootstrap_test`

**Description**
Perform bootstrap sampling test to assess variability of statistics

**Syntax**

```python
def bootstrap_test(df: pd.DataFrame, config_name, yaml, calc,
                   n_bootstrap: int = 10, sample_fraction: float = 0.9,
                   normalize: Optional[str] = None, 
                   random_seed: Optional[int] = None,
                   save_bootstrap_data: bool = True, 
                   orig_calc_table: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
```

**Parameters**

```
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
```

**Returns**

```
Dict[str, Dict[str, Any]]
    A dictionary containing the original values, bootstrap statistics, and p-values for each reducer.
```

---

## Function: `plot_bootstrap_distribution`

**Description**
Plot the distribution of bootstrap test values

**Syntax**

```python
def plot_bootstrap_distribution(results: Dict, reducer, figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

**Parameters**

```python
results : Dict
    Dictionary containing bootstrap test results.
reducer : str
    Name of the reducer
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (10, 6)
```

**Returns**

```
Figure
    Matplotlib figure object.
```

---

## Function: `plot_all_bootstrap_distributions`

**Description**
Plot bootstrap distributions for all reducers in the results dictionary

**Syntax**

```python
def plot_all_bootstrap_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

**Parameters**

```
results : Dict
    Dictionary containing bootstrap test results for multiple reducers.
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (12, 8)
```

**Returns**

```python
Dict[str, Figure]
    Dictionary mapping reducer names to their corresponding 
    matplotlib Figure objects.
```

---

## Function: `bootstrap_sample_columns`

**Description**
Create a bootstrap sample from the dataframe by sampling a fraction of columns without replacement.

**Syntax**

```python
def bootstrap_sample_columns(df: pd.DataFrame, sample_fraction: float = 0.9, random_seed: Optional[int] = None) -> pd.DataFrame:
```

**Parameters**

```
df : pd.DataFrame
    The original dataframe
sample_fraction : float, optional
    Fraction of columns to sample (default: 0.9)
random_seed : Optional[int], optional
    Seed for reproducible sampling (default: None)
```

**Returns**

```python
pd.DataFrame
    Bootstrap sampled dataframe (columns)
```

---

## Function: `pyspi_calc_bootstrap_columns`

**Description**
Perform bootstrap sampling on columns and calculate statistics using PySPoC

**Syntax**

```python
def pyspi_calc_bootstrap_columns(df: pd.DataFrame, config_name: str, yaml,
                                 normalize: Optional[str] = None,
                                 sample_fraction: float = 0.9,
                                 random_seed: Optional[int] = None):
```

**Parameters**

```
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
```

**Returns**

```python
Dictionary of calculated statistics
```

---

## Function: `bootstrap_test_columns`

**Description**
Perform bootstrap sampling test to assess variability of statistics

**Syntax**

```python
def bootstrap_test_columns(df: pd.DataFrame, config_name, yaml,
                           calc, n_bootstrap_columns: int = 10,
                           sample_fraction_columns: float = 0.9,
                           normalize: Optional[str] = None,
                           random_seed: Optional[int] = None,
                           save_bootstrap_data: bool = True,
                           orig_calc_table: Optional[Dict] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
```

**Parameters**

```
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
```

**Returns**

```python
Dictionary of calculated statistics
```

---

## Function: `plot_bootstrap_distribution_columns`

**Description**
Plot the distribution of bootstrap test values for columns

**Syntax**

```python
def plot_bootstrap_distribution_columns(results: Dict, reducer, figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

**Parameters**

```python
results : Dict
    Dictionary containing bootstrap test results.
reducer : str
    Name of the reducer
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (10, 6)
```

**Returns**

```python
Figure
    Matplotlib figure object.
```

---

## Function: `plot_all_bootstrap_distributions_columns`

**Description**
Plot bootstrap distributions for all reducers in the results dictionary (for columns)

**Syntax**

```python
def plot_all_bootstrap_distributions_columns(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]
```

**Parameters**

```
results : Dict
    Dictionary containing bootstrap test results for multiple 
    reducers.
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (12, 8)
```

**Returns**

```python
Dict[str, Figure]
    Dictionary mapping reducer names to their 
    corresponding matplotlib Figure objects.
```

---

## Function: `perturbation_test`

**Description**
Perform a perturbation test to assess statistical significance and calculate variance

**Syntax**

```python
def perturbation_test(df: pd.DataFrame, config_name, yaml, calc,
                      n_perturbations: int = 1000, scale_factor: float = 0.1,
                      normalize: Optional[str] = None,
                      random_seed: Optional[int] = None,
                      save_perturbation_data: bool = True,
                      orig_calc_table: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
```

**Parameters**

```
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
```

**Returns**

```python
Dict[str, Dict[str, Any]]
    A dictionary containing the original values, perturbation 
    statistics, and p-values for each reducer
```

---

## Function: `plot_perturbation_distribution`

**Description**
Plot the distribution of perturbation test values

**Syntax**

```python
def plot_perturbation_distribution(results: Dict, reducer: str, figsize: Tuple[int, int] = (10, 6)) -> Figure:
```

**Parameters**

```python
results : Dict
    Dictionary containing perturbation test results.
reducer : str
    Name of the reducer
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (10, 6)
```

**Returns**

```
Figure
    Matplotlib figure object.
```

---

## Function: `plot_all_perturbation_distributions`

**Description**
Plot perturbation distributions for all reducers in the results dictionary

**Syntax**

```python
def plot_all_perturbation_distributions(results: Dict, figsize: Tuple[int, int] = (12, 8)) -> Dict[str, Figure]:
```

**Parameters**

``` 
results : Dict
    Dictionary containing perturbation test results for 
    multiple reducers.
figsize : Tuple[int, int], optional
    Figure size (width, height), by default (12, 8)
```

**Returns**

```python
Dict[str, Figure]
    Dictionary mapping reducer names to their corresponding 
    matplotlib Figure objects.
```

---

## Function: `run_analysis`

**Description**
Run a comprehensive permutation, bootstrap and perturbation analysis with combined visualizations and detailed summaries

**Syntax**

```python
def run_analysis(df: pd.DataFrame, config_name: str, yaml,
                 n_permutations: int = 1000,
                 n_bootstrap_row: int = 1000,
                 n_bootstrap_col: int = 1000,
                 n_perturbations: int = 1000,
                 scale_factor: float = 0.1,
                 bootstrap_sample_fraction_row: float = 0.9,
                 bootstrap_sample_fraction_col: float = 0.9,
                 normalize: Optional[str] = None,
                 random_seed: Optional[int] = None,
                 pdf_output: Optional[str] = None) -> Dict:
```

**Parameters**
```df : pd.DataFrame
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
```



**Returns**

```
Dict
    A dictionary containing all results, including p-values, plots, and metadata.
```

---

## Function: `n_p_plane_bootstrap_gpr`

**Description**
For each reducer, computes the bootstrap distribution (or median) at each (n, p) grid point, then fits and plots a Gaussian Process Regression surface.

**Syntax**

```python
def n_p_plane_bootstrap_gpr(df, config_name, yaml,
                            n_grid=np.linspace(0.5, 1.0, 3),
                            p_grid=np.linspace(0.5, 1.0, 3),
                            n_bootstrap=10, normalize='z-score',
                            random_seed=42, median_only=False):
```

**Parameters**
```df : pd.DataFrame
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
    If True, only fit GPR on medians instead of full 
    bootstrap distributions (default: False)
```

**Returns**

```
Dict[str, Dict[str, Any]], Dict[str, Figure]
    A dictionary containing results for each reducer and a 
    dictionary of plots.
    Each reducer's results include means, GPR model, score, 
    and grid information.
    The plots dictionary contains 3D surface plots for 
    each reducer.
```


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




class RobustnessAnalyzer:
    def __init__(self, df, config_name, yaml, normalize=None, random_seed=None):
        self.df = df
        self.config_name = config_name
        self.yaml = yaml
        self.normalize = normalize
        self.random_seed = random_seed
        self.orig_calc_table = self._calc_orig()

    def _calc_orig(self):
        return pyspi_calc(self.df, self.config_name, self.yaml, normalize=self.normalize)

    def permutation(self, n_permutations=1000, save_data=True):
        return permutation_test(
            self.df, self.config_name, self.yaml, self.orig_calc_table,
            n_permutations=n_permutations, normalize=self.normalize,
            random_seed=self.random_seed, save_permutation_data=save_data,
            orig_calc_table=self.orig_calc_table
        )

    def bootstrap_row(self, n_bootstrap=1000, sample_fraction=0.9, save_data=True):
        return bootstrap_test(
            self.df, self.config_name, self.yaml, self.orig_calc_table,
            n_bootstrap=n_bootstrap, sample_fraction=sample_fraction,
            normalize=self.normalize, random_seed=self.random_seed,
            save_bootstrap_data=save_data, orig_calc_table=self.orig_calc_table
        )
    def bootstrap_column(self, n_bootstrap_columns=1000, sample_fraction_columns=0.9, save_data=True):
        return bootstrap_test_columns(
            self.df, self.config_name, self.yaml, self.orig_calc_table,
            n_bootstrap_columns=n_bootstrap_columns,
            sample_fraction_columns=sample_fraction_columns,
            normalize=self.normalize, random_seed=self.random_seed,
            save_bootstrap_data=save_data, orig_calc_table=self.orig_calc_table
        )
    def perturbation(self, n_perturbations=1000, scale_factor=0.1, save_data=True):
        return perturbation_test(
            self.df, self.config_name, self.yaml, self.orig_calc_table,
            n_perturbations=n_perturbations, scale_factor=scale_factor,
            normalize=self.normalize, random_seed=self.random_seed,
            save_perturbation_data=save_data, orig_calc_table=self.orig_calc_table
        )

    def run_all(self, n_permutations=1000, n_bootstrap_row=1000, n_bootstrap_col=1000, n_perturbations=1000,
                scale_factor=0.1, bootstrap_sample_fraction_row=0.9, bootstrap_sample_fraction_col=0.9,
                pdf_output=True):
        """
        Run all analyses and return a comprehensive results dictionary.
        """
        print(f"Running analysis with {n_permutations} permutations, {n_bootstrap_row} row bootstrap samples, "
              f"{n_bootstrap_col} column bootstrap samples, {n_perturbations} perturbations")
        results = run_analysis(
            self.df,
            self.config_name,   
            self.yaml,
            n_permutations=n_permutations,
            n_bootstrap_row=n_bootstrap_row,   
            n_bootstrap_col=n_bootstrap_col,
            n_perturbations=n_perturbations,
            scale_factor=scale_factor,
            bootstrap_sample_fraction_row=bootstrap_sample_fraction_row,
            bootstrap_sample_fraction_col=bootstrap_sample_fraction_col,
            normalize=self.normalize,
            random_seed=self.random_seed,
            pdf_output=pdf_output
        )
        return results  
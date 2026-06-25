from pyspoc import Reducer, ReducedStatistic
import numpy as np
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset
import numpy as np
from scipy.stats import gmean

class MaxCovPerFeature(Reducer):

    """
    max absolute off-diagonal value per feature, computed from the covariance matrix
    """

    name = "max absolute off-diagonal value per feature"
    identifier = "maxcov"
    labels = ["vector"]

    summary_fns_dict = {
        "median" : lambda dists : np.median(dists),
        "min" : lambda dists : np.min(dists),
        "max" : lambda dists : np.max(dists),
    }

    def __init__(self, stats: list[str] = ["mean"]):
        self.stats = stats
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray:

        assert data.shape[0]==data.shape[1]
        
        # mask diagonal value as nan
        mask = ~np.eye(data.shape[0], dtype=bool)        
        A_masked = np.where(mask, data, np.nan)
        
        max_vals = np.nanmax(np.abs(A_masked), axis=0)

        if set(self.stats) <= (set(self.summary_fns_dict.keys())):        
            # summary.append(np.median(dists))
            summary = [self.summary_fns_dict[v](max_vals) for v in self.stats]            
        else:
            raise ValueError(f"Unsupported stats: {self.stats}")

        return np.array(summary)

class EigenAMGMRatio(Reducer):
    name = "EigenAMGMRatio"
    identifier = "eig_summary"
    labels = ["isotropy"]

    def __init__(self,):
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray:
        """computes ratio of arithmetic mean / geometric mean.
        should always be >= 1
        the closer it is to 1, the more spherical data"""

        eigs = np.linalg.eigvals(data)
        
        if np.any(eigs <= 0):
            raise ValueError("Negative eigen values?")

        # arithmetic_mean = np.mean(eigs)
        # geometric_mean = gmean(eigs)
    
        return np.mean(eigs) / gmean(eigs)
    

class EigenEntropy(Reducer):
    # https://en.wikipedia.org/wiki/Von_Neumann_entropy
    name = "EigenEntropy"
    identifier = ""
    labels = ["isotropy"]

    def __init__(self,):
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        """

        # make trace = 1
        data_norm = data / np.trace(data)

        eigs = np.linalg.eigvals(data_norm)
        
        if np.any(eigs <= 0):
            raise ValueError("Negative eigen values?")

    
        return -np.sum(eigs * np.log(eigs))


"""
implements some Reducers based on covariance matrix, and also eigenvals of input matrix
"""

from pyspoc import Reducer, ReducedStatistic
import numpy as np
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset
from scipy.stats import gmean
import scipy.stats as sp

class MaxCovPerFeature(Reducer):

    """
    max absolute off-diagonal value per feature, computed from the covariance matrix
    """

    # name = "max absolute off-diagonal value per feature"
    # identifier = "maxcov"
    # labels = ["vector"]

    summary_fns_dict = {
        "median" : lambda dists : np.median(dists),
        "min" : lambda dists : np.min(dists),
        "max" : lambda dists : np.max(dists),
    }

    def __init__(self, stats: list[str] = ["mean"]):
        self.stats = stats
        super().__init__()

    def reduce(self, data: np.ndarray) -> np.ndarray:
        
        """        
        Summarize Statistic output as follows:
            Return median, min and max of the absolute off-diagonal value per column
        """
        

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

class SvdAMGMRatio(Reducer):
    # name = "EigenAMGMRatio"
    # identifier = "eig_summary"
    # labels = ["isotropy"]

    def __init__(self):
        super().__init__()

    def reduce(self, data: np.ndarray) -> np.ndarray:
        
        
        """
        Summarize Statistic output as follows:
            Find eigenvalues of the square matrix
            Return the ratio of the arithmetic / geometric mean of the eigenvals                
            (intuition- should always be >= 1, the closer it is to 1, the more spherical data)        
        """
        
        eigs = np.linalg.svdvals(data)
        
        if np.any(eigs <= 0):
            raise ValueError("Negative eigen values?")

        tol = 1e-14 * max(eigs.max(), 0)
        kept = eigs[eigs > tol]
    
        return np.array([gmean(kept) / np.mean(kept)])
    

class SvdEntropy(Reducer):
    # https://en.wikipedia.org/wiki/Von_Neumann_entropy
    # name = "EigenEntropy"
    # identifier = ""
    # labels = ["isotropy"]

    def __init__(self,square_sum=False):
        super().__init__()
        self.square_sum=square_sum

    def reduce(self, data: np.ndarray) -> np.ndarray:
        """
        Summarize Statistic output as follows:
            Normalize square matrix to have trace = 1
            Find eigenvals
            Return entropy of the eigenval distribution
            (ref. von neumann entropy, concept from physics)       
        """

        # make trace = 1
        # data_sym = (data + data.T)/2        
        # data_norm = data / np.trace(data)

        eigs_full = np.linalg.svdvals(data)
        if self.square_sum:
            eigs = eigs_full**2 / np.sum(eigs_full**2)            
        else:            
            eigs = eigs_full / np.sum(eigs_full)
        
        if np.any(eigs <= 0):
            raise ValueError("Negative eigen values?")
    
        return np.array([-np.sum(eigs * np.log(eigs))])

class EigenParticipationRatio(Reducer):
    pass
    # https://peterponyu.github.io/liora-ui/metrics/participation_ratio/

class MeanOfMoments(Reducer):

    _name = "MeanOfMoments"
    _identifier = "moment"
    _labels = ["vector"]
    
    def __init__(self, moments: list[int] = [2,4]):
        self._moments = moments
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        mom = sp.moment(data, order = self._moments)
        return mom.mean(axis=1)

class MeanOfMoments(Reducer):

    _name = "MeanOfMoments"
    _identifier = "moment"
    _labels = ["vector"]
    
    def __init__(self, moments: list[int] = [2,4]):
        self._moments = moments
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        n = data.shape[0]
        cols = data.T[~np.eye(n, dtype=bool)].reshape(n, n - 1)     # row i = column i of data, without its diagonal entry
        mom = sp.moment(cols, order = self._moments,axis=1)
        return mom.mean(axis=1)

import numpy as np

from typing import Union, Literal
from sklearn.metrics import pairwise_distances
from hyppo.independence import (
    MGC,
    Dcorr,
    HHG,
)

from pyspoc.statistics.base import Statistic, PairwiseStatistic


class PairwiseDistance(Statistic):

    _name = "Pairwise distance"
    _identifier = "pdist"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, dim: Literal["n", "p"] = "p", metric = "euclidean"):
        self._dim = dim
        self._metric = metric
        self._identifier += f".{metric}"
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray:
        if self._dim == "n":
            return pairwise_distances(data, metric=self._metric)

        return pairwise_distances(data.T, metric=self._metric)


class HellerHellerGorfine(PairwiseStatistic):
    """Heller-Heller-Gorfine independence criterion"""

    _name = "Heller-Heller-Gorfine Independence Criterion"
    _identifier = "hhg"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "directed"]

    def __init__(self, dim: str):
        super().__init__(dim=dim,
                         is_ordered=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        stat = HHG().statistic(x, y)
        return stat # type: ignore


class DistanceCorrelation(PairwiseStatistic):
    """Distance correlation"""

    _name = "Distance correlation"
    _identifier = "dcorr"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, dim: str, biased: bool):
        self.__biased = biased

        if biased:
            self._identifier += ".biased"

        super().__init__(dim=dim,
                         is_ordered=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        stat = Dcorr(bias=self.__biased).statistic(x, y)
        return stat


class MultiscaleGraphCorrelation(PairwiseStatistic):
    """Multiscale graph correlation"""

    _name = "Multiscale graph correlation"
    _identifier = "mgc"
    _labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]

    def __init__(self, dim: str):
        super().__init__(dim=dim,
                         is_ordered=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        stat = MGC().statistic(x, y)
        return stat


class GromovWasserstainTau(PairwiseStatistic):
    """Gromov-Wasserstain distance (GWTau)"""

    _name = "Gromov-Wasserstain Distance"
    _identifier = "gwtau"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self):
        super().__init__(dim="p",
                         is_ordered=False)

    @staticmethod
    def vec_geo_dist(x):
        diffs = np.diff(x, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.cumsum(distances)
    
    @staticmethod
    def wass_sorted(x1, x2):
        x1 = np.sort(x1)[::-1]  # sort in descending order
        x2 = np.sort(x2)[::-1]

        if len(x1) == len(x2):
            res = np.sqrt(np.mean((x1 - x2) ** 2))
        else:
            N, M = len(x1), len(x2)
            i_ratios = np.arange(1, N + 1) / N
            j_ratios = np.arange(1, M + 1) / M
        
            min_values = np.minimum.outer(i_ratios, j_ratios)
            max_values = np.maximum.outer(i_ratios - 1/N, j_ratios - 1/M)
        
            lam = np.where(min_values > max_values, min_values - max_values, 0)
        
            diffs_squared = (x1[:, None] - x2) ** 2
            my_sum = np.sum(lam * diffs_squared)
        
            res = np.sqrt(my_sum)

        return res
    
    @staticmethod
    def gwtau(xi, xj):
        timei = np.arange(len(xi))
        timej = np.arange(len(xj))
        traji = np.column_stack([timei, xi])
        trajj = np.column_stack([timej, xj])

        vi = GromovWasserstainTau.vec_geo_dist(traji)
        vj = GromovWasserstainTau.vec_geo_dist(trajj)
        gw = GromovWasserstainTau.wass_sorted(vi, vj)
    
        return gw

    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        stat = self.gwtau(x, y)
        return stat

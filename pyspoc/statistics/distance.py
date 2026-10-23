import numpy as np

from typing import Union, Literal, Any
from sklearn.metrics import pairwise_distances

from pyspoc.settings import settings
from pyspoc.statistics.base import Statistic, PairwiseStatistic
from pyspoc._tag import Tag, Type

# NOTE (GSC): The hyppo library is quite slow and likely these statistics will need to be
# implemented natively in order to be viable.


AVAILABLE_DISTANCES = (
    "euclidean",
    "nan_euclidean",
    "manhattan",
    "cosine",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "haversine",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalsneath",
    "sqeuclidean",
    "yule"
)

class PairwiseDistance(Statistic):

    _long_name = "Pairwise distance"
    _identifier = "pdist"
    _tags = (Tag.DISTANCE, Tag.SYMMETRICAL)
    _type = Type.DESCRIPTIVE
    _signatures = ()

    def __init__(
            self,
            metric: Literal[
                "euclidean",
                "nan_euclidean",
                "manhattan",
                "cubic",
                "cosine",
                "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "dice",
                "hamming",
                "jaccard",
                "haversine",
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalsneath",
                "sqeuclidean",
                "yule"] = "euclidean",
            force_finite: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            **kwargs: Any):
        
        self._metric = metric
        self._internal_metric = metric
        self._force_finite = force_finite
        self._identifier += f".{metric}"
        self._kwargs = kwargs

        if metric in ("correlation", "cosine"):
            self._tags = self._get_modified_tags(
                self._tags,
                add_tags=Tag.ASSOCIATION)

        if metric == "cubic":
            self._kwargs["p"] = 3
            self._internal_metric = "minkowski"
            
        super().__init__(dim=dim, nan_policy=nan_policy)


    @staticmethod
    def available_distances() -> tuple[str, ...]:
        return AVAILABLE_DISTANCES

    def _compute(self, data: np.ndarray) -> np.ndarray:
        n_jobs = settings.current.max_worker_threads or -1
        return pairwise_distances(
            data.T,
            metric=self._internal_metric,
            n_jobs=n_jobs,
            ensure_all_finite=self._force_finite)



class GromovWassersteinTau(PairwiseStatistic):
    """Gromov-Wasserstein distance (GWTau)"""

    _name = "Gromov-Wasserstein Distance"
    _identifier = "gwtau"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]
    _tags = (Tag.ORDERED,)

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def tags(self) -> list[str]:
        return self._labels

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

        vi = GromovWassersteinTau.vec_geo_dist(traji)
        vj = GromovWassersteinTau.vec_geo_dist(trajj)
        gw = GromovWassersteinTau.wass_sorted(vi, vj)
    
        return gw

    def _pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        stat = self.gwtau(x, y)
        return stat

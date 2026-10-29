import numpy as np

from abc import ABC
from typing import Literal

from pyspoc.settings import settings
from pyspoc.component.statistics.base import PairwiseStatistic
from pyspoc._tag import Tag, Type, Signature


class BasePairwiseDistance(PairwiseStatistic, ABC):

    _signatures = (Signature.DENSITY,)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.DISTANCE, Tag.SYMMETRICAL, Tag.NONPARAMETRIC)

    def __init__(
            self,
            metric: Literal[
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
                "yule"] = "euclidean",
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(is_ordered=False,
                         dim=dim,
                         nan_policy=nan_policy)

        self._metric = metric


class HellerHellerGorfine(BasePairwiseDistance):
    """Heller-Heller-Gorfine independence criterion"""

    _long_name = "Heller-Heller-Gorfine Independence Criterion"
    _identifier = "hhg"
    _tags = (Tag.ASSOCIATION, Tag.DISTANCE, Tag.NONPARAMETRIC, Tag.SYMMETRICAL, Tag.NONLINEAR)
    _signatures = (Signature.NONLINEARITY,)
    _type = Type.DESCRIPTIVE

    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[()], np.dtype[np.floating]] | np.floating | float:

        from hyppo.independence import HHG
        x_2d = x.reshape(-1,1) if x.ndim < 2 else x
        y_2d = y.reshape(-1,1) if y.ndim < 2 else y
        stat = HHG(compute_distance=self._metric).statistic(x_2d, y_2d)
        return stat # type: ignore



class DistanceCorrelation(BasePairwiseDistance):
    """Distance correlation"""

    _name = "Distance correlation"
    _identifier = "dcorr"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            metric: Literal[
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
                "yule"] = "euclidean",
            biased: bool = False):
        
        self._biased = biased

        if biased:
            self._identifier += ".biased"

        super().__init__(
            dim=dim,
            metric=metric)

    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[()], np.dtype[np.floating]] | np.floating | float:

        from hyppo.independence import Dcorr
        x_2d = x.reshape(-1,1) if x.ndim < 2 else x
        y_2d = y.reshape(-1,1) if y.ndim < 2 else y
        stat = Dcorr(
            compute_distance=self._metric,
            bias=self._biased).statistic(x_2d, y_2d)
        return stat



class MultiscaleGraphCorrelation(BasePairwiseDistance):
    """Multiscale graph correlation"""

    _name = "Multiscale graph correlation"
    _identifier = "mgc"
    _labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]


    def _pairwise_compute(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[()], np.dtype[np.floating]] | np.floating | float:

        from hyppo.independence import MGC
        x_2d = x.reshape(-1,1) if x.ndim < 2 else x
        y_2d = y.reshape(-1,1) if y.ndim < 2 else y
        stat = MGC(compute_distance=self._metric).statistic(x_2d, y_2d)
        return stat

from __future__ import annotations

import numpy as np

from typing import Literal, Iterable
from sklearn.decomposition import PCA
from abc import ABC

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.dataset import Dataset
from pyspoc.settings import settings
from pyspoc._tag import Type, Signature


class PCABase(ReducedStatistic, ABC):

    _cached_pcas: dict[Dataset, tuple[PCA, int]] = dict()
    _signatures = (Signature.DIMENSIONALITY,)
    _tags = ()
    _type = Type.DESCRIPTIVE
    

    def __init__(
            self,
            components: int | Iterable[int],
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        self._cached_pca = None
        
        self._components = tuple(range(1, components + 1)) if isinstance(components, int) \
            else tuple(components)

        self._n_components = max(self._components)
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)
    
    
    @property
    def components(self) -> tuple[int, ...]:
        return self._components
    
    def calculate(self, dataset: Dataset):
        cached_pca_tuple = self._cached_pcas.get(dataset)
        cached_pca = None
        curr_n_components = 0

        if cached_pca_tuple:
            cached_pca, curr_n_components = cached_pca_tuple

        if curr_n_components >= self._n_components:
            self._cached_pca = cached_pca

        result = super().calculate(dataset)

        # TEMPORARILY DISABLE CACHING, TODO: cache fix
        # self._cached_pcas[dataset] = (self._cached_pca, self._n_components)
        self._cached_pcas[dataset] = (None, 0)

        return result
    
    def _get_pca(self, data: np.ndarray) -> PCA:

        # TEMPORARILY DISABLE CACHING, TODO: cache fix
        # if self._cached_pca is not None:
        #     return self.__cached_pca
        
        pca = PCA(n_components=self._n_components)
        pca.fit(data)
        # self._cached_pca = pca
        return pca

   

class PCAVarianceExplainedRatio(PCABase):

    _long_name = "Principal Components Analysis - Variance Explained Ratio"
    _identifier = "pca-var"


    def _compute(self, data: np.ndarray) -> np.ndarray | float:
        pca = self._get_pca(data)
        indices = [i - 1 for i in self._components]
        return pca.explained_variance_ratio_[indices]



class PCAEigenVectors(PCABase):

    _long_name = "Principal Components Analysis - Eigen Vectors"
    _identifier = "pca-eig"


    def _compute(self, data: np.ndarray) -> np.ndarray | float:
        pca = self._get_pca(data)
        evectors = pca.components_
        indices = [i - 1 for i in self._components]
        return evectors[indices]

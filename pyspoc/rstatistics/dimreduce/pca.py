from __future__ import annotations

import numpy as np

from typing import Union, Iterable
from sklearn.decomposition import PCA
from abc import ABC

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.dataset import Dataset


class PCABase(ReducedStatistic, ABC):

    _cached_pcas: dict[Dataset, tuple[PCA, int]] = dict()
    _name = "Principal Components Analysis"
    _identifier = "pca"
    _labels = ["scalar", "linear"]

    def __init__(self, components: Union[int, Iterable[int]]):
        self._cached_pca = None
        
        self._components = tuple(range(1, components + 1)) if isinstance(components, int) \
            else tuple(components)

        self._n_components = max(self._components)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
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

    _name = "Principal Components Analysis - Variance Explained Ratio"
    _identifier = "pca-var"
    _labels = ["scalar", "linear"]

    def __init__(self, components: list[int]):
        super().__init__(components=components)

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        pca = self._get_pca(data)
        indices = [i - 1 for i in self._components]
        return pca.explained_variance_ratio_[indices]

class PCAEigenVectors(PCABase):

    _name = "Principal Components Analysis - Eigen Vectors"
    _identifier = "pca-eig"
    _labels = ["vector", "linear"]

    def __init__(self, principal_vectors: list[int]):
        super().__init__(components=principal_vectors)

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        pca = self._get_pca(data)
        evectors = pca.components_
        indices = [i - 1 for i in self._components]
        return evectors[indices]

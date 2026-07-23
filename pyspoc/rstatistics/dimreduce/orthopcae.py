from typing import Any, Union
from numpy import ndarray

from pyspoc.statistics.dimreduce.orthopcae.statistic import OrthogonalPCAEStatistic


class OrthogonalPCAEVarianceExplained(OrthogonalPCAEStatistic):
        
    _name = "Orthogonal Principal Component Autoencoder - Variance Explained Ratio"
    _identifier = "opcae-var"
    _labels = ["vector", "non-linear"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)
    
    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_result(self, results: dict[str, Any]) -> Union[ndarray, float]:
        return results["pseudo_variance_explained"]


class OrthogonalPCAEVarianceElbow(OrthogonalPCAEStatistic):
        
    _name = "Orthogonal Principal Component Autoencoder Variance Explained Elbow"
    _identifier = "opcae-var-elbow"
    _labels = ["scalar", "non-linear"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)
    
    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_result(self, results: dict[str, Any]) -> Union[ndarray, float]:
        return results["optimal_bottleneck_dimension"]
    


import numpy as np

from typing import Any

from ._base import OrthogonalPCAEReducedStatistic


class OrthogonalPCAEVarianceExplained(OrthogonalPCAEReducedStatistic):
        
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

    def _get_result(self, results: dict[str, Any], components: tuple[int, ...]
                    ) -> np.ndarray | float:
       var_explained = results["pseudo_variance_explained"]
       indices = np.asarray(components, dtype=int) - 1
       return var_explained[indices]


class OrthogonalPCAEVarianceElbow(OrthogonalPCAEReducedStatistic):
        
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

    def _get_result(self, results: dict[str, Any], components: tuple[int, ...]
                    ) -> np.ndarray | float:
        return results["optimal_bottleneck_dimension"]
    


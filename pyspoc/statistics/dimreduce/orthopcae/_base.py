import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Union

from pyspoc.statistic import Statistic
from ._mixin import OrthogonalPCAEMixin


class OrthogonalPCAEStatistic(OrthogonalPCAEMixin, Statistic, ABC):

    def compute(self, data: np.ndarray) -> np.ndarray | float:

        resolved_parameters = self._resolve_parameters(data)
        results = self._compute_estimator_output(data, resolved_parameters)
        return self._get_result(results, resolved_parameters.components)
        

    @abstractmethod
    def _get_result(self, results: dict[str, Any], components: tuple[int, ...]
                    ) -> Union[np.ndarray, float]:
        pass

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

from pyspoc.statistic import Statistic
from ._mixin import OrthogonalPCAEMixin

if TYPE_CHECKING:
    from ._estimator import OrthogonalPCAEEstimator


class OrthogonalPCAEStatistic(OrthogonalPCAEMixin, Statistic, ABC):

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        resolved_parameters = self._resolve_parameters(data)
        fitted_estimator = self._compute_estimator_output(data, resolved_parameters)
        return self._get_result(fitted_estimator, resolved_parameters.components)
        

    @abstractmethod
    def _get_result(
        self,
        fitted_estimator: OrthogonalPCAEEstimator,
        components: tuple[int, ...]) -> Union[np.ndarray, float]:

        pass

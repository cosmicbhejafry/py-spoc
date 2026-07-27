from typing import Any

from numpy import ndarray

from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._estimators import LazyFittedCachedEstimatorMixin

class KMeansEstimator(
    RuntimeTypeCheckedMixin,
    LazyFittedCachedEstimatorMixin):

    def __init__(self):
        pass

    def _compute_fitted(self, data: ndarray) -> Any:
        return super()._compute_fitted(data)

    def _fit_estimator(self, data: ndarray) -> None:
        return super()._fit_estimator(data)

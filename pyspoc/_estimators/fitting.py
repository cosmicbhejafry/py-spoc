import numpy as np

from typing import TypeVar, Generic, final
from abc import ABC, abstractmethod
from threading import RLock

from .caching import CachedEstimatorMixin


_TResult = TypeVar("_TResult")

class LazyFittedCachedEstimatorMixin(
    CachedEstimatorMixin,
    ABC,
    Generic[_TResult]):
    """Provide thread-safe, fit-on-first-computation behaviour."""

    def _initialize_estimator_state(self):
        super()._initialize_estimator_state()
        self._initialize_fitting_state()

    def _initialize_fitting_state(self):
        self._pyspoc_is_fitted = False
        self._pyspoc_fitting_lock = RLock()

    @final
    @CachedEstimatorMixin._updates_lru
    def compute(self, data: np.ndarray) -> _TResult:
        """Fit when necessary and then calculate estimator outputs."""
        self._ensure_fitted(data)
        return self._compute_fitted(data)
    

    def _ensure_fitted(self, data: np.ndarray) -> None:
        if self._pyspoc_is_fitted:
            self._validate_fitted_data(data)
            return

        with self._pyspoc_fitting_lock:
            if self._pyspoc_is_fitted:
                self._validate_fitted_data(data)
                return

            self._prepare_fitting_data(data)
            self._fit_estimator(data)

            # Publish fitted state only after fitting completes successfully.
            self._pyspoc_is_fitted = True


    def _prepare_fitting_data(self, data: np.ndarray) -> None:
        attached_data = self._get_attached_dataset()

        if attached_data is None:
            self._set_attached_dataset(data)
        elif not type(self)._is_data_match(self, data):
            raise ValueError(
                "Input data does not match the dataset associated "
                "with this estimator."
            )

    def _validate_fitted_data(self, data: np.ndarray) -> None:
        if not type(self)._is_data_match(self, data):
            raise ValueError(
                "Input data does not match the dataset used for fitting."
            )

    @abstractmethod
    def _fit_estimator(self, data: np.ndarray) -> None:
        """Fit the concrete estimator."""
        pass

    @abstractmethod
    def _compute_fitted(
        self,
        data: np.ndarray,) -> _TResult:
        """Calculate outputs using the fitted estimator."""
        pass

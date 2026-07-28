import numpy as np

from typing import TypeVar, final
from abc import ABC, abstractmethod
from threading import RLock

from .caching import CachedEstimatorMixin


_TEstimator = TypeVar("_TEstimator", bound="LazyFittedCachedEstimatorMixin")

class LazyFittedCachedEstimatorMixin(
    CachedEstimatorMixin,
    ABC):
    """Provide thread-safe, fit-on-first-computation behaviour."""

    def _after_component_init(self, init_args: dict[str, object]) -> None:
        """Initialize fitting state after successful construction."""
        super()._after_component_init(init_args)
        self._pyspoc_is_fitted = False
        self._pyspoc_fitting_lock = RLock()

    @final
    @CachedEstimatorMixin._updates_lru
    def fit(self: _TEstimator, data: np.ndarray) -> _TEstimator:
        """Fit when necessary and then calculate estimator outputs."""
        self._ensure_fitted(data)
        return self
    

    def _ensure_fitted(self, data: np.ndarray):
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

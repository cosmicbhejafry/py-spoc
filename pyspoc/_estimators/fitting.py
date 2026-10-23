import numpy as np

from typing import Self, Any, TypeVar, Generic, final
from abc import ABC, abstractmethod
from threading import RLock

from .caching import CachedEstimatorMixin


TModel = TypeVar("TModel")


class LazyFittedCachedEstimatorMixin(
    CachedEstimatorMixin,
    Generic[TModel],
    ABC):
    """Provide thread-safe, fit-on-first-computation behaviour. The method
    :meth:`fit` performs model fitting and stores the fitted model in the private
    field :attr:`_model_` for later use.
    """

    def __init__(self):
        self._model_: TModel | None = None
    
    
    def _after_component_init(self, init_args: dict[str, object]):
        """Initialize fitting state after successful construction."""
        super()._after_component_init(init_args)
        self._pyspoc_is_fitted = False
        self._pyspoc_fitting_lock = RLock()

    @final
    @CachedEstimatorMixin._updates_lru
    def fit(self, data: np.ndarray) -> Self:
        """Fit when necessary and then calculate estimator outputs."""
        self._ensure_fitted(data)
        return self

    def _get_model(self) -> TModel:
        """Return the fitted private model.

        Returns
        -------
        TModel
            Fitted model owned by this estimator for further inspection.

        Raises
        ------
        ValueError
            If :meth:`fit` has not yet completed successfully.

        Notes
        -----
        The returned model may underpin multiple cached Statistic objects and
        should therefore be treated as read-only.
        """
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return self._model_
    

    def _ensure_fitted(self, data: np.ndarray):
        if self._pyspoc_is_fitted:
            self._validate_fitted_data(data)
            return

        with self._pyspoc_fitting_lock:
            if self._pyspoc_is_fitted:
                self._validate_fitted_data(data)
                return

            self._prepare_fitting_data(data)
            model = self._fit_estimator(data)
            self._model_ = model

            # Publish fitted state only after fitting completes successfully.
            self._pyspoc_is_fitted = True


    def _prepare_fitting_data(self, data: np.ndarray):
        attached_data = self._get_attached_dataset()

        if attached_data is None:
            self._set_attached_dataset(data)
            
        elif not type(self)._is_data_match(self, data):
            raise ValueError(
                "Input data does not match the dataset associated "
                "with this estimator."
            )

    def _validate_fitted_data(self, data: np.ndarray):
        if not type(self)._is_data_match(self, data):
            raise ValueError(
                "Input data does not match the dataset used for fitting."
            )

    @abstractmethod
    def _fit_estimator(self, data: np.ndarray) -> Any:
        """Fit the concrete estimator."""
        pass

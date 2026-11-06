"""Lazy, single-flight fitting of models for ordered pairs of data columns."""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from operator import index as integer_index
from threading import RLock
from typing import Generic, Self, TypeVar, final

import numpy as np

from .caching import CachedEstimatorMixin


TModel = TypeVar("TModel")


class LazyFittedPairwiseCachedEstimatorMixin(
    CachedEstimatorMixin,
    Generic[TModel],
    ABC):
    """Cache an independently fitted model for each ordered column pair.

    ``get_or_create(data, ...)`` retains the existing dataset/constructor-based
    estimator cache. Unlike the whole-dataset fitting mixin, :meth:`fit` only
    attaches and validates data; :meth:`_get_model` fits a requested pair on first
    access. A Statistic can call it from ``_pairwise_compute_indexed`` so the
    existing pairwise executor supplies the parallelism; no nested pool is used.

    Pair keys refer to columns of the exact prepared matrix attached here, not
    necessarily the original input before axis selection, omission or sorting.
    Reversed pairs have separate models, even for symmetric Statistic outputs.
    This avoids assuming that fitted model internals are themselves symmetric.

    Notes
    -----
    Concrete implementations must be safe to fit distinct pairs concurrently
    before a Statistic declares ``thread_safety="safe"``. Treat input views and
    returned shared models as read-only. All requested models remain in memory
    until this estimator is evicted or released; this is not a memory-bounded
    kernel/barrier cache. Recursive requests for the same pending pair are not
    supported.
    """

    def _after_component_init(self, init_args: dict[str, object]) -> None:
        """Initialize independent pair state after the complete constructor.

        Parameters
        ----------
        init_args : dict[str, object]
            Normalized constructor arguments supplied by constructor capture.
        """
        super()._after_component_init(init_args)
        # Initialize here so concrete estimators need not call a mixin initializer.
        self._pair_models: dict[tuple[int, int], Future[TModel]] = {}
        self._pair_models_lock = RLock()

    @final
    @CachedEstimatorMixin._updates_lru
    def fit(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> Self:
        """Attach a matrix for lazy pair fitting, or validate an existing one.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional prepared matrix with objects compared along columns.

        Returns
        -------
        Self
            This estimator. No pair models are computed by this call.

        Raises
        ------
        ValueError
            If the input is not two-dimensional or differs from attached data.
        """
        if data.ndim != 2:
            raise ValueError("Pairwise fitting requires a two-dimensional array.")
        
        with self._pair_models_lock:
            if self._get_attached_dataset() is None:
                self._set_attached_dataset(data)

            elif not type(self)._is_data_match(self, data):
                raise ValueError(
                    "Input data does not match the dataset associated with this estimator.")
        return self
    

    @CachedEstimatorMixin._updates_lru
    def _get_model(
        self,
        index: tuple[int, int]) -> TModel:
        """Return a pair's model, fitting it once if necessary.

        Parameters
        ----------
        index : tuple[int, int]
            Nonnegative ``(i, j)`` column indices in the attached matrix.

        Returns
        -------
        TModel
            Shared model for the ordered pair. Concurrent callers for the same
            pair wait for the owner's Future rather than duplicating fitting.

        Raises
        ------
        ValueError
            If no dataset has been attached by ``fit`` or ``get_or_create``.
        IndexError
            If either index lies outside the matrix's columns.
        TypeError
            If the key is not a pair of integer indices.

        Notes
        -----
        Fitting failures propagate to all callers already waiting for that pair.
        The failed entry is removed, allowing a subsequent request to retry.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("Model index must be an (i, j) tuple.")
        
        if any(isinstance(value, (bool, np.bool_)) for value in index):
            raise TypeError("Model indices must be integers, not booleans.")
        
        key = (integer_index(index[0]), integer_index(index[1]))

        with self._pair_models_lock:
            data = self._get_attached_dataset()
            
            if data is None:
                raise ValueError("No dataset attached. Call fit() or get_or_create() first.")
            
            if data.ndim != 2:
                raise ValueError("Pairwise fitting requires a two-dimensional array.")
            
            if any(value < 0 or value >= data.shape[1] for value in key):
                raise IndexError(f"Pair index {key} is outside {data.shape[1]} columns.")

            # Register ownership under a short lock. Never hold it during fitting
            # or Future.result(): unrelated pairs must be free to make progress.
            future = self._pair_models.get(key)
            is_owner = future is None

            if future is None:
                future = Future()
                self._pair_models[key] = future

        if not is_owner:
            return future.result()

        try:
            # Read-only views discourage accidental cross-thread modification
            # without allocating another copy of each column for every pair.
            x, y = data[:, key[0]].view(), data[:, key[1]].view()
            x.flags.writeable = False
            y.flags.writeable = False
            model = self._fit_estimator(x, y)

        except BaseException as error:
            # Publish the failure before making the key available for retry.
            # Existing waiters keep their reference to this failed Future.
            with self._pair_models_lock:
                future.set_exception(error)
                del self._pair_models[key]
            raise
        else:
            future.set_result(model)
            return model


    @abstractmethod
    def _fit_estimator(
        self,
        x: np.ndarray[tuple[int], np.dtype[np.floating]],
        y: np.ndarray[tuple[int], np.dtype[np.floating]]) -> TModel:
        """Fit a model for one ordered pair of read-only column vectors.

        Parameters
        ----------
        x, y : numpy.ndarray
            One-dimensional views of the attached prepared matrix.

        Returns
        -------
        TModel
            Model to retain for this pair; it must not modify shared inputs.
        """
        raise NotImplementedError

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import TypeGuard, Literal, Any, overload

from pyspoc.statistics.base import StatisticBase
from pyspoc.settings import settings
from pyspoc.dataset import Dataset


class ReducedStatistic(StatisticBase, ABC):

    """
    Abstract Statistic class for computing a fully reduced statistical output, acting as both a
    Statistic and Reducer in a single operation.

    Output is NOT subject to further processing by applicable Reducers and only flattened,
    if required.
    """

    def calculate(self, dataset: Dataset
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        # temporarily uncache results, TODO: fix caching mechanisms
        self.uncache(dataset)

        result = super().calculate(dataset)

        if isinstance(result, np.ndarray):
            return result.flatten()

        return result
    

    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        """
        Compute the ReducedStatistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[tuple[int, int], np.dtype[np.floating]]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.
            

        Returns
        -------
        `np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] | \
         np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] | \
         np.floating | np.integer | \
         float | int`
            ReducedStatistic estimate.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            dataset_nan_policy: Literal["raise", "row-omit", "col-omit"]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        """
        Compute the ReducedStatistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[tuple[int, int], np.dtype[np.floating]]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.
            

        Returns
        -------
        `np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] | \
         np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] | \
         np.floating | np.integer | \
         float | int`
            ReducedStatistic estimate.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        """
        Compute the ReducedStatistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[tuple[int, int], np.dtype[np.floating]]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.
            
        ## Optional Parameters

        kwargs: Any
            Additional Statistic-specific keyword arguments used during computation.

        Returns
        -------
        `np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] | \
         np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] | \
         np.floating | np.integer | \
         float | int`
            ReducedStatistic estimate.
        """
        ...


    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        dataset_nan_policy = kwargs.get("dataset_nan_policy", settings.current.data_nan_policy)

        result = super().compute(
            data, dataset_nan_policy=dataset_nan_policy, kwargs=kwargs)

        if self._is_1d_or_less(result):
            return result

        raise ValueError(
            f"compute() returned a result of incorrect type: {type(result)}, "
            "expected: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]")
    
    
    @staticmethod
    def _is_1d_or_less(result: np.ndarray
    ) -> TypeGuard[
        np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
        np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
        np.floating | np.integer |
        float | int]:

        numpy_check = (isinstance(result, np.ndarray)
            and result.ndim < 2
            and (np.issubdtype(result.dtype, np.floating)
                 or np.issubdtype(result.dtype, np.integer)))

        if numpy_check:
            return numpy_check

        return (
            isinstance(result, np.floating) or # NOTE: Possibly redundant.
            isinstance(result, float) or
            isinstance(result, np.integer) or
            isinstance(result, int))


    @abstractmethod
    def _compute(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        pass


    @staticmethod
    def _get_component_type() -> type:
        return ReducedStatistic

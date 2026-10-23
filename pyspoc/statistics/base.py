from __future__ import annotations

import numpy as np
import gc
import warnings

from abc import ABC, abstractmethod
from typing import Any, TypeGuard, Literal, TypeAlias, overload
from types import FunctionType, MethodType
from pathlib import Path

from pyspoc.settings import settings
from pyspoc.dataset import Dataset
from pyspoc._tag import Tag, Type, Signature
from pyspoc._base import Component
from pyspoc._execution import execute_pairwise


StatisticResultNDArray: TypeAlias = (
    np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]] |
    np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
    np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
    np.floating | np.integer |
    float | int)



class StatisticBase(Component, ABC):

    """
    Base abstract Statistic class intended for computing a statistic from an entire static dataset
    (n x p). Provides a square matrix (p x p) output subject to further processing by applicable
    Reducers.

    Properties
    ----------
    name : string
        Readable name of the statistic.
    identifier : string
        A simpler and minimalist identifier for the statistic.
    labels : list[str]
        A list of labels to describe the type of statistic.

    Contracts
    ----------
        Input: (n x p) -> Output: (p x p)
    """

    _cached_results: dict[Dataset,
                          dict[dict[str, Any], StatisticResultNDArray]] = {}
    _warned_properties_missing = False
    _warned_properties_error = False

    @classmethod
    def _get_property_schema(cls) -> tuple[
        tuple[str, type, MethodType | FunctionType], ...]:
        
        return (
            ("long_name", str, cls.get_long_name),
            ("identifier", str, cls.get_identifier),
            ("type", Type, cls.get_type),
            ("tags", Tag, cls.get_tags),
            ("signatures", Signature, cls.get_signatures))

    @classmethod
    def _get_properties_yaml_path(cls) -> Path:
        return settings.current.statistic_properties_yaml_path
    

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cached_results = {}


    def __init__(
            self,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"],
            *,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe"):
        
        self._cached_result = None

        # Kwargs store for temporary calculate-level and compute-level arguments.
        self._calculate_kwargs = dict()
        self._compute_kwargs = dict()

        self._dim = dim
        self._nan_policy = nan_policy

        super().__init__(
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)


    @property
    def dim(self) -> str:
        return self._dim


    @property
    def nan_policy(self) -> str:
        return self._nan_policy

        
    def calculate(self, dataset: Dataset, **kwargs: Any
    ) -> (np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):

        self._calculate_kwargs = kwargs

        # Get result from instance cache if present.
        if self._cached_result:
            return self._cached_result

        # TODO: Implement class-wide Statistic caching.
        # Get result from class cache if present.
        # cached_result = self._get_cached_result(dataset)

        # if cached_result:
        #    return cached_result

        # Else compute from scratch with a copy of the data.
        data = dataset.get_data()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = self.compute(data, dataset_nan_policy=dataset.nan_policy)

            if captured and self._active_calculator is not None:
                self._active_calculator._add_warnings(captured)

        result = self._get_readonly_result_copy(result)

        # TODO: Implement class-wide Statistic caching.
        # Cache result in the class cache.
        # self._set_cached_result(dataset, result)

        # Cache result at instance level.
        self._cached_result = result
        self._calculate_kwargs = dict()
        return result
    
        
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> (np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        """
        Base compute method serving public-facing overloads. Computes the Statistic for the given
        point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.


        Returns
        -------
        :class:`StatisticResultNDArray`
            Statistic estimate.
        """

        self._compute_kwargs = kwargs
        dataset_nan_policy = kwargs.get("dataset_nan_policy")
        is_direct_compute = dataset_nan_policy is None
        matrix = np.asarray(data, copy=True) if is_direct_compute else data
        matrix = self._set_dim(matrix)

        if dataset_nan_policy == self._nan_policy:
            return self._compute(matrix)

        nan_rows = np.isnan(matrix).any(axis=1)
        has_nans = bool(nan_rows.any())

        if self._nan_policy == "raise" and has_nans:
            raise ValueError("The data matrix contains NaN values.")

        if has_nans:
            if self._nan_policy == "row-omit":
                matrix = matrix[~nan_rows]
            else:
                nan_cols = np.isnan(matrix).any(axis=0)
                matrix = matrix[:,~nan_cols]

        return self._compute(matrix)
    

    def _set_dim(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        
        if self._dim == "p":
            return data

        return data.T


    @abstractmethod
    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> StatisticResultNDArray:
        pass


    def _get_readonly_result_copy(
            self,
            result: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> StatisticResultNDArray:
        
        result_copy = np.array(result, copy=True)
        result_copy.flags.writeable = False
        return result_copy
    

    @classmethod
    def uncache(cls, dataset: Dataset, include_gc: bool = False):
        cached_dataset_results = cls._cached_results.get(dataset)

        # TODO: This needs to be reworked to clear Reducer cache without
        # instigating a circular import.
        if cached_dataset_results:
            for statistic in cached_dataset_results.keys():
                #Reducer.clear_cache(statistic, include_gc)
                pass

            cls._cached_results[dataset] = dict()

        if include_gc:
            gc.collect()


    def _get_cached_result(self, dataset: Dataset
    ) -> StatisticResultNDArray | None:
        T = type(self)
        datasets_dict = T._cached_results.get(dataset)

        if datasets_dict is None:
            return

        result = datasets_dict.get(self._params)
        return result


    def _set_cached_result(
            self,
            dataset: Dataset,
            result: StatisticResultNDArray):
        
        T = type(self)

        if dataset not in T._cached_results:
            dataset_dict = {}
            self._cached_results[dataset] = dataset_dict

        dataset_dict[self._params] = result


    def _get_result(self) -> StatisticResultNDArray | None:
        return self._cached_result



class Statistic(StatisticBase, ABC):

    """
    Base abstract Statistic class intended for computing a statistic from an entire static dataset
    (n x p). Provides a square matrix (p x p) output subject to further processing by applicable
    Reducers.

    Properties
    ----------
    name : string
        Readable name of the statistic.
    identifier : string
        A simpler and minimalist identifier for the statistic.
    labels : list[str]
        A list of labels to describe the type of statistic.

    Contracts
    ----------
        Input: (n x p) -> Output: (p x p)
    """
    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        """
        Compute the Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.
            

        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]`
            Statistic estimate.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            dataset_nan_policy: Literal["raise", "row-omit", "col-omit"]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        """
        Compute the Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.

        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]`
            Statistic estimate.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        """
        Compute the Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
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
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]`
            Statistic estimate.
        """
        ...


    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        dataset_nan_policy = kwargs.get("dataset_nan_policy", settings.current.data_nan_policy)

        result = super().compute(
            data, dataset_nan_policy=dataset_nan_policy, kwargs=kwargs)

        if self._is_2d(result):
            return result

        raise ValueError(
            f"compute() returned a result of incorrect type: {type(result)}, "
            "expected: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]")
    
    
    @staticmethod
    def _is_2d(result: np.ndarray
    ) -> TypeGuard[np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]]:
        return (isinstance(result, np.ndarray)
            and result.ndim == 2
            and (np.issubdtype(result.dtype, np.floating)
                 or np.issubdtype(result.dtype, np.integer)))


    @abstractmethod
    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        pass


    @staticmethod
    def _get_component_type() -> type:
        return Statistic



class PairwiseStatistic(Statistic):
    """
    Abstract Statistic class intended for computing statistics from pairwise comparisons over a
    static dataset (n x p) or time series dataset (n x p x t).

    Comparisons can be made between observations, variables or time processes based on instantiation
    arguments.

    Data can be ordered if required.

    Provides a square matrix (p x p, n x n, t x t) output subject to further processing by
    applicable Reducers.

    Parameters
    ----------
    dim : string
        Declares the data axis to perform pairwise comparisons over.
        For example, if dim="n", computation is applied to all possible observation pairings,
        resulting in n^2 comparisons and an n x n matrix result.

    is_ordered : boolean
        Declares whether the statistic requires ordered data (ie: the Wilcoxon signed-rank test).
        If False, computation is performed on the data as is.
        If True, the data along the dim axis is ordered first before computation.
    
    Contracts
    ----------
        Input: (n x p), dim: n -> Output: (n x n)
        Input: (n x p), dim: p -> Output: (p x p)
        Input: (n x p), dim: t -> Output: None
        Input: (n x p x t), dim: n -> Output: (n x n)
        Input: (n x p x t), dim: p -> Output: (p x p)
        Input: (n x p x t), dim: t -> Output: (t x t)
    """

    def __init__(
            self,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"],
            is_ordered: bool = False,
            symmetry_type: Literal["exact", "negative", "reciprocal", "triangular"] | None = None,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe"):

        self._is_ordered = is_ordered
        self._symmetry_type = symmetry_type
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

    @property
    def is_ordered(self) -> bool:
        return self._is_ordered

    @property
    def symmetry_type(self) -> str | None:
        return self._symmetry_type
    

    @abstractmethod
    def _pairwise_compute(
        self,
        x: np.ndarray[tuple[int], np.dtype[np.floating]],
        y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> (np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating | np.integer |
          float | int):
        pass


    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        """ Compute statistics over all pairwise permutations.
        """
        if self._dim == "n":
            data = data.T

        if self._is_ordered:
            data = np.sort(data, axis=0)

        max_workers = (1
            if self._thread_safety == "unsafe" or self._internal_parallelism == "parallel"
            else None)

        error_policy = settings.current.pairwise_error_policy
        throw_first_pairwise_error = error_policy == "first"
        tqdm_pos = self._calculate_kwargs.get("tqdm_pos")
       
        result, exceptions = execute_pairwise(
            data,
            self._pairwise_compute,
            max_workers=max_workers,
            symmetry_type=self._symmetry_type,
            throw_first_error=throw_first_pairwise_error,
            tqdm_pos=tqdm_pos)

        if exceptions and error_policy == "all":
            raise ExceptionGroup(
                "One or more exceptions occurred during "
                f"{self.__class__.__qualname__}.compute()",
                exceptions)

        return result

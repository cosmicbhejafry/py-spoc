from __future__ import annotations

import numpy as np
import gc
import warnings

from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING

from pyspoc.reducers.base import Reducer
from pyspoc._base import Component
from pyspoc._execution import execute_pairwise


if TYPE_CHECKING:
    from pyspoc.dataset import Dataset
    

class Statistic(Component, ABC):

    """
    Base abstract Statistic class intended for computing a statistic from an entire static dataset (n x p).
    Provides a square matrix (p x p) output subject to further processing by applicable Reducers.

    Required Properties
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

    _cached_results: dict[Dataset, dict[Statistic, np.ndarray]] = dict()

    def __init__(self,
                 internal_parallelism: Literal["serial", "parallel"] = "serial",
                 thread_safety: Literal["safe", "unsafe"] = "unsafe"):
        self._cached_result = None
        super().__init__(
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety
        )

    def calculate(self, dataset: Dataset) -> np.ndarray:

        # temporarily uncache results, TODO: fix caching mechanisms
        self.uncache(dataset)

        # Get result from class cache if present.
        dataset_results = self._cached_results.get(dataset)

        if dataset_results is not None:
            result = dataset_results.get(self)

            if result is not None:
                return result

        # Else compute from scratch with a copy of the data.
        data = dataset.get_data()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = self.compute(data)

            if captured and self._active_calculator is not None:
                self._active_calculator._add_warnings(captured)

        result = self._get_readonly_result_copy(result)

        # Cache result in the hierarchy.
        if dataset_results is None:
            self._cached_results[dataset] = {
                self: result
            }
        else:
            dataset_results[self] = result

        # Cache result locally.
        self._cached_result = result
        return result
    

    @abstractmethod
    def compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]):
        pass


    def _get_readonly_result_copy(self, result: np.ndarray) -> np.ndarray:
        result_copy = np.array(result, copy=True)
        result_copy.flags.writeable = False
        return result_copy
    

    @classmethod
    def uncache(cls, dataset: Dataset, include_gc: bool = False):
        cached_dataset_results = cls._cached_results.get(dataset)

        if cached_dataset_results:
            for statistic in cached_dataset_results.keys():
                Reducer.clear_cache(statistic, include_gc)

            cls._cached_results[dataset] = dict()

        if include_gc:
            gc.collect()


    def get_result(self) -> np.ndarray | None:
        return self._cached_result
    

    @staticmethod
    def _get_component_type() -> type:
        return Statistic


class PairwiseStatistic(Statistic):

    """
    Abstract Statistic class intended for computing statistics from pairwise comparisons over a static dataset (n x p)
    or time series dataset (n x p x t).

    Comparisons can be made between observations, variables or time processes based on instantiation arguments.

    Data can be ordered if required.

    Provides a square matrix (p x p, n x n, t x t) output subject to further processing by applicable Reducers.

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

    def __init__(self,
                 dim: Literal["n", "p"] = "p",
                 is_ordered: bool = False,
                 symmetry_type: Literal["exact", "negative", "reciprocal"] | None = None,
                 internal_parallelism: Literal["serial", "parallel"] = "serial",
                 thread_safety: Literal["safe", "unsafe"] = "unsafe"
                 ):

        self._dim = dim
        self._is_ordered = is_ordered
        self._symmetry_type = symmetry_type
        super().__init__(
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

    @property
    def dim(self) -> str:
        return self._dim

    @property
    def is_ordered(self) -> bool:
        return self._is_ordered

    @property
    def symmetry_type(self) -> str | None:
        return self._symmetry_type
    

    @abstractmethod
    def pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> np.ndarray | float:
        pass

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        """ Compute statistics over all pairwise permutations.
        """
        if self._dim == "n":
            data = data.T

        if self._is_ordered:
            data = np.sort(data, axis=0)

        max_workers = (1
            if self._thread_safety == "unsafe" or self._internal_parallelism == "parallel"
            else None)

        return execute_pairwise(
            data,
            self.pairwise_compute,
            max_workers=max_workers,
            symmetry_type=self._symmetry_type,
        )

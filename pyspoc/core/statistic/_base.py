from __future__ import annotations

import runpy
import gc
import pkgutil
import numpy as np
import warnings

from collections.abc import Collection
from abc import abstractmethod, ABC
from typing import final
from pathlib import Path

from pyspoc._caching.statistic import statistic_result_cache
from pyspoc.core._component import Component
from pyspoc.core.types import NumpyFloatCubicTensorUpTo2D
from pyspoc.dataset import Dataset
from pyspoc.settings import settings


class Statistic(Component, ABC):

    """
    Base abstract Statistic class intended for computing a statistic from an entire
    static dataset (n x p). Provides a square matrix (p x p) output subject to further
    processing by applicable Reducers.

    Required Properties
    ----------
        name : string
            Readable name of the statistic.
        identifier : string
            A simpler and minimalist identifier for the statistic.
        labels : list[str]
            A list of labels to describe the type of statistic.
    """

    def __init__(self, short_name: str, labels: Collection[str]):
        self._cached_results: dict[Dataset, NumpyFloatCubicTensorUpTo2D] = dict()
        super().__init__(short_name=short_name, labels=labels)


    @final
    @classmethod
    def get_available_statistics(cls):
        glb_copy = globals()
        stats = set()

        for obj in glb_copy.values():
            obj_cls = type(obj)

            if issubclass(obj_cls, cls):
                stats.add(obj_cls)

        this_path = Path(__file__)
        dir_path = this_path.parent
        paths = [str(dir_path / "statistics")]

        while paths:
            path = paths.pop()
            mods = list(pkgutil.iter_modules([path]))

            for mod in mods:
                mod_path = path / (mod.name + ".py")
                loaded_mod = runpy.run_path(str(mod_path))

                for obj, val in loaded_mod.items():
                    print(obj, val)
   

    @final
    @classmethod
    def clear_cache(cls, dataset: Dataset, include_gc: bool = False):
        statistic_result_cache.clear_dataset(dataset)

        # Reducer entries are keyed independently by Dataset, so invalidation
        # remains complete even if a Statistic entry was already evicted.
        from pyspoc.core.reducer._base import Reducer

        Reducer.clear_cache(dataset, include_gc=False)

        if include_gc:
            gc.collect()


    @final
    def calculate_summary(self, dataset: Dataset) -> NumpyFloatCubicTensorUpTo2D:

        # Get result from class cache if present.
        if settings.current.statistic_caching:
            cached_result = self._get_cached_result(dataset)
        
            if cached_result is not None:
                return cached_result

        # Else compute from scratch with a copy of the data.
        data = dataset.get_data()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = self.summarize(data)

            if captured and self._active_calculator is not None:
                self._active_calculator._add_warnings(captured)

        result = self._prepare_component_result(result)

        # Cache result in the hierarchy.
        if settings.current.statistic_caching:
            self._cache_result(dataset, result)
        
        return result
    

    def summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyFloatCubicTensorUpTo2D:

        # NumPy's native generic annotation communicates rank and dtype to
        # static tooling, but Typeguard does not currently enforce those
        # parameters at runtime. Preserve the public contract explicitly at
        # this boundary before concrete numerical code receives the input.
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"{self.__class__.__name__} {self.name} received invalid data type. "
                "Input data must be a NumPy array.")
        
        if data.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} {self.name} received invalid data shape. "
                f"Input data must be two-dimensional, but got shape {data.shape}."
            )
        
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError(
                f"{self.__class__.__name__} {self.name} received invalid Numpy dtype. "
                f"Input data must have a floating dtype, but got {data.dtype}."
            )

        # Compute the statistic and return.
        result = self._summarize(data)

        # Perform validation based on Statistic return type.
        result = self._get_validated_result(result)

        return result


    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyFloatCubicTensorUpTo2D:
        pass
   

    @final
    def _get_cached_result(self, dataset: Dataset) -> NumpyFloatCubicTensorUpTo2D | None:
        return statistic_result_cache.get(self, dataset)


    @final
    def _cache_result(self, dataset: Dataset, result: NumpyFloatCubicTensorUpTo2D):
        statistic_result_cache.store(
            self,
            dataset,
            result,
            settings.current.max_statistic_cache_memory_fraction,
        )
        
    @abstractmethod
    def _get_validated_result(
        self,
        result: NumpyFloatCubicTensorUpTo2D) -> NumpyFloatCubicTensorUpTo2D:
        """Validate and return a value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : :class:`NumpyFloatCubicTensorUpTo2D`
            Candidate statistic result. It must be a cubic float tensor of at
            most two dimensions, hence the following results are accepted:
            - a `float` scalar.
            - an :class:`np.floating` scalar.
            - a `float` vector: a one-dimensional Numpy array of
                :class:`np.floating` dtype.
            - a `float` square matrix: a two-dimensional Numpy array of
                :class:`np.floating` dtype with equal row and column counts.

        Returns
        -------
        :class:`NumpyFloatCubicTensorUpTo2D`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a float or Numpy cubic tensor of at most
            two dimensions.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`compute` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """
        pass
    

    @classmethod
    def _get_component_type(cls) -> type:
        return Statistic

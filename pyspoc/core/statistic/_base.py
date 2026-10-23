from __future__ import annotations

import runpy
import gc
import pkgutil
import numpy as np
import warnings

from collections.abc import Collection
from abc import abstractmethod, ABC
from typing import Generic, TypeVar, ClassVar, Any, final, get_origin, get_args
from types import get_original_bases
from typeguard import check_type
from pathlib import Path

from pyspoc._caching.statistic import statistic_result_cache
from pyspoc.core._component import Component
from pyspoc.core.types import NumpyRealTensorUpTo2D, NumpyRealMatrix
from pyspoc.core.iovalidation import MatrixValidator
from pyspoc.core.iovalidation._base import Validator
from pyspoc.dataset import Dataset
from pyspoc.settings import settings


StatisticOutputT = TypeVar(
    "StatisticOutputT",
    bound=NumpyRealTensorUpTo2D,
    covariant=True)


class Statistic(
    Component,
    Generic[StatisticOutputT],
    ABC):

    """
    Base abstract Statistic class intended for computing a statistic from an entire
    static dataset (n x p). Provides a square matrix (p x p) output subject to further
    processing by applicable Reducers.

    Required Properties
    ----------
        full_name : str
            Readable name of the statistic.
        identifier : str
            A simpler and minimalist identifier for the statistic.
        labels : list[str]
            A list of labels to describe the type of statistic.
    """

    _output_type = ClassVar[Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for declared_base in get_original_bases(cls):
            if get_origin(declared_base) is Statistic:
                cls._output_type = get_args(declared_base)[0]
                break


    def __init__(self, full_name: str, short_name: str, labels: Collection[str]):
        self._cached_results: dict[Dataset, StatisticOutputT] = dict()
        super().__init__(full_name=full_name, short_name=short_name, labels=labels)


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
    def calculate_summary(self, dataset: Dataset) -> StatisticOutputT:

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
            statistic_result_cache.store(
                self,
                dataset,
                result,
                settings.current.max_statistic_cache_memory_fraction,
            )
        
        return result
    

    def summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> StatisticOutputT:
        # NumPy's native generic annotation communicates rank and dtype to
        # static tooling, but Typeguard does not currently enforce those
        # parameters at runtime. We preserve the public contract explicitly at
        # this boundary but run validation on the jaxtyped object with _validate_input().
        if settings.current.result_validation == "both":
            self._validate_input(data)

        # Compute the statistic and return.
        result = self._summarize(data)

        # Perform validation based on Statistic return type.
        self._validate_output(result)

        return result


    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> StatisticOutputT:
        pass
   

    @final
    def _get_cached_result(self, dataset: Dataset) -> StatisticOutputT | None:
        return statistic_result_cache.get(self, dataset)

    @final
    def _get_validator(self) -> Validator[NumpyRealTensorUpTo2D]:
        output_type = type(self)._output_type
        return Validator._get_validator(output_type)


    def _validate_input(
            self,
            input: NumpyRealMatrix):
        """Validate the data passed into the Statistic.

        Parameters
        ----------
        input : :class:`NumpyFloatMatrix`
            Candidate statistic result. It must be a float matrix,
            hence the following results are accepted:
            - a `float` matrix: a two-dimensional Numpy array of
                :class:`np.floating` dtype.

        Raises
        ------
        TypeCheckError
            If ``data`` is not a two-dimensional Numpy array of floating type.

        Notes
        -----
        Subclasses can override with narrower jaxtyping annotations and
        additional semantic checks, calling ``super()._validate_input(input)``
        to retain the base contract.
        """

        MatrixValidator.validate(self, input)
    

    @abstractmethod
    def _validate_output(
            self,
            result: NumpyRealTensorUpTo2D):
        """Validate and return the value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : :class:`NumpyFloatCubicTensorUpTo2D`
            Candidate statistic result. It must be a cubic float tensor of at
            most two dimensions, hence the following results are accepted:
            - a `float` scalar.
            - an :class:`np.floating` scalar.
            - a `float` embedded scalar: a zero-dimensional Numpy array of
                :class:`np.floating` dtype.
            - a `float` vector: a one-dimensional Numpy array of
                :class:`np.floating` dtype.
            - a `float` matrix: a two-dimensional Numpy array of
                :class:`np.floating` dtype.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a float or Numpy cubic tensor of at most
            two dimensions.

        Notes
        -----
        Subclasses should override with narrower jaxtyping annotations and
        additional semantic checks, calling ``super()._validate_output(result)``
        to retain the base contract.
        """

        check_type(result, NumpyRealTensorUpTo2D)
    

    @classmethod
    def _get_component_type(cls) -> type:
        return Statistic

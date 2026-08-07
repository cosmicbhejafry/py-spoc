from __future__ import annotations

import runpy
import gc
import pkgutil

from collections.abc import Collection
from abc import abstractmethod, ABC
from typing import final, TYPE_CHECKING
from pathlib import Path
from typeguard import check_type

from pyspoc._core import Component
from pyspoc._core.types import NumpyDataMatrix, NumpyRealTensorAtMost2D
from pyspoc.reducer import Reducer
from pyspoc.settings import settings

if TYPE_CHECKING:
    from pyspoc.dataset import Dataset


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

    _cached_results: dict[Dataset, dict[Statistic, NumpyRealTensorAtMost2D]] = dict()

    def __init__(self, short_name: str, labels: Collection[str]):
        self._cached_result = None
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
    def uncache(cls, dataset: Dataset, include_gc: bool = False):
        cached_dataset_results = cls._cached_results.get(dataset)

        if cached_dataset_results:
            for statistic in cached_dataset_results.keys():
                Reducer.uncache(statistic, include_gc)

            cls._cached_results[dataset] = dict()

        if include_gc:
            gc.collect()


    @final
    def calculate(self, dataset: Dataset) -> NumpyRealTensorAtMost2D:

        if settings.current.statistic_caching:
            cached_result = self._get_cached_result(dataset)

            if cached_result is not None:
                return cached_result

        # Access the underlying data.
        data = dataset.get_data()

        # Compute the statistic and return.
        result = self.compute(data)

        # Perform validation based on Statistic return type.
        result = self._get_validated_result(result)

        # Cache result in the hierarchy.
        if settings.current.statistic_caching:
            self._cache_result(dataset, result)
        
        return result
    

    @final
    def _get_cached_result(self, dataset: Dataset) -> NumpyRealTensorAtMost2D | None:
        # temporarily uncache results, TODO: fix caching mechanisms
        self.uncache(dataset)

        # Get result from class cache if present.
        dataset_results = self._cached_results.get(dataset)

        if dataset_results is not None:
            result = dataset_results.get(self)

            if result is not None:
                return result


    @final
    def _cache_result(self, dataset: Dataset, result: NumpyRealTensorAtMost2D):
        # Get result from class cache if present.
        dataset_results = self._cached_results.get(dataset)

        if dataset_results is not None:
            dataset_results[self] = result
        else:
            self._cached_results[dataset] = {
                self: result
            }


    @abstractmethod
    def compute(self, data: NumpyDataMatrix) -> NumpyRealTensorAtMost2D:
        pass

        
    def _get_validated_result(
        self,
        result: NumpyRealTensorAtMost2D) -> NumpyRealTensorAtMost2D:
        """Validate and return a value produced by :meth:`compute`.

        Parameters
        ----------
        result : :class:`NumpyRealTensorAtMost2D`
            Candidate statistic result. It must be a Numpy array with real-valued
            dtype of at most two dimensions.

        Returns
        -------
        :class:`NumpyRealTensorAtMost2D`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a two-dimensional, real-valued Numpy array.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`compute` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """
        # Typeguard delegates the array-specific part of this annotation to
        # jaxtyping. Both symbolic axes occur in this single check, so no
        # shared argument/return dimension context is required here.
        check_type(result, NumpyRealTensorAtMost2D)

        # Validation must not copy or otherwise transform statistic results.
        return result

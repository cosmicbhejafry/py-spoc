from __future__ import annotations

import warnings
import numpy as np
import gc

from abc import ABC, abstractmethod
from types import MethodType, FunctionType
from typing import Iterable, TypeAlias, TypeVar, Generic, overload
from pathlib import Path
from dataclasses import dataclass

from pyspoc.statistics.base import Statistic, StatisticResultNDArray
from pyspoc._base import Component
from pyspoc._tag import Tag, Type, Signature
from pyspoc.settings import settings


ReducerResultNDArray: TypeAlias = (
    np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
    np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
    np.floating | np.integer |
    float | int)


TFilter = TypeVar("TFilter", bound=Tag | Type | Signature | str)

@dataclass(frozen=True)
class AnyOf(Generic[TFilter]):
    descriptors: tuple[TFilter, ...]

    @overload
    def __init__(self, *args: TFilter): ...

    @overload
    def __init__(self, args: Iterable[TFilter], /): ...

    
    def __init__(self, *args: TFilter | Iterable[TFilter]):
        values = tuple(args)

        if len(values) == 1 and isinstance(values[0], Iterable):
            values = tuple(values[0])
        else:
            # Reject mixed forms such as example(1, (2, 3)).
            if any(isinstance(value, tuple) for value in values):
                raise TypeError("Pass individual descriptors or one tuple of descriptors.")

        object.__setattr__(self, "descriptors", values)


FilterCondition: TypeAlias = TFilter | AnyOf[TFilter]


class Reducer(Component, ABC):

    _cached_results: dict[Statistic, dict[Reducer, np.ndarray]] = dict()

    @classmethod
    def _get_property_schema(cls) -> tuple[
        tuple[str, type, MethodType | FunctionType], ...]:
        
        return (
            ("long_name", str, cls.get_long_name),
            ("identifier", str, cls.get_identifier),
            ("type", Type, cls.get_type),
            ("tags", Tag, cls.get_tags),
            ("signatures", Signature, cls.get_signatures),
            ("statistics_filter", str, cls.get_statistics_filter),
            ("statistic_types_filter", Type, cls.get_statistic_types_filter),
            ("statistic_tags_filter", Tag, cls.get_statistic_tags_filter),
            ("statistic_signatures_filter", Signature, cls.get_statistic_signatures_filter)
            )


    @classmethod
    def get_statistics_filter(cls) -> tuple[str, ...]:
        """Return the Statistic names that the Reducer is restricted to. Higher level or abstract
        Statistics can be provided to select an entire family of Statistics.
        
        To provide filters, a new Reducer class must implement either:
        - _statistics_filter: A `tuple[str, ...]` class-level private field.
        - get_statistics_filter(): A `tuple[str, ...]` class method
            overriding this method.
        """
        try:
            return getattr(cls, "_statistics_filter")
        except AttributeError:
            try:
                return getattr(cls, "_pyspoc_statistics_filter")
            except AttributeError:
                return tuple()


    @classmethod
    def get_statistic_types_filter(cls) -> tuple[FilterCondition[Type], ...]:
        """Return the Statistic Types that the Reducer is restricted to.
        
        To provide filters, a new Reducer class must implement either:
        - _statistic_types_filter: A `tuple[FilterCondition[Type], ...]` class-level private field.
        - get_statistic_types_filter(): A `tuple[FilterCondition[Type], ...]` class method
            overriding this method.
        """
        try:
            return getattr(cls, "_statistic_types_filter")
        except AttributeError:
            try:
                return getattr(cls, "_pyspoc_statistic_types_filter")
            except AttributeError:
                return tuple()


    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[FilterCondition[Tag], ...]:
        """Return the Statistic Tags that the Reducer is restricted to.
        
        To provide filters, a new Reducer class must implement either:
        - _statistic_tags_filter: A `tuple[FilterCondition[Tag], ...]` class-level private field.
        - get_statistic_tags_filter(): A `tuple[FilterCondition[Tag], ...]` class method
            overriding this method.
        """
        try:
            return getattr(cls, "_statistic_tags_filter")
        except AttributeError:
            try:
                return getattr(cls, "_pyspoc_statistic_tags_filter")
            except AttributeError:
                return tuple()

    @classmethod
    def get_statistic_signatures_filter(cls) -> tuple[FilterCondition[Signature], ...]:
        """Return the Statistic Signatures that the Reducer is restricted to.
        
        To provide filters, a new Reducer class must implement either:
        - _statistic_tags_filter: A `tuple[FilterCondition[Signature], ...]` class-level private
            field.
        - get_statistic_tags_filter(): A `tuple[FilterCondition[Signature], ...]` class method
            overriding this method.
        """
        try:
            return getattr(cls, "_statistic_signatures_filter")
        except AttributeError:
            try:
                return getattr(cls, "_pyspoc_statistic_signatures_filter")
            except AttributeError:
                return tuple()
    
    
    def evaluate_filters(
            self,
            statistic: Statistic
        ) -> bool:
        stats = self.get_statistics_filter()

        if stats:
            if statistic.get_name() not in stats:
                return False

        tag_filters = (
            (Tag, self.get_statistic_tags_filter),
            (Type, self.get_statistic_types_filter),
            (Signature, self.get_statistic_signatures_filter)
        )

        for filter_type, filter_func in tag_filters:
            matches = self._matches_tags(
                filter_type=filter_type,
                available_tags=statistic.get_descriptors(filter_type),
                conditions=filter_func()
            )
            if not matches:
                return False

        return True
            

    def _matches_tags(
            self,
            filter_type: type[TFilter],
            available_tags: set[TFilter],
            conditions: tuple[FilterCondition[TFilter], ...],
    ) -> bool:
        return all(
            condition in available_tags
            if isinstance(condition, filter_type)
            else any(tag in available_tags for tag in condition.descriptors)
            for condition in conditions
        )

    @classmethod
    def _get_properties_yaml_path(cls) -> Path:
        return settings.current.reducer_properties_yaml_path
    

    def calculate(
            self,
            statistic: Statistic) -> ReducerResultNDArray:

        # temporarily uncache results, TODO: fix caching mechanisms
        self.clear_cache(statistic)

        statistic_result = self._get_statistic_result(statistic)
        
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = self.reduce(statistic_result)

            if captured and self._active_calculator is not None:
                self._active_calculator._add_warnings(captured)

        result = np.array(result)
        self._cache_result(statistic=statistic, result=result)
        return result


    def _get_statistic_result(
            self,
            statistic: Statistic) -> StatisticResultNDArray | None:
        
        statistic_results = self._cached_results.get(statistic)

        if statistic_results is not None:
            result = statistic_results.get(self)

            if result is not None:
                return result

        statistic_result = statistic._get_locally_cached_result()
        
        if isinstance(statistic_result, np.ndarray):
            statistic_result.flags.writeable = False

        return statistic_result


    def _cache_result(
            self,
            statistic: Statistic,
            result: ReducerResultNDArray):

        statistic_results = self._cached_results.get(statistic)

        if statistic_results is None:
            self._cached_results[statistic] = {
                self: result
            }
        else:
            statistic_results[self] = result

        return result
    

    @abstractmethod
    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        """
        Compute the reduction method on the statistical data provided.

        Parameters
        ----------

        data: numpy.ndarray
            A numpy array generated by a Statistic.

        Returns
        ----------
        numpy.array or float
            Reduction of the statistic as either a vector or scalar value.

        """
        pass

    @classmethod
    def clear_cache(cls, statistic: Statistic, include_gc: bool = False):
        cached_statistic_results = cls._cached_results.get(statistic)

        if cached_statistic_results:
            cls._cached_results[statistic] = dict()

        if include_gc:
            gc.collect()


    @staticmethod
    def _get_component_type() -> type:
        return Reducer

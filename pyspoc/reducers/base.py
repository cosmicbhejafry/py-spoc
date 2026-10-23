from __future__ import annotations

import warnings
import numpy as np
import copy
import gc

from abc import ABC, abstractmethod
from types import MethodType, FunctionType
from pathlib import Path

from pyspoc.statistics.base import Statistic
from pyspoc._base import Component
from pyspoc._tag import Tag, Type, Signature
from pyspoc.settings import settings
from pyspoc._exceptions import MissingComponentPropertyError



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
    def get_statistics_filter(cls) -> tuple[Statistic, ...]:
        """Return the Statistic names that the Reducer is restricted to. Higher level or abstract
        Statistics can be provided to select an entire family of Statistics.
        
        A new Reducer class must implement either:
        - _statistics_filter: A `tuple[str, ...]` class-level private field.
        - get_statistics_filter(): A `tuple[str, ...]` class method overriding this method.
        """
        try:
            return getattr(cls, "_statistics_filter")
        except AttributeError as e:
            raise MissingComponentPropertyError(
                cls, "statistics_filter", tuple[str, ...]) from e


    @classmethod
    def get_statistic_types_filter(cls) -> tuple[Type, ...]:
        """Return the Statistic Types that the Reducer is restricted to.
        
        A new Reducer class must implement either:
        - _statistic_types_filter: A `tuple[Type, ...]` class-level private field.
        - get_statistic_types_filter(): A `tuple[Type, ...]` class method overriding this method.
        """
        try:
            return getattr(cls, "_statistic_types_filter")
        except AttributeError as e:
            raise MissingComponentPropertyError(
                cls, "statistic_types_filter", tuple[Type, ...]) from e


    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Return the Statistic Tags that the Reducer is restricted to.
        
        A new Reducer class must implement either:
        - _statistic_tags_filter: A `tuple[Tag, ...]` class-level private field.
        - get_statistic_tags_filter(): A `tuple[Tag, ...]` class method overriding this method.
        """
        try:
            return getattr(cls, "_statistic_tags_filter")
        except AttributeError as e:
            raise MissingComponentPropertyError(
                cls, "statistic_tags_filter", tuple[Tag, ...]) from e

    @classmethod
    def get_statistic_signatures_filter(cls) -> tuple[Signature, ...]:
        """Return the Statistic Signatures that the Reducer is restricted to.
        
        A new Reducer class must implement either:
        - _statistic_tags_filter: A `tuple[Signature, ...]` class-level private field.
        - get_statistic_tags_filter(): A `tuple[Signature, ...]` class method overriding \
            this method.
        """
        try:
            return getattr(cls, "_statistic_signatures_filter")
        except AttributeError as e:
            raise MissingComponentPropertyError(
                cls, "statistic_signatures_filter", tuple[Signature, ...]) from e
    

    @classmethod
    def _get_properties_yaml_path(cls) -> Path:
        return settings.current.reducer_properties_yaml_path
    

    def calculate(self,
                  statistic: Statistic) -> np.ndarray:

        # temporarily uncache results, TODO: fix caching mechanisms
        self.clear_cache(statistic)

        statistic_results = self._cached_results.get(statistic)

        if statistic_results is not None:
            result = statistic_results.get(self)

            if result is not None:
                return result

        statistic_result = statistic._get_result()
        statistic_result_cp = copy.deepcopy(statistic_result)
        
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = self.reduce(statistic_result_cp)

            if captured and self._active_calculator is not None:
                self._active_calculator._add_warnings(captured)

        result = np.array(result)

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

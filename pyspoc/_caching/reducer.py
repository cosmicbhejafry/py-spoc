"""Memory-bounded caching for results produced by Reducers."""

from __future__ import annotations

import sys

from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary, WeakSet

import numpy as np

from pyspoc._argchecking import check_float
from pyspoc._execution.memory import get_available_memory_bytes


class ReducerResultCache:
    """Coordinate Reducer result stores under one byte-bounded global LRU."""

    def __init__(
        self,
        available_memory_getter: Callable[[], int] = get_available_memory_bytes,
    ) -> None:
        self._lock = RLock()
        self._lru: OrderedDict[tuple[Any, Any, Any], int] = OrderedDict()
        self._registry: WeakKeyDictionary[Any, WeakSet[Any]] = WeakKeyDictionary()
        self._size = 0
        self._available_memory_getter = available_memory_getter
        self._available_memory: int | None = None

    @property
    def size(self) -> int:
        """Return the bytes currently attributed to cached Reducer results."""
        with self._lock:
            return self._size

    @property
    def available_memory(self) -> int | None:
        """Return the memory snapshot used during the active cache lifetime."""
        with self._lock:
            return self._available_memory

    def get(self, reducer: Any, statistic: Any, dataset: Any) -> Any | None:
        """Return a cached result and mark its entry as recently used."""
        key = (reducer, statistic, dataset)

        with self._lock:
            result = self._get_results(reducer).get((statistic, dataset))

            if result is None:
                return None

            if key not in self._lru:
                size = self._get_result_size(result)
                self._lru[key] = size
                self._size += size

            self._lru.move_to_end(key)
            return result

    def store(
        self,
        reducer: Any,
        statistic: Any,
        dataset: Any,
        result: Any,
        memory_fraction: float,
    ) -> bool:
        """Store a result, evicting least-recently-used entries as needed."""
        key = (reducer, statistic, dataset)
        result_size = self._get_result_size(result)
        checked_fraction = check_float(
            memory_fraction,
            exclusive_minimum=0.0,
            maximum=1.0,
            arg_name="memory_fraction",
        )

        with self._lock:
            maximum_size = self._get_maximum_size(checked_fraction)
            self._remove_key(key)

            if result_size > maximum_size:
                if not self._lru:
                    self._available_memory = None
                return False

            self._get_results(reducer)[(statistic, dataset)] = result
            self._lru[key] = result_size
            self._size += result_size
            self._register(reducer, dataset)

            while self._size > maximum_size:
                self._remove_key(next(iter(self._lru)))

            return key in self._lru

    def clear_dataset(self, dataset: Any) -> None:
        """Remove all Reducer results derived from one dataset."""
        with self._lock:
            reducers = tuple(self._registry.pop(dataset, ()))

            for reducer in reducers:
                keys = (
                    key for key in tuple(self._lru)
                    if key[0] is reducer and key[2] is dataset
                )

                for key in keys:
                    self._remove_key(key)

            if not self._lru:
                self._available_memory = None

    def clear(self) -> None:
        """Remove every cached Reducer result and reset memory accounting."""
        with self._lock:
            for reducer, statistic, dataset in tuple(self._lru):
                self._get_results(reducer).pop((statistic, dataset), None)

            self._lru.clear()
            self._registry = WeakKeyDictionary()
            self._size = 0
            self._available_memory = None

    def _get_maximum_size(self, memory_fraction: float) -> int:
        if self._available_memory is None:
            self._available_memory = self._available_memory_getter()

        return int(self._available_memory * memory_fraction)

    def _register(self, reducer: Any, dataset: Any) -> None:
        reducers = self._registry.get(dataset)

        if reducers is None:
            reducers = WeakSet()
            self._registry[dataset] = reducers

        reducers.add(reducer)

    def _remove_key(self, key: tuple[Any, Any, Any]) -> None:
        reducer, statistic, dataset = key
        size = self._lru.pop(key, None)
        self._get_results(reducer).pop((statistic, dataset), None)

        if size is None:
            return

        self._size -= size
        reducers = self._registry.get(dataset)

        if reducers is not None:
            # A Reducer can have several Statistics cached for one Dataset.
            if not any(key[0] is reducer and key[2] is dataset for key in self._lru):
                reducers.discard(reducer)

            if not reducers:
                self._registry.pop(dataset, None)

    @staticmethod
    def _get_results(reducer: Any) -> dict[tuple[Any, Any], Any]:
        results = getattr(reducer, "_cached_results", None)

        if not isinstance(results, dict):
            results = {}
            reducer._cached_results = results

        return results

    @staticmethod
    def _get_result_size(result: Any) -> int:
        if isinstance(result, np.ndarray):
            return result.nbytes

        return sys.getsizeof(result)


reducer_result_cache = ReducerResultCache()


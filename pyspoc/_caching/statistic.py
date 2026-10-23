"""Memory-bounded caching for results produced by Statistics."""

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


class StatisticResultCache:
    """Coordinate per-Statistic result stores under one byte-bounded LRU.

    Statistic objects retain the actual results in their local
    ``_cached_results`` mappings. This manager owns global byte accounting,
    access ordering, dataset lookup, eviction, and synchronization.
    """

    def __init__(
        self,
        available_memory_getter: Callable[[], int] = get_available_memory_bytes,
    ) -> None:
        self._lock = RLock()
        self._lru: OrderedDict[tuple[Any, Any], int] = OrderedDict()
        self._registry: WeakKeyDictionary[Any, WeakSet[Any]] = WeakKeyDictionary()
        self._size = 0
        self._available_memory_getter = available_memory_getter
        self._available_memory: int | None = None

    @property
    def size(self) -> int:
        """Return the number of bytes currently attributed to cached results."""
        with self._lock:
            return self._size

    @property
    def available_memory(self) -> int | None:
        """Return the available-memory snapshot used by the active cache."""
        with self._lock:
            return self._available_memory

    def get(self, statistic: Any, dataset: Any) -> Any | None:
        """Return a cached result and mark the entry as recently used."""
        key = (statistic, dataset)

        with self._lock:
            result = statistic._cached_results.get(dataset)

            if result is None:
                return None

            # The local mapping is authoritative. Repair missing LRU metadata
            # defensively rather than returning a result that cannot be evicted.
            if key not in self._lru:
                size = self._get_result_size(result)
                self._lru[key] = size
                self._size += size

            self._lru.move_to_end(key)
            return result

    def store(
        self,
        statistic: Any,
        dataset: Any,
        result: Any,
        memory_fraction: float,
    ) -> bool:
        """Store a result, evicting least-recently-used entries as required."""
        key = (statistic, dataset)
        result_size = self._get_result_size(result)
        checked_fraction = check_float(
            memory_fraction,
            exclusive_minimum=0.0,
            maximum=1.0,
            arg_name="memory_fraction",
        )

        with self._lock:
            maximum_size = self._get_maximum_size(checked_fraction)

            # Replacement must remove the old entry from both stores and from
            # byte accounting before the new result is considered.
            self._remove_key(key)

            # A result larger than the complete budget can never coexist with
            # a valid bounded cache. Return it to the caller without retaining it.
            if result_size > maximum_size:
                if not self._lru:
                    self._available_memory = None
                return False

            statistic._cached_results[dataset] = result
            self._lru[key] = result_size
            self._size += result_size
            self._register(statistic, dataset)

            while self._size > maximum_size:
                oldest_key = next(iter(self._lru))
                self._remove_key(oldest_key)

            return key in self._lru

    def clear_dataset(self, dataset: Any) -> tuple[Any, ...]:
        """Remove every Statistic result associated with one dataset."""
        with self._lock:
            statistics = tuple(self._registry.pop(dataset, ()))

            for statistic in statistics:
                self._remove_key((statistic, dataset))

            if not self._lru:
                self._available_memory = None

            return statistics

    def clear(self) -> None:
        """Remove all cached Statistic results and reset accounting."""
        with self._lock:
            for statistic, dataset in tuple(self._lru):
                statistic._cached_results.pop(dataset, None)

            self._lru.clear()
            self._registry = WeakKeyDictionary()
            self._size = 0
            # A completely cleared cache begins a new lifetime. Defer taking
            # its memory snapshot until another result is actually stored.
            self._available_memory = None

    def _get_maximum_size(self, memory_fraction: float) -> int:
        if self._available_memory is None:
            self._available_memory = self._available_memory_getter()

        return int(self._available_memory * memory_fraction)

    def _register(self, statistic: Any, dataset: Any) -> None:
        statistics = self._registry.get(dataset)

        if statistics is None:
            statistics = WeakSet()
            self._registry[dataset] = statistics

        statistics.add(statistic)

    def _remove_key(self, key: tuple[Any, Any]) -> None:
        statistic, dataset = key
        size = self._lru.pop(key, None)

        if size is None:
            # A local result may predate or otherwise lack manager metadata.
            statistic._cached_results.pop(dataset, None)
            return

        statistic._cached_results.pop(dataset, None)
        self._size -= size

        statistics = self._registry.get(dataset)

        if statistics is not None:
            statistics.discard(statistic)

            if not statistics:
                self._registry.pop(dataset, None)

    @staticmethod
    def _get_result_size(result: Any) -> int:
        if isinstance(result, np.ndarray):
            return result.nbytes

        return sys.getsizeof(result)


statistic_result_cache = StatisticResultCache()

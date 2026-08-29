"""Tests for byte-bounded LRU caching of Reducer results."""

from __future__ import annotations

import numpy as np
import pytest

from pyspoc._caching.reducer import ReducerResultCache, reducer_result_cache
from pyspoc._caching.statistic import statistic_result_cache
from pyspoc.core.reducer.vectoscalar import VectorToScalarReducer
from pyspoc.core.statistic import VectorStatistic
from pyspoc.dataset import Dataset
from pyspoc.settings import settings


class _CacheOwner:
    """Stand in for a Reducer when testing the cache manager directly."""

    pass


class _Dataset:
    pass


class _Statistic:
    pass


class _VectorStatistic(VectorStatistic):
    """Produce a cheap vector and count upstream computations."""

    def __init__(self) -> None:
        self.calls = 0
        super().__init__("test-vector", [])

    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        self.calls += 1
        return data.mean(axis=0)


class _SumReducer(VectorToScalarReducer):
    """Produce a cheap scalar and count actual reduction calls."""

    def __init__(self) -> None:
        self.calls = 0
        super().__init__("sum", [])

    def _reduce(
        self,
        data: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> float:
        self.calls += 1
        return float(data.sum())


@pytest.fixture(autouse=True)
def clear_global_result_caches():
    """Isolate tests from both process-global pipeline caches."""
    reducer_result_cache.clear()
    statistic_result_cache.clear()
    yield
    reducer_result_cache.clear()
    statistic_result_cache.clear()


def test_cache_key_includes_statistic_and_dataset() -> None:
    """Reducer entries must distinguish both their Statistic and Dataset."""
    cache = ReducerResultCache(lambda: 1_000)
    reducer = _CacheOwner()
    first_statistic = _Statistic()
    second_statistic = _Statistic()
    first_dataset = _Dataset()
    second_dataset = _Dataset()

    cache.store(reducer, first_statistic, first_dataset, 1.0, 1.0)
    cache.store(reducer, second_statistic, first_dataset, 2.0, 1.0)
    cache.store(reducer, first_statistic, second_dataset, 3.0, 1.0)

    assert cache.get(reducer, first_statistic, first_dataset) == 1.0
    assert cache.get(reducer, second_statistic, first_dataset) == 2.0
    assert cache.get(reducer, first_statistic, second_dataset) == 3.0


def test_cache_hit_refreshes_reducer_lru_order() -> None:
    """A Reducer cache hit should protect that entry from eviction."""
    result = np.ones(2, dtype=np.float64)
    cache = ReducerResultCache(lambda: result.nbytes * 2)
    reducer = _CacheOwner()
    statistic = _Statistic()
    datasets = [_Dataset() for _ in range(3)]

    cache.store(reducer, statistic, datasets[0], result, 1.0)
    cache.store(reducer, statistic, datasets[1], result, 1.0)
    # Refresh the first entry, making the second entry least recently used.
    assert cache.get(reducer, statistic, datasets[0]) is result
    cache.store(reducer, statistic, datasets[2], result, 1.0)

    assert cache.get(reducer, statistic, datasets[0]) is result
    assert cache.get(reducer, statistic, datasets[1]) is None


def test_clear_dataset_preserves_other_dataset_entries() -> None:
    """Dataset invalidation should not disturb reductions of other datasets."""
    cache = ReducerResultCache(lambda: 1_000)
    reducer = _CacheOwner()
    statistic = _Statistic()
    removed = _Dataset()
    retained = _Dataset()

    cache.store(reducer, statistic, removed, 1.0, 1.0)
    cache.store(reducer, statistic, retained, 2.0, 1.0)
    cache.clear_dataset(removed)

    assert cache.get(reducer, statistic, removed) is None
    assert cache.get(reducer, statistic, retained) == 2.0


def test_replacing_reducer_result_updates_size() -> None:
    """Replacing a Reducer entry should account only for the new result."""
    cache = ReducerResultCache(lambda: 100)
    reducer = _CacheOwner()
    statistic = _Statistic()
    dataset = _Dataset()
    first = np.ones(2, dtype=np.float64)
    replacement = np.ones(5, dtype=np.float64)

    cache.store(reducer, statistic, dataset, first, 1.0)
    cache.store(reducer, statistic, dataset, replacement, 1.0)

    assert cache.size == replacement.nbytes
    assert cache.get(reducer, statistic, dataset) is replacement


def test_oversized_reducer_result_is_not_cached() -> None:
    """A Reducer result exceeding its complete budget should not be retained."""
    result = np.ones(8, dtype=np.float64)
    cache = ReducerResultCache(lambda: result.nbytes - 1)
    reducer = _CacheOwner()
    statistic = _Statistic()
    dataset = _Dataset()

    stored = cache.store(reducer, statistic, dataset, result, 1.0)

    assert stored is False
    assert cache.size == 0
    assert cache.get(reducer, statistic, dataset) is None


def test_reducer_memory_snapshot_refreshes_after_complete_clear() -> None:
    """A complete Reducer-cache clear should trigger a fresh memory probe."""
    reported_memory = iter((100, 200))
    cache = ReducerResultCache(lambda: next(reported_memory))
    reducer = _CacheOwner()
    statistic = _Statistic()
    dataset = _Dataset()

    cache.store(reducer, statistic, dataset, 1.0, 0.5)
    assert cache.available_memory == 100

    cache.clear()
    cache.store(reducer, statistic, dataset, 1.0, 0.5)

    assert cache.available_memory == 200


def test_reducer_calculation_uses_cached_result() -> None:
    """Repeated pipeline execution should reuse both intermediate results."""
    dataset = Dataset(np.arange(12, dtype=np.float64).reshape(6, 2), normalise=False)
    statistic = _VectorStatistic()
    reducer = _SumReducer()

    with settings.override(statistic_caching=True, reducer_caching=True):
        first = reducer.calculate_reduction(dataset, statistic)
        second = reducer.calculate_reduction(dataset, statistic)

    assert first == second
    assert reducer.calls == 1
    assert statistic.calls == 1


def test_disabled_reducer_cache_recomputes_reduction_only() -> None:
    """A disabled Reducer cache should not disable upstream Statistic reuse."""
    dataset = Dataset(np.arange(12, dtype=np.float64).reshape(6, 2), normalise=False)
    statistic = _VectorStatistic()
    reducer = _SumReducer()

    with settings.override(statistic_caching=True, reducer_caching=False):
        reducer.calculate_reduction(dataset, statistic)
        reducer.calculate_reduction(dataset, statistic)

    assert statistic.calls == 1
    assert reducer.calls == 2


def test_reducer_cache_distinguishes_reducers() -> None:
    """Different Reducer instances must not share one derived result entry."""
    dataset = Dataset(np.arange(12, dtype=np.float64).reshape(6, 2), normalise=False)
    statistic = _VectorStatistic()
    first_reducer = _SumReducer()
    second_reducer = _SumReducer()

    with settings.override(statistic_caching=True, reducer_caching=True):
        first_reducer.calculate_reduction(dataset, statistic)
        second_reducer.calculate_reduction(dataset, statistic)

    assert first_reducer.calls == 1
    assert second_reducer.calls == 1


def test_dataset_invalidation_clears_entire_pipeline() -> None:
    """Dataset invalidation should force both pipeline stages to recompute."""
    dataset = Dataset(np.arange(12, dtype=np.float64).reshape(6, 2), normalise=False)
    statistic = _VectorStatistic()
    reducer = _SumReducer()

    with settings.override(statistic_caching=True, reducer_caching=True):
        reducer.calculate_reduction(dataset, statistic)
        # One invalidation call must remove the Statistic result and every
        # Reducer result derived from it for this Dataset.
        dataset.uncache()
        reducer.calculate_reduction(dataset, statistic)

    assert statistic.calls == 2
    assert reducer.calls == 2

"""Tests for byte-bounded LRU caching of Statistic results."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import numpy as np
import pytest
import sys

from pyspoc._caching.statistic import StatisticResultCache
from pyspoc._caching.statistic import statistic_result_cache
from pyspoc.core.statistic import ScalarStatistic, VectorStatistic
from pyspoc.dataset import Dataset
from pyspoc.settings import settings


class _CacheOwner:
    """Provide the minimum per-object storage expected by the cache manager."""

    def __init__(self) -> None:
        self._cached_results: dict[object, object] = {}


class _Dataset:
    pass


class _CountingScalarStatistic(ScalarStatistic):
    """Return a cheap scalar while recording actual computations."""

    def __init__(self) -> None:
        self.calls = 0
        super().__init__("counting-scalar", [])

    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> float:
        self.calls += 1
        return float(data.size)


class _CountingVectorStatistic(VectorStatistic):
    """Expose the produced array so result ownership policies can be tested."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_created: np.ndarray | None = None
        super().__init__("counting-vector", [])

    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]] | None:
        
        self.calls += 1
        self.last_created = data.mean(axis=0)
        return self.last_created


class _RacingVectorStatistic(VectorStatistic):
    """Force concurrent callers to compute after observing the same cache miss."""

    def __init__(self, participants: int) -> None:
        self.calls = 0
        self._counter_lock = Lock()
        self._barrier = Barrier(participants)
        super().__init__("racing-vector", [])

    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        with self._counter_lock:
            # ``calls += 1`` is a read-modify-write operation. Protect it so
            # the assertion below measures every computation reliably.
            self.calls += 1

        # Ensure every caller has observed the same cache miss before any one
        # of them can finish computing and publish a result.
        self._barrier.wait(timeout=5)
        return data.mean(axis=0)


@pytest.fixture(autouse=True)
def clear_global_result_caches():
    """Prevent the process-global cache from leaking state between tests."""
    statistic_result_cache.clear()
    yield
    statistic_result_cache.clear()


def test_cache_hit_refreshes_lru_order() -> None:
    """Accessing the oldest entry should protect it from the next eviction."""
    result = np.ones(2, dtype=np.float64)
    cache = StatisticResultCache(lambda: result.nbytes * 2)
    statistic = _CacheOwner()
    first = _Dataset()
    second = _Dataset()

    cache.store(statistic, first, result, memory_fraction=1.0)
    cache.store(statistic, second, result, memory_fraction=1.0)

    # Reading ``first`` moves it behind ``second`` in recency order.
    assert cache.get(statistic, first) is result

    # Only two arrays fit. Adding a third must therefore evict ``second``,
    # which became the least-recently-used entry after the read above.
    third = _Dataset()
    cache.store(statistic, third, result, memory_fraction=1.0)

    assert cache.get(statistic, first) is result
    assert cache.get(statistic, second) is None
    assert cache.get(statistic, third) is result


def test_replacing_result_updates_size_without_double_counting() -> None:
    """Replacing one key should subtract its old bytes before adding new bytes."""
    cache = StatisticResultCache(lambda: 100)
    statistic = _CacheOwner()
    dataset = _Dataset()
    first = np.ones(2, dtype=np.float64)
    replacement = np.ones(5, dtype=np.float64)

    cache.store(statistic, dataset, first, memory_fraction=1.0)
    cache.store(statistic, dataset, replacement, memory_fraction=1.0)

    assert cache.size == replacement.nbytes
    assert cache.get(statistic, dataset) is replacement


def test_result_larger_than_budget_is_not_cached() -> None:
    """A result that cannot fit alone should be returned but not retained."""
    result = np.ones(8, dtype=np.float64)
    cache = StatisticResultCache(lambda: result.nbytes - 1)
    statistic = _CacheOwner()
    dataset = _Dataset()

    stored = cache.store(
        statistic,
        dataset,
        result,
        memory_fraction=1.0,
    )

    assert stored is False
    assert cache.size == 0
    assert cache.get(statistic, dataset) is None


def test_clear_dataset_removes_results_and_updates_size() -> None:
    """Dataset clearing should remove all related Statistics and only their bytes."""
    cache = StatisticResultCache(lambda: 100)
    first_statistic = _CacheOwner()
    second_statistic = _CacheOwner()
    retained_statistic = _CacheOwner()
    removed_dataset = _Dataset()
    retained_dataset = _Dataset()
    result = np.ones(3, dtype=np.float32)

    cache.store(first_statistic, removed_dataset, result, memory_fraction=1.0)
    cache.store(second_statistic, removed_dataset, result, memory_fraction=1.0)
    cache.store(retained_statistic, retained_dataset, result, memory_fraction=1.0)

    affected = cache.clear_dataset(removed_dataset)

    assert set(affected) == {first_statistic, second_statistic}
    assert cache.get(first_statistic, removed_dataset) is None
    assert cache.get(second_statistic, removed_dataset) is None
    assert cache.get(retained_statistic, retained_dataset) is result
    assert cache.size == result.nbytes


def test_clear_removes_all_results() -> None:
    """A complete clear should empty local storage and global accounting."""
    cache = StatisticResultCache(lambda: 100)
    statistic = _CacheOwner()
    dataset = _Dataset()
    result = np.ones(3, dtype=np.float32)
    cache.store(statistic, dataset, result, memory_fraction=1.0)

    cache.clear()

    assert cache.size == 0
    assert statistic._cached_results == {}


def test_available_memory_is_snapshotted_until_cache_is_empty() -> None:
    """Available memory should remain stable for one non-empty cache lifetime."""
    reported_memory = iter((100, 200))
    cache = StatisticResultCache(lambda: next(reported_memory))
    statistic = _CacheOwner()
    first_dataset = _Dataset()
    second_dataset = _Dataset()
    result = np.ones(1, dtype=np.float64)

    cache.store(statistic, first_dataset, result, memory_fraction=0.5)
    cache.store(statistic, second_dataset, result, memory_fraction=0.5)

    assert cache.available_memory == 100

    # A complete clear ends that lifetime, so the next store probes memory again.
    cache.clear()
    cache.store(statistic, first_dataset, result, memory_fraction=0.5)

    assert cache.available_memory == 200


def test_clearing_final_dataset_refreshes_memory_on_next_store() -> None:
    """Removing the final dataset should also end the current cache lifetime."""
    reported_memory = iter((100, 200))
    cache = StatisticResultCache(lambda: next(reported_memory))
    statistic = _CacheOwner()
    dataset = _Dataset()
    result = np.ones(1, dtype=np.float64)

    cache.store(statistic, dataset, result, memory_fraction=0.5)
    cache.clear_dataset(dataset)

    assert cache.available_memory is None

    cache.store(statistic, dataset, result, memory_fraction=0.5)

    assert cache.available_memory == 200


def test_calculate_summary_caches_and_preserves_python_scalar() -> None:
    """The public Statistic pipeline should compute once and preserve scalars."""
    dataset = Dataset(np.ones((4, 2), dtype=np.float64), normalise=False)
    statistic = _CountingScalarStatistic()

    with settings.override(statistic_caching=True):
        first = statistic.calculate_summary(dataset)
        second = statistic.calculate_summary(dataset)

    assert first == 8.0
    assert type(first) is float
    assert second is first
    assert statistic.calls == 1


def test_disabled_statistic_cache_recomputes() -> None:
    """Disabling Statistic caching should make every request recompute."""
    dataset = Dataset(np.ones((4, 2), dtype=np.float64), normalise=False)
    statistic = _CountingScalarStatistic()

    with settings.override(statistic_caching=False):
        statistic.calculate_summary(dataset)
        statistic.calculate_summary(dataset)

    assert statistic.calls == 2


def test_statistic_cache_distinguishes_datasets() -> None:
    """One Statistic instance must maintain independent entries per Dataset."""
    first_dataset = Dataset(np.ones((4, 2), dtype=np.float64), normalise=False)
    second_dataset = Dataset(np.full((4, 2), 2.0), normalise=False)
    statistic = _CountingVectorStatistic()

    with settings.override(statistic_caching=True):
        first = statistic.calculate_summary(first_dataset)
        second = statistic.calculate_summary(second_dataset)

    assert statistic.calls == 2
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("policy", ["freeze", "copy"])
def test_statistic_result_policy_returns_readonly_array(policy: str) -> None:
    """Both policies freeze results, but only ``copy`` changes array identity."""
    dataset = Dataset(np.ones((4, 2), dtype=np.float64), normalise=False)
    statistic = _CountingVectorStatistic()

    with settings.override(result_array_policy=policy):
        result = statistic.calculate_summary(dataset)

    if isinstance(result, np.ndarray):
        assert result.flags.writeable is False

    assert statistic.last_created is not None
    assert (result is statistic.last_created) is (policy == "freeze")


def test_concurrent_cache_misses_may_duplicate_computation_safely() -> None:
    """Concurrent misses may recompute, but must not corrupt cached state.

    The cache lock makes each lookup and store atomic, but it is deliberately
    not held during numerical computation. The barrier inside the dummy
    Statistic ensures all workers reach computation before any can publish a
    cached result, deterministically exercising that gap between operations.
    """
    workers = 4
    dataset = Dataset(np.arange(12, dtype=np.float64).reshape(6, 2), normalise=False)
    statistic = _RacingVectorStatistic(workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # All tasks use the exact same cache key: this Statistic and Dataset.
        results = list(executor.map(lambda _: statistic.calculate_summary(dataset), range(workers)))

    # No single-flight coordination exists yet, so every synchronized miss is
    # expected to perform the computation. Each result must still be correct.
    assert statistic.calls == workers
    assert all(np.array_equal(result, results[0]) for result in results)

    # Locked replacement removes the preceding value's byte contribution.
    # Although four workers stored the key, exactly one result remains counted.
    size = results[0].nbytes if isinstance(results[0], np.ndarray) else sys.getsizeof(results[0])
    assert statistic_result_cache.size == size

    # Once publication has completed, a later caller should observe the cache
    # hit and leave the computation count unchanged.
    statistic.calculate_summary(dataset)
    assert statistic.calls == workers

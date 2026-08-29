"""Tests for dynamically scheduled pairwise execution."""

from __future__ import annotations

import threading
import time
from typing import Literal

import numpy as np
import pytest

from pyspoc._execution import execute_pairwise, execute_pairwise_barrier
from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic


class ExamplePairwiseStatistic(PairwiseStatistic):
    """Concrete Statistic used to exercise executor integration."""

    def __init__(self,
                 dim: Literal['n','p'] = "p",
                 is_ordered: bool = False,
                 symmetry_type: None | Literal['exact','negative','reciprocal'] = None):

        super().__init__(
            dim=dim,
            is_ordered=is_ordered,
            symmetry_type=symmetry_type,
            thread_safety="safe",
        )

    @property
    def identifier(self) -> str:
        return "example-pairwise"

    @property
    def labels(self) -> list[str]:
        return ["test"]

    def pairwise_compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(x, y))


def test_directed_pairwise_execution_computes_every_ordered_pair() -> None:
    data = np.arange(20, dtype=float).reshape(5, 4)

    result = execute_pairwise(
        data,
        lambda x, y: np.sum(x) - np.sum(y),
        symmetry_type=None,
        max_workers=2,
        chunks_per_worker=2,
    )

    column_sums = np.sum(data, axis=0)
    expected = column_sums[:, None] - column_sums[None, :]
    np.testing.assert_allclose(result, expected)


def test_pairwise_statistic_delegates_compute_to_concurrent_executor() -> None:
    data = np.arange(20, dtype=float).reshape(5, 4)
    statistic = ExamplePairwiseStatistic(symmetry_type="exact")

    with settings.override(max_worker_threads=2):
        result = statistic.compute(data)

    np.testing.assert_allclose(result, data.T @ data)


@pytest.mark.parametrize("max_workers", [1, 3])
def test_pairwise_execution_compares_columns_across_two_arrays(
    max_workers: int,
) -> None:
    data_a = np.arange(20, dtype=float).reshape(5, 4)
    data_b = np.arange(15, dtype=float).reshape(5, 3)

    result = execute_pairwise(
        data_a,
        np.dot,
        data_b,
        max_workers=max_workers,
    )

    assert result.shape == (4, 3)
    np.testing.assert_allclose(result, data_a.T @ data_b)


def test_pairwise_execution_rejects_symmetry_between_two_arrays() -> None:
    with pytest.raises(ValueError, match="symmetry"):
        execute_pairwise(
            np.ones((4, 2)),
            np.dot,
            np.ones((4, 3)),
            symmetry_type="exact",
        )


@pytest.mark.parametrize("max_workers", [1, 3])
def test_symmetric_pairwise_execution_fills_both_triangles(max_workers: int) -> None:
    data = np.arange(20, dtype=float).reshape(5, 4)

    result = execute_pairwise(
        data,
        np.dot,
        symmetry_type="exact",
        max_workers=max_workers,
    )

    np.testing.assert_allclose(result, data.T @ data)


def test_negative_symmetry_negates_opposite_triangle() -> None:
    data = np.arange(15, dtype=float).reshape(5, 3)
    result = execute_pairwise(
        data,
        lambda x, y: np.sum(x) - np.sum(y),
        symmetry_type="negative",
        max_workers=2,
    )

    np.testing.assert_allclose(result, -result.T)


def test_reciprocal_symmetry_inverts_opposite_triangle() -> None:
    data = np.arange(1, 16, dtype=float).reshape(5, 3)
    result = execute_pairwise(
        data,
        lambda x, y: np.sum(x) / np.sum(y),
        symmetry_type="reciprocal",
        max_workers=2,
    )

    np.testing.assert_allclose(result * result.T, np.ones_like(result))


def test_worker_chunks_receive_active_settings_context() -> None:
    data = np.ones((4, 3))

    with settings.override(random_seed=47):
        result = execute_pairwise(
            data,
            lambda _x, _y: settings.current.random_seed,
            symmetry_type=None,
            max_workers=2,
        )

    np.testing.assert_array_equal(result, np.full((3, 3), 47.0))


def test_configured_worker_limit_is_validated_when_used() -> None:
    with settings.override(max_worker_threads=0):
        with pytest.raises(ValueError, match="settings.max_worker_threads"):
            execute_pairwise(np.ones((3, 2)), np.dot)


def test_execution_uses_more_than_one_worker_thread() -> None:
    data = np.ones((10, 6))
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def record_worker(_x: np.ndarray, _y: np.ndarray) -> float:
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.01)
        return 1.0

    execute_pairwise(
        data,
        record_worker,
        symmetry_type=None,
        max_workers=3,
        chunks_per_worker=4,
    )

    assert len(thread_ids) > 1


def test_worker_exception_propagates_to_caller() -> None:
    def fail(_x: np.ndarray, _y: np.ndarray) -> float:
        raise RuntimeError("pair failed")

    with pytest.raises(RuntimeError, match="pair failed"):
        execute_pairwise(np.ones((3, 2)), fail, max_workers=2)


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("max_workers", 0, ValueError),
        ("max_workers", True, TypeError),
        ("chunks_per_worker", 0, ValueError),
    ],
)
def test_execution_validates_worker_arguments(
    argument: str,
    value: object,
    exception: type[Exception],
) -> None:
    arguments = {argument: value}

    with pytest.raises(exception):
        execute_pairwise(np.ones((3, 2)), np.dot, **arguments)


def test_barrier_executor_reuses_anchor_blocks_and_streamed_values() -> None:
    data = np.tile(np.arange(6.0), (4, 1))
    preparation_counts = np.zeros(6, dtype=int)

    def prepare(column: np.ndarray) -> float:
        index = int(column[0])
        preparation_counts[index] += 1
        return float(index)

    result = execute_pairwise_barrier(
        data,
        prepare,
        lambda first, second: first + second,
        max_workers=2,
    )

    expected = np.add.outer(np.arange(6.0), np.arange(6.0))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(preparation_counts, [1, 1, 2, 2, 3, 3])


def test_barrier_executor_caches_more_anchors_than_workers() -> None:
    data = np.tile(np.arange(6.0), (4, 1))
    preparation_counts = np.zeros(6, dtype=int)

    def prepare(column: np.ndarray) -> float:
        index = int(column[0])
        preparation_counts[index] += 1
        return float(index)

    result = execute_pairwise_barrier(
        data,
        prepare,
        lambda first, second: first + second,
        max_workers=1,
        anchor_count=4,
    )

    np.testing.assert_array_equal(
        result,
        np.add.outer(np.arange(6.0), np.arange(6.0)),
    )
    np.testing.assert_array_equal(preparation_counts, [1, 1, 1, 1, 2, 2])


@pytest.mark.parametrize("max_workers", [1, 2])
def test_barrier_executor_compares_prepared_columns_across_two_arrays(
    max_workers: int,
) -> None:
    data_a = np.tile(np.arange(3.0), (4, 1))
    data_b = np.tile(np.arange(10.0, 14.0), (4, 1))

    result = execute_pairwise_barrier(
        data_a,
        lambda column: float(column[0]),
        lambda first, second: first + second,
        data_b,
        max_workers=max_workers,
        anchor_count=2,
    )

    expected = np.add.outer(np.arange(3.0), np.arange(10.0, 14.0))
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, expected)


def test_barrier_executor_prepares_anchor_block_concurrently() -> None:
    data = np.tile(np.arange(3.0), (4, 1))
    preparation_barrier = threading.Barrier(3)
    preparation_threads: set[int] = set()
    thread_lock = threading.Lock()

    def prepare(column: np.ndarray) -> float:
        with thread_lock:
            preparation_threads.add(threading.get_ident())
        preparation_barrier.wait(timeout=2.0)
        return float(column[0])

    result = execute_pairwise_barrier(
        data,
        prepare,
        lambda first, second: first + second,
        max_workers=3,
        anchor_count=3,
    )

    np.testing.assert_array_equal(
        result,
        np.add.outer(np.arange(3.0), np.arange(3.0)),
    )
    assert len(preparation_threads) == 3


def test_barrier_executor_bounds_live_prepared_values() -> None:
    class PreparedValue:
        live_count = 0
        maximum_live_count = 0

        def __init__(self, value: float) -> None:
            self.value = value
            type(self).live_count += 1
            type(self).maximum_live_count = max(
                type(self).maximum_live_count,
                type(self).live_count,
            )

        def __del__(self) -> None:
            type(self).live_count -= 1

    data = np.tile(np.arange(7.0), (3, 1))

    result = execute_pairwise_barrier(
        data,
        lambda column: PreparedValue(float(column[0])),
        lambda first, second: first.value + second.value,
        max_workers=2,
    )

    np.testing.assert_array_equal(
        result,
        np.add.outer(np.arange(7.0), np.arange(7.0)),
    )
    assert PreparedValue.maximum_live_count <= 3

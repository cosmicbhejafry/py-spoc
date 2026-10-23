"""Tests for memory-aware worker limits."""

from __future__ import annotations

import numpy as np
import pytest

from pyspoc._execution.memory import (
    estimate_dense_array_bytes,
    get_available_memory_bytes,
    maximum_workers_permitted_by_memory,
)


def test_estimate_dense_array_bytes_uses_shape_and_dtype() -> None:
    assert estimate_dense_array_bytes(10, dtype=np.float64) == 800
    assert estimate_dense_array_bytes(10, dtype=np.float32) == 400


def test_memory_worker_limit_reserves_shared_values_and_output() -> None:
    workers = maximum_workers_permitted_by_memory(
        100,
        memory_fraction=0.5,
        shared_prepared_values=1,
        reserved_bytes=100,
        available_memory_bytes=1_000,
    )

    # 500-byte budget - 100-byte output = four prepared slots. One slot is
    # shared, leaving three independently owned worker values.
    assert workers == 3


def test_memory_worker_limit_retains_serial_fallback() -> None:
    assert maximum_workers_permitted_by_memory(
        1_000,
        available_memory_bytes=100,
    ) == 1


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("prepared_value_bytes", 0, ValueError),
        ("memory_fraction", 0.0, ValueError),
        ("memory_fraction", 1.1, ValueError),
        ("shared_prepared_values", -1, ValueError),
        ("reserved_bytes", -1, ValueError),
    ],
)
def test_memory_worker_limit_validates_arguments(
    argument: str,
    value: int | float,
    exception: type[Exception],
) -> None:
    arguments = {
        "prepared_value_bytes": 100,
        "available_memory_bytes": 1_000,
        argument: value,
    }

    with pytest.raises(exception):
        maximum_workers_permitted_by_memory(**arguments)


def test_available_memory_is_positive() -> None:
    assert get_available_memory_bytes() > 0

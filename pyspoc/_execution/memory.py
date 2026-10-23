"""Memory-aware limits for bounded concurrent execution."""

from __future__ import annotations

import ctypes
import os
import sys

from typing import Any

import numpy as np

from pyspoc._argchecking import (
    check_float, check_integer, check_natural_number)
from pyspoc.settings import settings


def get_available_memory_bytes() -> int:
    """Return the currently available physical memory in bytes.

    ``psutil`` is used when installed. Standard-library platform fallbacks
    keep memory-aware core execution independent of the profiling extra.
    """
    try:
        import psutil
    except ImportError:
        pass
    else:
        return int(psutil.virtual_memory().available)

    if sys.platform == "win32":
        return _get_available_memory_windows()

    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(available_pages * page_size)


def estimate_dense_array_bytes(
    side_length: int,
    *,
    dtype: Any = np.float64,
) -> int:
    """Estimate storage for one dense square NumPy array."""
    check_integer(side_length, minimum=0, arg_name="side_length")
    return side_length * side_length * np.dtype(dtype).itemsize


def maximum_workers_permitted_by_memory(
    prepared_value_bytes: int,
    *,
    memory_fraction: float | None = None,
    shared_prepared_values: int = 1,
    reserved_bytes: int = 0,
    available_memory_bytes: int | None = None,
) -> int:
    """Return the worker count fitting a prepared-value memory budget.

    At least one worker is returned because serial pairwise execution still
    requires an anchor and a comparison value. The estimate controls planned
    concurrency but cannot guarantee allocation because available memory may
    change between measurement and use.
    """
    check_natural_number(prepared_value_bytes, arg_name="prepared_value_bytes")

    if memory_fraction is not None:
        checked_fraction = check_float(
            memory_fraction,
            exclusive_minimum=0.0,
            maximum=1.0,
            arg_name="memory_fraction",
        )
    else:
        memory_fraction = settings.current.max_memory_fraction

        
    check_integer(
        shared_prepared_values,
        minimum=0,
        arg_name="shared_prepared_values",
    )
    check_integer(reserved_bytes, minimum=0, arg_name="reserved_bytes")

    available = (
        get_available_memory_bytes()
        if available_memory_bytes is None
        else check_natural_number(
            available_memory_bytes,
            arg_name="available_memory_bytes",
        )
    )
    budget = max(0, int(available * checked_fraction) - reserved_bytes)
    prepared_slots = budget // prepared_value_bytes
    return max(1, prepared_slots - shared_prepared_values)


def _get_available_memory_windows() -> int:
    """Query Windows' ``GlobalMemoryStatusEx`` without a required dependency."""

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not succeeded:
        raise OSError("Windows could not report available physical memory.")
    return int(status.ullAvailPhys)


__all__ = [
    "estimate_dense_array_bytes",
    "get_available_memory_bytes",
    "maximum_workers_permitted_by_memory",
]

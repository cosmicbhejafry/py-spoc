"""Private execution machinery shared by pySPoC computation layers."""

from pyspoc._execution.memory import (
    estimate_dense_array_bytes,
    get_available_memory_bytes,
    maximum_workers_permitted_by_memory,
)
from pyspoc._execution.pairwise import (
    execute_pairwise,
    execute_pairwise_barrier,
    resolve_worker_limit,
)

__all__ = [
    "estimate_dense_array_bytes",
    "execute_pairwise",
    "execute_pairwise_barrier",
    "get_available_memory_bytes",
    "maximum_workers_permitted_by_memory",
    "resolve_worker_limit",
]

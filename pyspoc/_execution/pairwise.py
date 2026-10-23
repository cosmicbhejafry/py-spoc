"""Concurrent execution of scalar operations over pairs of data columns."""

from __future__ import annotations

import math
import os
import numpy as np

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Literal, TypeAlias, TypeVar
from tqdm.auto import tqdm

from pyspoc._argchecking import check_natural_number
from pyspoc.settings import settings


PairwiseSymmetryType: TypeAlias = Literal["exact", "negative", "reciprocal", "triangular"]
PairwiseOperation: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray | float]
PairIndex: TypeAlias = tuple[int, int]
PairResult: TypeAlias = tuple[int, int, np.ndarray | float]
PreparedPairValue = TypeVar("PreparedPairValue")

_DEFAULT_CHUNKS_PER_WORKER = 8


def execute_pairwise(
    data: np.ndarray,
    operation: PairwiseOperation,
    data_b: np.ndarray | None = None,
    *,
    symmetry_type: PairwiseSymmetryType | None = None,
    max_workers: int | None = None,
    chunks_per_worker: int = _DEFAULT_CHUNKS_PER_WORKER,
    throw_first_error: bool = False,
    tqdm_pos: int | None = None
) -> tuple[np.ndarray, list[Exception]]:

    """Apply an operation to all required pairs of data columns concurrently.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional first input whose columns form the objects being compared.
        The array is shared read-only by convention across worker threads.
    operation : Callable
        Function accepting two one-dimensional column views and returning the
        value assigned to their result-matrix entry.
    data_b : numpy.ndarray or None, optional
        Optional second input. When supplied, every column of ``data`` is
        compared with every column of ``data_b`` and the result is rectangular.
    symmetry_type : {"exact", "negative", "reciprocal", "triangular}, optional
        Relationship used to derive ``(j, i)`` from a computed ``(i, j)``.
        ``None`` computes every ordered pair independently. Symmetry is only
        valid when ``data_b`` is omitted.
    max_workers : int or None, optional
        Maximum worker threads. If ``None``, use
        ``settings.current.max_worker_threads`` and then the logical CPUs
        available to this process when that setting is also ``None``.
    chunks_per_worker : int, default=8
        Target number of dynamically scheduled chunks per worker. More chunks
        improve load balancing; fewer reduce scheduling overhead.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(data.shape[1], data_b.shape[1])``. When
        ``data_b`` is omitted, both dimensions are ``data.shape[1]``.

    Raises
    ------
    TypeError
        If an integer execution argument has an incompatible type.
    ValueError
        If input rank, symmetry, or execution arguments are invalid.

    Notes
    -----
    One thread pool is created for the complete matrix. Pair indices are
    divided into more chunks than workers, and the executor assigns the next
    chunk whenever a worker becomes available. This retains dynamic load
    balancing without allocating one Future for every pair.

    Each chunk receives a separate copy of the caller's context so temporary
    pySPoC settings remain visible without entering one Context concurrently
    from several threads.
    """

    # Resolve basic argument requirements.
    if data.ndim != 2:
        raise ValueError("Pairwise execution requires a two-dimensional data array.")
    if data_b is not None and data_b.ndim != 2:
        raise ValueError("Pairwise execution requires a two-dimensional second data array.")
    if data_b is not None and symmetry_type is not None:
        raise ValueError("Pairwise symmetry cannot be used with two data arrays.")
    if symmetry_type not in (None, "exact", "negative", "reciprocal", "triangular"):
        raise ValueError(f"Unsupported pairwise {symmetry_type = !r}.")
    if tqdm_pos is None:
        tqdm_pos = 2

    # Resolve semantic argument requirements.
    check_natural_number(chunks_per_worker, arg_name="chunks_per_worker")
    worker_limit = resolve_worker_limit(max_workers)

    # Get second data matrix information (if provided)
    column_count = data.shape[1]
    second_column_count = column_count if data_b is None else data_b.shape[1]
    second_data = data if data_b is None else data_b

    # Initialise the results matrix
    pair_count = (
        column_count * second_column_count
        if data_b is not None or symmetry_type is None
        else column_count * (column_count + 1) // 2
    )
    result = np.empty((column_count, second_column_count), dtype=float)

    if pair_count == 0:
        return result, []

    # Set up async worker count and chunking.
    worker_count = min(worker_limit, pair_count)
    target_chunk_count = min(pair_count, worker_count * chunks_per_worker)
    chunk_size = math.ceil(pair_count / target_chunk_count)
    chunks = _chunk_pairs(
        _iter_pair_indices(column_count, second_column_count, symmetry_type),
        chunk_size,
    )

    # Initialise exceptions list.
    exceptions = []

    # Avoid executor setup when configuration or input size permits only one
    # worker. The same chunk machinery keeps serial and concurrent semantics
    # aligned.
    if worker_count == 1:
        chunks = list(chunks)
        n_chunks = len(chunks)

        with tqdm(
            total=n_chunks,
            position=tqdm_pos,
            leave=False) as chunk_pbar:

            for chunk in chunks:

                chunk_results, chunk_exceptions = _compute_chunk(
                    data,
                    second_data,
                    operation,
                    chunk,
                    throw_first_error)
                
                _store_chunk_results(
                    result,
                    chunk_results,
                    symmetry_type,
                )
                exceptions.extend(chunk_exceptions)
                chunk_pbar.update(1)

        return result, exceptions

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pyspoc-pairwise",
    )
    futures: list[Future[list[PairResult]]] = []

    try:
        for chunk in chunks:
            # Context objects cannot be entered by multiple threads at once;
            # create an independent snapshot for each scheduled chunk.
            context = copy_context()
            future = executor.submit(
                context.run,
                _compute_chunk,
                data,
                second_data,
                operation,
                chunk,
                throw_first_error
            )
            futures.append(future)

        n_futures = len(futures)

        with tqdm(
            total=n_futures,
            position=tqdm_pos,
            leave=False) as future_pbar:

            # Consume chunks in completion order. A fast worker immediately takes
            # another queued chunk rather than waiting for earlier submissions.
            for i, future in enumerate(as_completed(futures)):
                chunk_results, chunk_exceptions = future.result()
                _store_chunk_results(result, chunk_results, symmetry_type)
                exceptions.extend(chunk_exceptions)
                future_pbar.update(1)

    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return result, exceptions


def execute_pairwise_barrier(
    data: np.ndarray,
    prepare: Callable[[np.ndarray], PreparedPairValue],
    operation: Callable[[PreparedPairValue, PreparedPairValue], float],
    data_b: np.ndarray | None = None,
    *,
    max_workers: int | None = None,
    anchor_count: int | None = None,
    throw_first_error: bool = False,
    tqdm_pos: int | None = None
) -> tuple[np.ndarray, list[Exception]]:

    """Execute symmetric pair comparisons with a bounded prepared-data cache.

    At most one prepared value outside the anchor block is retained, and every
    comparison against that shared value must finish before the next value is
    prepared. Consequently, persistent prepared storage is bounded by
    ``anchor_count + 1`` values.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional first input whose columns are prepared as anchors.
    prepare : Callable
        Convert one column into the potentially expensive representation used
        by ``operation``.
    operation : Callable
        Compare two prepared values and return one scalar. The comparison is
        assumed to be symmetric when ``data_b`` is omitted.
    data_b : numpy.ndarray or None, optional
        Optional second input. Its prepared columns are streamed through each
        anchor block, producing a rectangular cross-comparison matrix.
    max_workers : int or None, optional
        Maximum worker threads, resolved identically to :func:`execute_pairwise`.
    anchor_count : int or None, optional
        Maximum prepared anchors retained in one block. If ``None``, use the
        resolved worker count. The worker count is capped by this value.

    Returns
    -------
    numpy.ndarray
        Symmetric square matrix when ``data_b`` is omitted; otherwise a matrix
        with shape ``(data.shape[1], data_b.shape[1])``.

    Notes
    -----
    Anchor values are prepared concurrently, up to ``max_workers`` at a time,
    and must all complete before pair comparisons begin. Values streamed after
    the anchor block remain sequential so only one non-anchor prepared value
    exists during a comparison phase.

    Later columns are prepared again for each preceding anchor block. This is
    the deliberate computation-for-memory trade-off that permits a strict
    cache bound without storing every prepared column. Decoupling anchors from
    workers lets serial execution reuse several expensive prepared values
    whenever memory permits.
    """

    # Basic argument checking.
    if data.ndim != 2:
        raise ValueError("Pairwise execution requires a two-dimensional data array.")
    if data_b is not None and data_b.ndim != 2:
        raise ValueError("Pairwise execution requires a two-dimensional second data array.")
    if tqdm_pos is None:
        tqdm_pos = 2

    # Get column counts and initialise result matrix.
    column_count = data.shape[1]
    second_column_count = column_count if data_b is None else data_b.shape[1]
    result = np.empty((column_count, second_column_count), dtype=float)

    if column_count == 0 or second_column_count == 0:
        return result, []

    # Set up the worker count.
    requested_workers = min(resolve_worker_limit(max_workers), column_count)

    # Set up the anchors. These are memoised results from a certain number of
    # computations that speed up pairwise comparisons. The number stored in memory
    # is dictated by memory availability.
    if anchor_count is None:
        selected_anchor_count = requested_workers
    else:
        selected_anchor_count = min(
            check_natural_number(anchor_count, arg_name="anchor_count"),
            column_count,
        )

    # Get the resolved worker count.
    worker_count = min(requested_workers, selected_anchor_count)

    # Retain the allocation-free serial fast path only when there is also a
    # single anchor. Multiple cached anchors still require blocked scheduling,
    # even though its executor may contain only one worker.
    if worker_count == 1 and selected_anchor_count == 1:
        return _execute_pairwise_barrier_serial(
            data, prepare, operation, data_b, throw_first_error)

    # Initialise exceptions for gathering pairwise errors.
    exceptions = []

    # Begin thread pool execution.
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pyspoc-pairwise-barrier",
    ) as executor:

        n_iter = 0
        block_starts = list(range(0, column_count, selected_anchor_count))

        for block_start in block_starts:
            j_start = block_start if data_b is None else 0
            n_iter += max(0, second_column_count - j_start)

        with tqdm(
                total=n_iter,
                position=tqdm_pos,
                leave=False) as iter_pbar:

            # Iterate through the left pairwise blocks.
            for block_start in block_starts:
                block_stop = min(block_start + selected_anchor_count, column_count)
                anchor_indices = tuple(range(block_start, block_stop))
                anchors = _prepare_anchor_block(
                    executor,
                    data,
                    prepare,
                    anchor_indices,
                )

                # Iterate through the right pairwise comparison blocks.
                j_start = block_start if data_b is None else 0

                for j in range(j_start, second_column_count):
                    within_anchor_block = data_b is None and j < block_stop
                    shared = (
                        anchors[j - block_start]
                        if within_anchor_block
                        else prepare((data if data_b is None else data_b)[:, j])
                    )
                    active_anchors = (
                        j - block_start + 1
                        if within_anchor_block
                        else len(anchor_indices)
                    )
                    futures: list[Future[float]] = []

                    try:
                        for anchor in anchors[:active_anchors]:
                            context = copy_context()
                            futures.append(
                                executor.submit(context.run, operation, anchor, shared)
                            )
                        del anchor

                        # Resolving every Future is the phase barrier: ``shared``
                        # remains alive and the next column is not prepared until
                        # the complete worker group has finished with this one.
                        for offset, future in enumerate(futures):
                            i = anchor_indices[offset]

                            try:
                                value = future.result()
                            except Exception as e:
                                e.add_note(f"Exception occurred for pair {i=}, {j=}.")

                                if throw_first_error:
                                    raise e
                                
                                exceptions.append(e)
                                value = np.nan

                            result[i, j] = value

                            if data_b is None:
                                result[j, i] = value

                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
                    finally:
                        # Do not retain a completed phase's streamed value while
                        # constructing the next potentially large value.
                        del shared
                        del futures
                        iter_pbar.update(1)

            # Assignment of the next block would otherwise evaluate its
            # preparations while this complete block was still referenced.
            del anchors

    return result, exceptions


def _prepare_anchor_block(
    executor: ThreadPoolExecutor,
    data: np.ndarray,
    prepare: Callable[[np.ndarray], PreparedPairValue],
    anchor_indices: Sequence[int],
) -> tuple[PreparedPairValue, ...]:
    """Prepare an anchor block concurrently and return it in index order.

    Every preparation receives its own context snapshot so execution-scoped
    pySPoC settings propagate into worker threads. Resolving all futures before
    returning forms the preparation barrier: pair comparisons cannot begin
    with a partially constructed anchor cache.
    """
    futures: list[Future[PreparedPairValue]] = []

    try:
        for i in anchor_indices:
            context = copy_context()
            futures.append(
                executor.submit(context.run, prepare, data[:, i]))

        # Futures are resolved in anchor-index order so result coordinates do
        # not depend on which kernel happens to finish first.
        return tuple(future.result() for future in futures)
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def _execute_pairwise_barrier_serial(
    data: np.ndarray,
    prepare: Callable[[np.ndarray], PreparedPairValue],
    operation: Callable[[PreparedPairValue, PreparedPairValue], float],
    data_b: np.ndarray | None = None,
    throw_first_error: bool = False,
    tqdm_pos: int | None = None
) -> tuple[np.ndarray, list[Exception]]:
    
    """Serial implementation retaining one anchor and one shared value."""

    if tqdm_pos is None:
        tqdm_pos = 2

    # Column count and results matrix initialisation.
    column_count = data.shape[1]
    second_column_count = column_count if data_b is None else data_b.shape[1]
    result = np.empty((column_count, second_column_count), dtype=float)

    # Initialise exceptions for gathering pairwise errors.
    exceptions = []

    cols = list(range(column_count))
    n_iter = 0

    for i in cols:
        j_start = i if data_b is None else 0
        n_iter += max(0, second_column_count - j_start)

    with tqdm(
            total=n_iter,
            position=tqdm_pos,
            leave=False) as iter_pbar:

        # Left pairwise iteration
        for i in cols:
            anchor = prepare(data[:, i])
            
            # Right pairwise iteration
            j_start = i if data_b is None else 0
            for j in range(j_start, second_column_count):
                shared = (
                    anchor
                    if data_b is None and i == j
                    else prepare((data if data_b is None else data_b)[:, j])
                )

                try:
                    value = operation(anchor, shared)
                except Exception as e:
                    e.add_note(f"Exception occurred for pair {i=}, {j=}.")

                    if throw_first_error:
                        raise e
                    
                    exceptions.append(e)
                    value = np.nan

                result[i, j] = value

                if data_b is None:
                    result[j, i] = value

                del shared
                iter_pbar.update(1)

    return result, exceptions


def resolve_worker_limit(max_workers: int | None) -> int:
    """Resolve an explicit, configured, or process-aware worker limit."""
    if max_workers is not None:
        return check_natural_number(max_workers, arg_name="max_workers")

    configured_workers = settings.current.max_worker_threads
    if configured_workers is not None:
        return check_natural_number(
            configured_workers,
            arg_name="settings.max_worker_threads",
        )

    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    return process_cpu_count() or 1


def _iter_pair_indices(
    column_count: int,
    second_column_count: int,
    symmetry_type: PairwiseSymmetryType | None) -> Iterator[PairIndex]:
    """Yield ordered or upper-triangular pair coordinates lazily."""

    for i in range(column_count):
        first_j = 0 if symmetry_type is None else i

        for j in range(first_j, second_column_count):
            yield i, j


def _chunk_pairs(
    pairs: Iterator[PairIndex],
    chunk_size: int,) -> Iterator[tuple[PairIndex, ...]]:
    """Yield bounded immutable batches from a lazy pair-index iterator."""

    chunk: list[PairIndex] = []

    for pair in pairs:
        chunk.append(pair)
        if len(chunk) == chunk_size:
            yield tuple(chunk)
            chunk.clear()

    if chunk:
        yield tuple(chunk)


def _compute_chunk(
    data: np.ndarray,
    data_b: np.ndarray,
    operation: PairwiseOperation,
    pairs: Sequence[PairIndex],
    throw_first_error: bool = False) -> tuple[list[PairResult], list[Exception]]:
    """Compute one batch serially inside a worker thread."""

    results = list()
    exceptions = list()
    
    for i, j in pairs:
        try:
            value = operation(data[:, i], data_b[:, j])
        except Exception as e:
            e.add_note(f"Exception occurred for pair {i=}, {j=}.")

            if throw_first_error:
                raise e

            exceptions.append(e)
            value = np.nan
            
        finally:
            result = (i,j, value)
            results.append(result)
    
    return results, exceptions


def _store_chunk_results(
    output: np.ndarray,
    chunk_results: Sequence[PairResult],
    symmetry_type: PairwiseSymmetryType | None):
    """Store completed values and derive their opposite-triangle entries."""

    for i, j, value in chunk_results:
        output[i, j] = value

        if i == j or symmetry_type is None:
            continue

        match symmetry_type:
            case "exact":
                output[j, i] = value
            case "negative":
                output[j, i] = -value
            case "reciprocal":
                output[j, i] = 1 / value
            case "triangular":
                output[j, i] = 0

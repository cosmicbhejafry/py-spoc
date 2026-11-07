"""Concurrent execution of scalar operations over pairs of data columns."""

from __future__ import annotations

import os
import numpy as np

from collections.abc import Iterator
from typing import Callable, Concatenate, TypeAlias, Sequence, TypeGuard, overload
from functools import partial
from math import ceil

from pyspoc.settings import settings
from pyspoc._argchecking import check_natural_number
from .execution import (
    execute_parallel,
    execute_serial,
    ChunkOperation,
    ChunkResult,
    OperationIndex,
    OperationInput)


Operation: TypeAlias = Callable[
    Concatenate[np.ndarray, ...],
    np.ndarray | float,
]


def prepare_chunk_operation(
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        operation: Operation,
        inputs: Iterator[tuple[OperationIndex, OperationInput]],
        n_operations: int,
        *,
        max_workers: int | None = None,
        throw_first_error: bool = False
) -> tuple[
        ChunkOperation | None,
        Iterator[Sequence[tuple[OperationIndex, OperationInput]]],
        int,
        np.ndarray[tuple[int], np.dtype[np.floating]]]:

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
        available to this process when that setting is ``-1`` or ``None``.
    chunks_per_worker : int, default=8
        Target number of dynamically scheduled chunks per worker. More chunks
        improve load balancing; fewer reduce scheduling overhead.
    indexed_operation : Callable or None, optional
        Alternative callback accepting ``(x, y, (i, j))``. When supplied, this
        is called instead of ``operation``, exposing stable column indices for
        per-pair estimator lookup without inspecting or hashing column values.

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
        raise ValueError("Execution requires a two-dimensional data array.")

    # Resolve semantic argument requirements.
    worker_limit = resolve_worker_limit(max_workers)

    # Initialise the results matrix
    results = np.empty(n_operations, dtype=float)

    # Set up async worker count and chunking.
    n_workers = min(worker_limit, n_operations)
    chunk_size = ceil(n_operations / n_workers)
    chunks = _chunk_inputs(inputs, chunk_size=chunk_size)

    chunk_operation = partial(
        _compute_chunk,
            data=data,
            operation=operation,
            throw_first_error=throw_first_error)
    
    return chunk_operation, chunks, n_workers, results


@overload
def execute_async(
        data: np.ndarray,
        operation: Operation,
        *,
        inputs: Iterator[tuple[OperationIndex, OperationInput]] | Iterator[OperationInput],
        n_operations: int,
        max_workers: int | None = None,
        throw_first_error: bool = False,
        tqdm_pos: int | None = 2,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], list[Exception]]:
    ...


@overload
def execute_async(
        data: np.ndarray,
        operation: Operation,
        *,
        inputs: Sequence[tuple[OperationIndex, OperationInput]] | Sequence[OperationInput],
        max_workers: int | None = None,
        throw_first_error: bool = False,
        tqdm_pos: int | None = 2,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], list[Exception]]:
    ...


@overload
def execute_async(
        data: np.ndarray,
        operation: Operation,
        *,
        n_operations: int,
        max_workers: int | None = None,
        throw_first_error: bool = False,
        tqdm_pos: int | None = 2,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], list[Exception]]:
    ...


def execute_async(
        data: np.ndarray,
        operation: Operation,
        *,
        inputs: (Iterator[tuple[OperationIndex, OperationInput]] |
                 Iterator[OperationInput] |
                 Sequence[tuple[OperationIndex, OperationInput]] |
                 Sequence[OperationInput] |
                 None) = None,
        n_operations: int | None = None,
        max_workers: int | None = None,
        throw_first_error: bool = False,
        tqdm_pos: int | None = 2,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], list[Exception]]:

    """Apply an operation to all required pairs of data columns concurrently.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional first input whose columns form the objects being compared.
        The array is shared read-only by convention across worker threads.
    operation : Callable
        Function accepting data array and any additional key-word supporting arguments,
        returning a scalar value that will be assigned to the result matrix.
    inputs : Iterator[tuple[OperationIndex, OperationInput]] or \
             Iterator[OperationInput] or \
             Sequence[tuple[OperationIndex, OperationInput]] or \
             Sequence[OperationInput] or \
             None

        Iterator or sequence of operation key-word arguments as a dictionary. Optionally,
        it can include the indexes to which the results will be assigned in the result matrix.
    n_operations: int or None, required if `inputs` is an `Iterator` or `None`
        Specifies the number of operation executions. If a :class:`Sequence` is provided for
        `inputs` argument this parameter can be left blank.
    max_workers : int or None, optional
        Maximum worker threads. If ``None``, use
        ``settings.current.max_worker_threads`` and then the logical CPUs
        available to this process when that setting is ``-1`` or ``None``.
    throw_first_error: bool, default = False
        Declares whether the first error encountered in the execution is immediately thrown.
        When set to `False`, exceptions are gathered and returned as a sequence.
    tqdm_pos: int or None, optional, default=2
        Specifies the `tqdm` position or level for exposing a `tqdm` loading bar.

    Returns
    -------
    tuple[np.ndarray[tuple[int], np.dtype[np.floating]], list[Exception]]
        - Matrix with shape ``(n_operations,)``, provided by the parameter `n_operations`
        or derived from the length of `inputs` if a :class:`Sequence`
        - A list of exceptions if encountered.

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
    if n_operations is None:
        if isinstance(inputs, Iterator):
            raise ValueError("When providing an iterator of inputs, n_operations "
                "must be specified to allow preallocation of the results array.")

        if inputs is None:
            raise ValueError("When providing no inputs, n_operations must be specified "
                "to allow preallocation of the results array.")

        n_operations = len(inputs)

        if n_operations == 0:
            raise ValueError("n_operations must be provided when the sequence of inputs is empty.")
    
    else:
        try:
            check_natural_number(n_operations, "n_operations")
        except:
            raise ValueError("n_operations must be a positive integer.")
        
    inputs = _resolve_inputs(inputs, n_operations=n_operations)
    
    chunk_operation, chunks, worker_count, results = prepare_chunk_operation(
        data=data,
        operation=operation,
        inputs=inputs,
        max_workers=max_workers,
        n_operations=n_operations,
        throw_first_error=throw_first_error)

    # Get second data matrix information (if provided)
    if chunk_operation is None:
        return results, []
    
    # Initialise exceptions list.
    exceptions = []

    # Avoid executor setup when configuration or input size permits only one
    # worker. The same chunk machinery keeps serial and concurrent semantics
    # aligned.
    if worker_count == 1:
        results, exceptions = execute_serial(
            chunks=chunks,
            results=results,
            chunk_operation=chunk_operation,
            tqdm_pos=tqdm_pos)
    else:
        results, exceptions = execute_parallel(
            chunk_operation=chunk_operation,
            chunks=chunks,
            worker_count=worker_count,
            results=results,
            tqdm_pos=tqdm_pos)

    return results, exceptions


def resolve_worker_limit(max_workers: int | None) -> int:
    """Resolve an explicit, configured, or process-aware worker limit."""
    if max_workers is not None:
        return check_natural_number(max_workers, arg_name="max_workers")

    configured_workers = settings.current.max_worker_threads
    if configured_workers is not None and configured_workers != -1:
        return check_natural_number(
            configured_workers,
            arg_name="settings.max_worker_threads",
        )

    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    return process_cpu_count() or 1


def _resolve_inputs(
    inputs: (Iterator[tuple[OperationIndex, OperationInput]] |
             Iterator[OperationInput] |
             Sequence[tuple[OperationIndex, OperationInput]] |
             Sequence[OperationInput] |
             None),
    n_operations: int
) -> (Iterator[tuple[OperationIndex, OperationInput]]):
    index = 0
    n_inputs_processed = 0

    if inputs is not None:
        for input in inputs:
            n_inputs_processed += 1
            if _is_unindexed_input(input):
                standardised: tuple[OperationIndex, OperationInput] = ((index,), input)
                yield standardised
                index += 1
            
            if isinstance(input, tuple):
                yield input

    # Blank iterator/sequence provided, assume no args required.
    if n_inputs_processed == 0:
        for i in range(n_operations):
            yield ((index,), {})
            index += 1

    # Otherwise, if the inputs are incomplete, we have an error.
    elif n_inputs_processed < n_operations:
        raise ValueError("The number of inputs provided is less than the "
            "declared number of operations. ")

    elif n_inputs_processed > n_operations:
        raise ValueError("The number of inputs exceeds the "
            "declared number of operations. ")


def _is_unindexed_input(
        input: tuple[OperationIndex, OperationInput] | OperationInput
) -> TypeGuard[OperationInput]:
    return (isinstance(input, dict))



def _chunk_inputs(
    inputs: Iterator[tuple[OperationIndex, OperationInput]],
    chunk_size: int
) -> Iterator[Sequence[tuple[OperationIndex, OperationInput]]]:
    """Yield bounded immutable batches from a lazy pair-index iterator."""

    chunks: list[tuple[OperationIndex, OperationInput]] = []
    
    for input in inputs:
        chunks.append(input)

        if len(chunks) == chunk_size:
            yield tuple(chunks)
            chunks.clear()
    
    if chunks:
        yield tuple(chunks)


def _compute_chunk(
    chunks: Sequence[tuple[OperationIndex, OperationInput]],
    *,
    data: np.ndarray,
    operation: Operation,
    throw_first_error: bool = False,
) -> tuple[list[ChunkResult], list[Exception]]:
    """Compute one batch serially inside a worker thread."""

    results: list[ChunkResult] = list()
    exceptions: list[Exception] = list()
    
    for chunk in chunks:
        index, kwargs = chunk

        try:
            value = operation(data, **kwargs)

        except Exception as e:
            e.add_note(f"Exception occurred for chunk with arguments {chunk}.")

            if throw_first_error:
                raise e

            exceptions.append(e)
            value = np.nan
            
        # Only store a value after successful computation or handled failure.
        # A finally block here would mask a re-raised exception if value was
        # never assigned by the failing operation.
        result = (index, value)
        results.append(result)
    
    return results, exceptions

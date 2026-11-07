"""Concurrent execution of scalar operations over numpy arrays."""

from __future__ import annotations

import numpy as np

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import TypeAlias, Sequence, Any
from tqdm.auto import tqdm


OperationInput: TypeAlias = dict[str, Any]
OperationIndex: TypeAlias = tuple[int, ...]
ChunkResult: TypeAlias = tuple[OperationIndex, np.ndarray | float]
ChunkOperation: TypeAlias = Callable[
    [Sequence[tuple[OperationIndex, OperationInput]]], tuple[list[ChunkResult], list[Exception]]]


def execute_parallel(
        chunk_operation: ChunkOperation,
        chunks: Iterator[Sequence[tuple[OperationIndex, OperationInput]]],
        worker_count: int,
        results: np.ndarray,
        *,
        tqdm_pos: int | None = None
) -> tuple[np.ndarray, list[Exception]]:

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pyspoc-parallel",
    )
    futures: list[Future[tuple[list[ChunkResult], list[Exception]]]] = []
    exceptions: list[Exception] = []

    try:
        for chunk in chunks:
            # Context objects cannot be entered by multiple threads at once;
            # create an independent snapshot for each scheduled chunk.
            context = copy_context()
            future = executor.submit(
                context.run,
                chunk_operation,
                chunk
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

                for chunk_result in chunk_results:
                    index, value = chunk_result
                    results[index] = value

                exceptions.extend(chunk_exceptions)
                future_pbar.update(1)

    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return results, exceptions


def execute_serial(
        chunk_operation: ChunkOperation,
        chunks: Iterator[Sequence[tuple[OperationIndex, OperationInput]]],
        results: np.ndarray,
        *,
        tqdm_pos: int | None = None
) -> tuple[np.ndarray, list[Exception]]:

    # Initialise exceptions list.
    exceptions: list[Exception] = []

    # Avoid executor setup when configuration or input size permits only one
    # worker. The same chunk machinery keeps serial and concurrent semantics
    # aligned.
    chunk_list = list(chunks)
    n_chunks = len(chunk_list)

    with tqdm(
        total=n_chunks,
        position=tqdm_pos,
        leave=False) as chunk_pbar:

        for chunk in chunk_list:
            chunk_results, chunk_exceptions = chunk_operation(chunk)

            for chunk_result in chunk_results:
                index, value = chunk_result
                results[index] = value

            exceptions.extend(chunk_exceptions)
            chunk_pbar.update(1)
        
        return results, exceptions



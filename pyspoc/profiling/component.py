"""Measure CPU parallelism used while a Component computes a result."""

from __future__ import annotations

import os
import threading
import time

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import numpy as np

from pyspoc._argchecking import check_integer, check_natural_number
from pyspoc.exceptions import OptionalDependencyMissingError
from pyspoc.settings import settings
from .reports import (
    CPUProfileRun,
    CPUProfileSample,
    ComponentCPUProfile,
    CPUScalingProfileEntry,
    ComponentCPUScalingProfile
)

try:
    import psutil
except ImportError as error:  # pragma: no cover - depends on optional environment
    raise OptionalDependencyMissingError(
        "psutil",
        feature="Component CPU profiling",
        install_hint="Install pySPoC with the 'profiling' extra.",
    ) from error


if TYPE_CHECKING:
    from .reports import ComputableComponent

def profile_component_cpu(
    component: ComputableComponent,
    data: np.ndarray,
    *,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    sample_interval: float = 0.05,
    copy_data: bool = True,
) -> ComponentCPUProfile:
    """Profile CPU resources used by a Component's ``compute`` method.

    Parameters
    ----------
    component : ComputableComponent
        Statistic or Reducer instance exposing ``compute(data)``.
    data : numpy.ndarray
        Input supplied directly to ``component.compute``. For a Statistic this
        is dataset data; for a Reducer it is a previously computed Statistic
        result.
    warmup_runs : int, default=1
        Number of unmeasured invocations used to initialize imports, compiled
        code, and retained worker pools.
    measured_runs : int, default=3
        Number of sampled invocations included in the report.
    sample_interval : float, default=0.05
        Seconds between process-tree observations. Shorter intervals improve
        detection of brief workers but add profiling overhead.
    copy_data : bool, default=True
        If ``True``, give every invocation a fresh copy prepared before timing.
        This prevents one run's input mutation from affecting later runs.

    Returns
    -------
    ComponentCPUProfile
        Immutable aggregate report containing measured runs and raw samples.

    Raises
    ------
    TypeError
        If ``component`` has no callable ``compute`` method, ``data`` is not a
        NumPy array, or integer parameters have incompatible types.
    ValueError
        If run counts or the sampling interval are outside their valid ranges.

    Notes
    -----
    The operation runs in the calling process. Other work executing in that
    process or its descendants can contribute to the measurements. Avoid
    running unrelated work while profiling and use inputs whose computation
    lasts several seconds.
    """
    compute = getattr(component, "compute", None)
    
    if not callable(compute):
        raise TypeError("component must expose a callable compute(data) method.")
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")

    check_integer(warmup_runs, minimum=0, arg_name="warmup_runs")
    check_natural_number(measured_runs, arg_name="measured_runs")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive.")

    def make_input() -> np.ndarray:
        return np.array(data, copy=True) if copy_data else data

    for _ in range(warmup_runs):
        compute(make_input())

    root_process = psutil.Process()
    available_cpus = _get_available_logical_cpus(root_process)
    runs = []
    for _ in range(measured_runs):
        # Prepare a defensive input copy before timing so the report measures
        # Component computation rather than memory-allocation throughput.
        run_data = make_input()
        runs.append(
            _profile_operation(
                lambda run_data=run_data: compute(run_data),
                root_process=root_process,
                available_cpus=available_cpus,
                sample_interval=sample_interval,
            )
        )

    component_type = f"{type(component).__module__}.{type(component).__qualname__}"
    return ComponentCPUProfile(
        component_type=component_type,
        available_logical_cpus=available_cpus,
        warmup_runs=warmup_runs,
        runs=tuple(runs),
    )


def profile_component_cpu_scaling(
    component: ComputableComponent,
    data: np.ndarray,
    *,
    maximum_workers: int | None = None,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    sample_interval: float = 0.05,
    copy_data: bool = True,
) -> ComponentCPUScalingProfile:
    """Profile a Component with geometrically increasing worker limits.

    Parameters
    ----------
    component : ComputableComponent
        Statistic or Reducer instance exposing ``compute(data)``.
    data : numpy.ndarray
        Input supplied directly to ``component.compute``.
    maximum_workers : int or None, optional
        Largest pySPoC worker limit to measure. If ``None``, use
        ``settings.current.max_worker_threads`` when configured, otherwise use
        the logical CPUs available to the process.
    warmup_runs : int, default=1
        Unmeasured invocations performed for every worker count.
    measured_runs : int, default=3
        Sampled invocations performed for every worker count.
    sample_interval : float, default=0.05
        Seconds between process-tree observations.
    copy_data : bool, default=True
        If ``True``, give every invocation a fresh input copy prepared outside
        the measured interval.

    Returns
    -------
    ComponentCPUScalingProfile
        Immutable scaling report. Worker limits increase in powers of two and
        always include ``maximum_workers``. For example, a maximum of six
        produces ``(1, 2, 4, 6)``.

    Raises
    ------
    TypeError
        If ``maximum_workers`` is not an integer or another profiling argument
        has an incompatible type.
    ValueError
        If ``maximum_workers`` is less than one or another profiling argument
        is outside its valid range.

    Notes
    -----
    The worker setting affects Components that cooperate with pySPoC's
    execution machinery. Components that manage parallelism independently may
    produce similar measurements at every worker limit; that is itself useful
    diagnostic evidence.
    """
    root_process = psutil.Process()

    if maximum_workers is None:
        maximum_workers = settings.current.max_worker_threads
        if maximum_workers is None:
            maximum_workers = _get_available_logical_cpus(root_process)

    check_natural_number(maximum_workers, arg_name="maximum_workers")
    worker_counts = _get_geometric_worker_counts(maximum_workers)
    entries = []

    for worker_count in worker_counts:
        with settings.override(max_worker_threads=worker_count):
            profile = profile_component_cpu(
                component,
                data,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                sample_interval=sample_interval,
                copy_data=copy_data,
            )

        entries.append(
            CPUScalingProfileEntry(
                worker_count=worker_count,
                profile=profile,
            )
        )

    component_type = f"{type(component).__module__}.{type(component).__qualname__}"
    return ComponentCPUScalingProfile(
        component_type=component_type,
        maximum_workers=maximum_workers,
        entries=tuple(entries),
    )


def _profile_operation(
    operation: Callable[[], Any],
    *,
    root_process: psutil.Process,
    available_cpus: int,
    sample_interval: float,
) -> CPUProfileRun:
    """Execute and monitor one operation, propagating its original errors."""
    monitor = _ProcessTreeMonitor(root_process, sample_interval)
    monitor.start()
    start_time = time.perf_counter()

    try:
        operation()
    finally:
        wall_seconds = time.perf_counter() - start_time
        monitor.stop()

    samples = monitor.samples
    cpu_seconds = monitor.cpu_seconds
    average_cpu_cores = cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0
    machine_percent = 100 * average_cpu_cores / available_cpus

    return CPUProfileRun(
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        average_cpu_cores=average_cpu_cores,
        average_machine_cpu_percent=machine_percent,
        peak_cpu_cores=max((sample.cpu_cores for sample in samples), default=0.0),
        peak_process_count=max((sample.process_count for sample in samples), default=1),
        peak_child_process_count=max(
            (sample.child_process_count for sample in samples),
            default=0,
        ),
        peak_thread_count=max((sample.thread_count for sample in samples), default=1),
        samples=samples,
    )


class _ProcessTreeMonitor:
    """Sample CPU time and worker counts for one process tree."""

    def __init__(self, root_process: psutil.Process, sample_interval: float) -> None:
        self._root_process = root_process
        self._sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(
            target=self._monitor,
            name="pyspoc-component-profiler",
            daemon=True,
        )
        self._start_time = 0.0
        self._last_sample_time = 0.0
        self._last_cpu_times: dict[tuple[int, float], float] = {}
        self._cpu_seconds = 0.0
        self._samples: list[CPUProfileSample] = []
        self._monitor_error: BaseException | None = None

    @property
    def cpu_seconds(self) -> float:
        """Return observed cumulative process-tree CPU time."""
        return self._cpu_seconds

    @property
    def samples(self) -> tuple[CPUProfileSample, ...]:
        """Return an immutable snapshot of the collected samples."""
        return tuple(self._samples)

    def start(self) -> None:
        """Start sampling and wait until the initial baseline is ready."""
        self._start_time = time.perf_counter()
        self._last_sample_time = self._start_time
        self._thread.start()
        self._ready_event.wait()

        if self._monitor_error is not None:
            raise RuntimeError(
                "Unable to start process-tree CPU monitoring."
            ) from self._monitor_error

    def stop(self) -> None:
        """Request a final sample, stop the monitor, and surface failures."""
        self._stop_event.set()
        self._thread.join()

        if self._monitor_error is not None:
            raise RuntimeError("Process-tree CPU monitoring failed.") from self._monitor_error

    def _monitor(self) -> None:
        """Record an initial baseline followed by periodic CPU deltas."""
        try:
            self._last_cpu_times, _, _, _ = self._snapshot_process_tree()
            self._ready_event.set()

            while not self._stop_event.wait(self._sample_interval):
                self._take_sample()

            self._take_sample()
        except BaseException as error:
            self._monitor_error = error
            self._ready_event.set()

    def _take_sample(self) -> None:
        """Record CPU-time growth and resource counts since the last sample."""
        current_time = time.perf_counter()
        current_cpu_times, process_count, child_count, thread_count = (
            self._snapshot_process_tree()
        )
        interval_seconds = current_time - self._last_sample_time
        interval_cpu_seconds = 0.0

        for identity, cpu_time in current_cpu_times.items():
            previous_cpu_time = self._last_cpu_times.get(identity, 0.0)
            interval_cpu_seconds += max(0.0, cpu_time - previous_cpu_time)

        self._cpu_seconds += interval_cpu_seconds
        interval_cpu_cores = (
            interval_cpu_seconds / interval_seconds
            if interval_seconds > 0
            else 0.0
        )
        self._samples.append(
            CPUProfileSample(
                elapsed_seconds=current_time - self._start_time,
                cpu_cores=interval_cpu_cores,
                process_count=process_count,
                child_process_count=child_count,
                thread_count=thread_count,
            )
        )
        self._last_sample_time = current_time
        self._last_cpu_times = current_cpu_times

    def _snapshot_process_tree(
        self,
    ) -> tuple[dict[tuple[int, float], float], int, int, int]:
        """Return CPU times and worker counts for live tree members."""
        try:
            processes = [
                self._root_process,
                *self._root_process.children(recursive=True),
            ]
        except psutil.Error:
            processes = [self._root_process]

        cpu_times: dict[tuple[int, float], float] = {}
        thread_count = 0
        live_process_count = 0

        for process in processes:
            try:
                times = process.cpu_times()
                identity = (process.pid, process.create_time())
                cpu_times[identity] = times.user + times.system
                thread_count += process.num_threads()
                live_process_count += 1
            except psutil.Error:
                continue

        child_count = max(0, live_process_count - 1)
        return cpu_times, live_process_count, child_count, thread_count


def _get_available_logical_cpus(process: psutil.Process) -> int:
    """Return logical CPUs available under affinity or platform defaults."""
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, NotImplementedError, psutil.Error):
        affinity = []

    if affinity:
        return len(affinity)

    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    return process_cpu_count() or 1


def _get_geometric_worker_counts(maximum_workers: int) -> tuple[int, ...]:
    """Return powers of two followed by an exact non-power maximum."""
    worker_counts = []
    worker_count = 1

    while worker_count <= maximum_workers:
        worker_counts.append(worker_count)
        worker_count *= 2

    if worker_counts[-1] != maximum_workers:
        worker_counts.append(maximum_workers)

    return tuple(worker_counts)

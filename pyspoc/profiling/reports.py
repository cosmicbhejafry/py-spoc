
import statistics
import numpy as np
import pandas as pd

from typing import Any, Protocol
from dataclasses import dataclass

class ComputableComponent(Protocol):
    """Structural contract required by :func:`profile_component_cpu`."""

    def compute(self, data: np.ndarray) -> Any:
        """Compute a result from an input array."""


@dataclass(frozen=True, slots=True)
class CPUProfileSample:
    """Represent one process-tree sample taken during a computation.

    Attributes
    ----------
    elapsed_seconds : float
        Wall time since the start of the measured computation.
    cpu_cores : float
        CPU time accrued during the preceding interval divided by that
        interval's wall time. A value near two indicates approximately two
        logical CPUs executing concurrently during the interval.
    process_count : int
        Number of observed processes, including the pySPoC process.
    child_process_count : int
        Number of observed descendant processes.
    thread_count : int
        Total operating-system threads across all observed processes. This
        includes idle threads and the profiler's sampling thread.
    """

    elapsed_seconds: float
    cpu_cores: float
    process_count: int
    child_process_count: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class CPUProfileRun:
    """Summarize one measured invocation of ``Component.compute``.

    Attributes
    ----------
    wall_seconds : float
        Elapsed wall-clock time.
    cpu_seconds : float
        CPU time observed across the root process and its descendants.
    average_cpu_cores : float
        CPU seconds divided by wall seconds.
    average_machine_cpu_percent : float
        Average CPU use as a percentage of the logical CPUs available to the
        process. One fully occupied CPU on a six-CPU allocation is about
        16.7 percent.
    peak_cpu_cores : float
        Largest interval CPU-use sample expressed as logical CPUs.
    peak_process_count : int
        Largest process-tree size observed.
    peak_child_process_count : int
        Largest number of descendants observed.
    peak_thread_count : int
        Largest total operating-system thread count observed. It includes
        idle worker pools and therefore is diagnostic rather than proof of
        concurrent execution.
    samples : tuple of CPUProfileSample
        Chronologically ordered samples for more detailed analysis.
    """

    wall_seconds: float
    cpu_seconds: float
    average_cpu_cores: float
    average_machine_cpu_percent: float
    peak_cpu_cores: float
    peak_process_count: int
    peak_child_process_count: int
    peak_thread_count: int
    samples: tuple[CPUProfileSample, ...]


@dataclass(frozen=True, slots=True)
class ComponentCPUProfile:
    """Aggregate repeated CPU-profile runs for a Component.

    Attributes
    ----------
    component_type : str
        Fully qualified class name of the profiled Component.
    available_logical_cpus : int
        Logical CPUs available to the current process when profiling began.
    warmup_runs : int
        Unmeasured invocations completed before sampling.
    runs : tuple of CPUProfileRun
        Individual measured runs.

    Notes
    -----
    CPU accounting for very short-lived child processes may be incomplete if
    a process starts and exits between samples. Profile representative inputs
    that run for several seconds and repeat measurements for stable results.
    """

    component_type: str
    available_logical_cpus: int
    warmup_runs: int
    runs: tuple[CPUProfileRun, ...]

    @property
    def median_wall_seconds(self) -> float:
        """Return the median elapsed wall time across measured runs."""
        return statistics.median(run.wall_seconds for run in self.runs)

    @property
    def median_cpu_seconds(self) -> float:
        """Return the median observed CPU time across measured runs."""
        return statistics.median(run.cpu_seconds for run in self.runs)

    @property
    def median_average_cpu_cores(self) -> float:
        """Return the median average number of concurrently occupied CPUs."""
        return statistics.median(run.average_cpu_cores for run in self.runs)

    @property
    def median_machine_cpu_percent(self) -> float:
        """Return median CPU use as a percentage of available CPU capacity."""
        return statistics.median(run.average_machine_cpu_percent for run in self.runs)

    @property
    def peak_cpu_cores(self) -> float:
        """Return the greatest interval CPU concurrency across all runs."""
        return max(run.peak_cpu_cores for run in self.runs)

    @property
    def peak_thread_count(self) -> int:
        """Return the greatest observed process-tree thread count."""
        return max(run.peak_thread_count for run in self.runs)

    @property
    def peak_child_process_count(self) -> int:
        """Return the greatest observed number of child processes."""
        return max(run.peak_child_process_count for run in self.runs)

    @property
    def used_multiple_cpu_cores(self) -> bool:
        """Return whether median CPU use provides evidence of parallel work.

        A tolerance above one core avoids classifying ordinary measurement
        noise and the lightweight sampling thread as component parallelism.
        This remains empirical evidence rather than a permanent guarantee
        about the Component's behavior on every input and environment.
        """
        return self.median_average_cpu_cores > 1.15

    def __str__(self) -> str:
        """Return a concise human-readable CPU profile report."""
        parallelism = "detected" if self.used_multiple_cpu_cores else "not detected"
        return (
            f"Component CPU profile: {self.component_type}\n"
            f"Available logical CPUs: {self.available_logical_cpus}\n"
            f"Measured runs: {len(self.runs)} (after {self.warmup_runs} warmup)\n"
            f"Median wall time: {self.median_wall_seconds:.4f} s\n"
            f"Median CPU time: {self.median_cpu_seconds:.4f} s\n"
            f"Median average CPU cores: {self.median_average_cpu_cores:.2f}\n"
            f"Median machine CPU use: {self.median_machine_cpu_percent:.1f}%\n"
            f"Peak interval CPU cores: {self.peak_cpu_cores:.2f}\n"
            f"Peak process-tree threads: {self.peak_thread_count}\n"
            f"Peak child processes: {self.peak_child_process_count}\n"
            f"Multi-core CPU execution: {parallelism}"
        )


@dataclass(frozen=True, slots=True)
class CPUScalingProfileEntry:
    """Associate one worker count with its repeated CPU profile.

    Attributes
    ----------
    worker_count : int
        Maximum pySPoC worker threads active for this measurement.
    profile : ComponentCPUProfile
        Repeated CPU profile collected under that worker limit.
    """

    worker_count: int
    profile: ComponentCPUProfile


@dataclass(frozen=True, slots=True)
class ComponentCPUScalingProfile:
    """Report how Component CPU use changes with pySPoC worker count.

    Attributes
    ----------
    component_type : str
        Fully qualified class name of the profiled Component.
    maximum_workers : int
        Largest worker limit included in the experiment.
    entries : tuple of CPUScalingProfileEntry
        Profiles ordered by increasing worker count.
    """

    component_type: str
    maximum_workers: int
    entries: tuple[CPUScalingProfileEntry, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the scaling results as a tabular pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per worker limit with median timing and CPU-use metrics.
        """
        return pd.DataFrame.from_records(
            [
                {
                    "Workers": entry.worker_count,
                    "Median wall time (s)": entry.profile.median_wall_seconds,
                    "Median CPU time (s)": entry.profile.median_cpu_seconds,
                    "Average CPU cores": entry.profile.median_average_cpu_cores,
                    "Machine CPU use (%)": entry.profile.median_machine_cpu_percent,
                    "Peak CPU cores": entry.profile.peak_cpu_cores,
                    "Peak threads": entry.profile.peak_thread_count,
                    "Peak child processes": entry.profile.peak_child_process_count,
                }
                for entry in self.entries
            ]
        )

    def __str__(self) -> str:
        """Return a readable worker-scaling table."""
        table = self.to_dataframe().to_string(
            index=False,
            formatters={
                "Median wall time (s)": lambda value: f"{value:.4f}",
                "Median CPU time (s)": lambda value: f"{value:.4f}",
                "Average CPU cores": lambda value: f"{value:.2f}",
                "Machine CPU use (%)": lambda value: f"{value:.1f}",
                "Peak CPU cores": lambda value: f"{value:.2f}",
            },
        )
        return f"Component CPU scaling profile: {self.component_type}\n{table}"

"""Tests for Component CPU profiling."""

import time

import numpy as np
import pytest

from pyspoc.profiling import (
    ComponentCPUProfile,
    ComponentCPUScalingProfile,
    profile_component_cpu,
    profile_component_cpu_scaling,
)
from pyspoc.settings import settings


class ExampleComponent:
    """Small structural Component used to exercise the profiling API."""

    def compute(self, data: np.ndarray) -> float:
        deadline = time.perf_counter() + 0.08
        result = 0.0

        while time.perf_counter() < deadline:
            result += float(np.sum(data * data))

        return result


def test_profile_component_cpu_returns_structured_report() -> None:
    report = profile_component_cpu(
        ExampleComponent(),
        np.arange(100, dtype=float),
        warmup_runs=0,
        measured_runs=2,
        sample_interval=0.01,
    )

    assert isinstance(report, ComponentCPUProfile)
    assert len(report.runs) == 2
    assert report.available_logical_cpus >= 1
    assert report.median_wall_seconds > 0
    assert report.median_cpu_seconds >= 0
    assert report.median_average_cpu_cores >= 0
    assert report.peak_thread_count >= 1
    assert "Component CPU profile" in str(report)


def test_profile_component_cpu_rejects_object_without_compute() -> None:
    with pytest.raises(TypeError, match="callable compute"):
        profile_component_cpu(object(), np.ones((2, 2)))


def test_profile_component_cpu_scaling_uses_powers_and_exact_maximum() -> None:
    report = profile_component_cpu_scaling(
        ExampleComponent(),
        np.arange(10, dtype=float),
        maximum_workers=6,
        warmup_runs=0,
        measured_runs=1,
        sample_interval=0.01,
    )

    assert isinstance(report, ComponentCPUScalingProfile)
    assert tuple(entry.worker_count for entry in report.entries) == (1, 2, 4, 6)

    table = report.to_dataframe()
    assert table["Workers"].tolist() == [1, 2, 4, 6]
    assert "Average CPU cores" in table.columns
    assert "Component CPU scaling profile" in str(report)


def test_profile_component_cpu_scaling_honours_configured_maximum() -> None:
    with settings.override(max_worker_threads=3):
        report = profile_component_cpu_scaling(
            ExampleComponent(),
            np.arange(10, dtype=float),
            warmup_runs=0,
            measured_runs=1,
            sample_interval=0.01,
        )

    assert tuple(entry.worker_count for entry in report.entries) == (1, 2, 3)


@pytest.mark.parametrize(
    ("argument", "value", "exception"),
    [
        ("warmup_runs", -1, ValueError),
        ("measured_runs", 0, ValueError),
        ("measured_runs", 1.5, TypeError),
        ("sample_interval", 0, ValueError),
    ],
)
def test_profile_component_cpu_validates_parameters(
    argument: str,
    value: object,
    exception: type[Exception],
) -> None:
    arguments = {argument: value}

    with pytest.raises(exception):
        profile_component_cpu(ExampleComponent(), np.ones((2, 2)), **arguments)

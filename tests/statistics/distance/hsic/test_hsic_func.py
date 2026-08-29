from __future__ import annotations

import numpy as np
import pytest

from sklearn.metrics import pairwise_kernels

import pyspoc.statistics.distance.hsic.func as hsic_func
from pyspoc.settings import settings
from pyspoc.statistics.distance.hsic.func import (
    center_kernel,
    compute_kernel,
    hsic_from_kernels,
    pairwise_hsic,
)


@pytest.mark.parametrize("biased", [False, True])
def test_precomputed_hsic_agrees_with_reference_formula(biased: bool) -> None:
    rng = np.random.default_rng(14)
    x = rng.normal(size=(16, 1))
    y = x**2 + rng.normal(scale=0.2, size=(16, 1))
    kernel_x = compute_kernel(x, "rbf")
    kernel_y = compute_kernel(y, "rbf")

    sample_count = kernel_x.shape[0]
    if biased:
        centering = np.eye(sample_count) - np.ones((sample_count, sample_count)) / sample_count
        centered_x = centering @ kernel_x @ centering
        centered_y = centering @ kernel_y @ centering
    else:
        def u_center(kernel: np.ndarray) -> np.ndarray:
            centered = (
                kernel
                - kernel.sum(axis=1)[:, None] / (sample_count - 2)
                - kernel.sum(axis=0)[None, :] / (sample_count - 2)
                + kernel.sum() / ((sample_count - 1) * (sample_count - 2))
            )
            np.fill_diagonal(centered, 0.0)
            return centered

        centered_x = u_center(kernel_x)
        centered_y = u_center(kernel_y)

    covariance = np.sum(centered_x * centered_y)
    denominator = np.sqrt(
        np.sum(centered_x * centered_x) * np.sum(centered_y * centered_y)
    )
    expected = covariance / denominator
    if biased:
        expected = np.sqrt(max(0.0, expected))
    actual = hsic_from_kernels(kernel_x, kernel_y, biased=biased)

    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-12)


@pytest.mark.parametrize(
    ("metric", "sklearn_metric", "kwargs"),
    [
        ("additive_chi2", "additive_chi2", {}),
        ("chi2", "chi2", {"gamma": 0.4}),
        ("linear", "linear", {}),
        ("poly", "poly", {"gamma": 0.4, "degree": 2, "coef0": 0.2}),
        ("polynomial", "polynomial", {"gamma": 0.4, "degree": 2}),
        ("rbf", "rbf", {"gamma": 0.4}),
        ("gaussian", "rbf", {"gamma": 0.4}),
        ("laplacian", "laplacian", {"gamma": 0.4}),
        ("sigmoid", "sigmoid", {"gamma": 0.4, "coef0": 0.2}),
        ("cosine", "cosine", {}),
    ],
)
def test_compute_kernel_delegates_supported_metrics_to_sklearn(
    metric: str,
    sklearn_metric: str,
    kwargs: dict[str, float | int],
) -> None:
    data = np.arange(1.0, 13.0).reshape(6, 2)

    expected = pairwise_kernels(data, metric=sklearn_metric, n_jobs=1, **kwargs)
    actual = compute_kernel(data, metric, **kwargs)

    np.testing.assert_allclose(actual, expected)


def test_default_rbf_uses_median_distance_bandwidth() -> None:
    data = np.array([0.0, 1.0, 4.0, 10.0])[:, None]
    median = np.median([1.0, 4.0, 10.0, 3.0, 9.0, 6.0])
    expected = pairwise_kernels(
        data,
        metric="rbf",
        n_jobs=1,
        gamma=1.0 / (2.0 * median**2),
    )

    np.testing.assert_allclose(compute_kernel(data), expected)


@pytest.mark.parametrize("metric", ["linear", "rbf"])
def test_kernel_preparation_preserves_float32_storage(metric: str) -> None:
    data = np.linspace(-1.0, 1.0, 12, dtype=np.float32)[:, None]

    kernel = compute_kernel(data, metric)
    centered = center_kernel(kernel, biased=True)

    assert kernel.dtype == np.float32
    assert centered.dtype == np.float32


def test_python_and_numba_hsic_agree() -> None:
    rng = np.random.default_rng(22)
    kernel_x = compute_kernel(rng.normal(size=18))
    kernel_y = compute_kernel(rng.normal(size=18))

    with settings.override(numba_mode="python"):
        python_result = hsic_from_kernels(kernel_x, kernel_y)
    with settings.override(numba_mode="numba"):
        numba_result = hsic_from_kernels(kernel_x, kernel_y)

    assert numba_result == pytest.approx(python_result, rel=1e-12, abs=1e-12)


def test_pairwise_hsic_is_symmetric_and_worker_stable() -> None:
    rng = np.random.default_rng(31)
    data = rng.normal(size=(14, 5))

    serial = pairwise_hsic(data, max_workers=1)
    concurrent = pairwise_hsic(data, max_workers=3)

    np.testing.assert_allclose(concurrent, serial, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(concurrent, concurrent.T)


@pytest.mark.parametrize("max_workers", [1, 3])
def test_pairwise_hsic_compares_variables_across_two_arrays(
    max_workers: int,
) -> None:
    rng = np.random.default_rng(32)
    data_a = rng.normal(size=(14, 3))
    data_b = rng.normal(size=(14, 4))

    result = pairwise_hsic(data_a, data_b, max_workers=max_workers)
    expected = np.empty((3, 4))
    for i in range(3):
        for j in range(4):
            expected[i, j] = hsic_from_kernels(
                compute_kernel(data_a[:, i]),
                compute_kernel(data_b[:, j]),
            )

    assert result.shape == (3, 4)
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-12)


def test_pairwise_hsic_requires_matching_sample_counts() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        pairwise_hsic(np.ones((8, 2)), np.ones((7, 3)))


def test_unbiased_hsic_requires_four_samples() -> None:
    with pytest.raises(ValueError, match="at least four samples"):
        pairwise_hsic(np.ones((3, 2)), biased=False)


def test_pairwise_hsic_separates_memory_anchors_from_workers(monkeypatch) -> None:
    selected: dict[str, int] = {}

    monkeypatch.setattr(hsic_func, "resolve_worker_limit", lambda _workers: 1)

    def capture_memory_limit(prepared_value_bytes, **_kwargs):
        selected["kernel_bytes"] = prepared_value_bytes
        return 3

    monkeypatch.setattr(
        hsic_func,
        "maximum_workers_permitted_by_memory",
        capture_memory_limit,
    )

    def capture_executor(
        data,
        _prepare,
        _compare,
        *,
        max_workers,
        anchor_count,
    ):
        selected["workers"] = max_workers
        selected["anchors"] = anchor_count
        return np.zeros((data.shape[1], data.shape[1]))

    monkeypatch.setattr(hsic_func, "execute_pairwise_barrier", capture_executor)

    result = pairwise_hsic(
        np.ones((5, 4), dtype=np.float32),
        biased=True,
        max_workers=1,
    )

    assert result.shape == (4, 4)
    assert selected == {"kernel_bytes": 100, "workers": 1, "anchors": 3}

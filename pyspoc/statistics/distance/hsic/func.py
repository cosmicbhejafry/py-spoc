"""Kernel construction and bounded-memory HSIC computation."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias
from numba import njit
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_kernels

from pyspoc._execution import (
    estimate_dense_array_bytes,
    execute_pairwise_barrier,
    maximum_workers_permitted_by_memory,
    resolve_worker_limit,
)
from pyspoc._numba import numba_dispatch


KernelMetric: TypeAlias = Literal[
    "additive_chi2",
    "chi2",
    "linear",
    "poly",
    "polynomial",
    "rbf",
    "gaussian",
    "laplacian",
    "sigmoid",
    "cosine",
]
PreparedKernel: TypeAlias = tuple[np.ndarray, float]

SUPPORTED_KERNELS: frozenset[str] = frozenset(
    {
        "additive_chi2",
        "chi2",
        "linear",
        "poly",
        "polynomial",
        "rbf",
        "gaussian",
        "laplacian",
        "sigmoid",
        "cosine",
    }
)


def _kernel_float_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return the supported floating dtype used for kernel storage.

    Parameters
    ----------
    dtype : numpy.dtype
        Input data type.

    Returns
    -------
    numpy.dtype
        ``float32`` for floating inputs no wider than 32 bits; otherwise
        ``float64``. Integer inputs therefore retain the previous float64
        computation path.
    """
    normalized = np.dtype(dtype)
    if np.issubdtype(normalized, np.floating) and normalized.itemsize <= 4:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def compute_kernel(
    values: np.ndarray,
    metric: KernelMetric = "rbf",
    **kernel_kwargs: Any,
) -> np.ndarray:
    """Construct a sample-by-sample kernel similarity matrix.

    Parameters
    ----------
    values : numpy.ndarray
        Observations with shape ``(n_samples,)`` or
        ``(n_samples, n_features)``.
    metric : KernelMetric, default="rbf"
        Kernel accepted by :func:`sklearn.metrics.pairwise_kernels`.
        ``"gaussian"`` is treated as an alias for ``"rbf"``.
    **kernel_kwargs : Any
        Additional parameters forwarded to scikit-learn's kernel function.

    Returns
    -------
    numpy.ndarray
        Dense kernel matrix with shape ``(n_samples, n_samples)``. Float32
        input produces float32 storage; other supported inputs use float64.

    Raises
    ------
    ValueError
        If ``metric`` is unsupported or ``values`` is not one- or
        two-dimensional.

    Notes
    -----
    RBF and Gaussian kernels use the median pairwise-distance bandwidth when
    ``gamma`` is omitted. Scikit-learn is restricted to ``n_jobs=1`` because
    pySPoC owns parallelism at the outer HSIC scheduling level; allowing each
    kernel construction to recruit workers would risk oversubscription.
    """
    if metric not in SUPPORTED_KERNELS:
        raise ValueError(f"Unsupported HSIC kernel metric: {metric!r}.")

    # scikit-learn expects samples on axis zero and features on axis one.
    # Preserve multivariate inputs while promoting a single variable to the
    # documented two-dimensional representation.
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    elif matrix.ndim != 2:
        raise ValueError("Kernel input must be one- or two-dimensional.")

    # Scikit-learn supports float32 kernels, but some metrics or input dtypes
    # may otherwise promote implicitly. Normalize before calculation and cast
    # the returned kernel to the same supported storage dtype.
    kernel_dtype = _kernel_float_dtype(matrix.dtype)
    matrix = np.asarray(matrix, dtype=kernel_dtype)

    # Work with a private mapping because the RBF bandwidth may be filled in
    # below; callers should never observe their mapping being mutated.
    kwargs = dict(kernel_kwargs)
    sklearn_metric = "rbf" if metric == "gaussian" else metric

    if sklearn_metric == "rbf" and "gamma" not in kwargs:
        # ``pdist`` stores only one triangle, halving the temporary memory
        # needed to determine the same off-diagonal median used by hyppo.
        distances = pdist(matrix, metric="euclidean")
        median_distance = float(np.median(distances)) if distances.size else 0.0
        # Identical observations have a zero median distance. Substituting one
        # avoids an infinite gamma while retaining a well-defined kernel.
        if median_distance == 0.0:
            median_distance = 1.0
        kwargs["gamma"] = 1.0 / (2.0 * median_distance**2)
        # Anchor kernels may be prepared concurrently. Release the packed
        # distance vector before allocating the dense kernel so each worker
        # does not retain both large representations at the same time.
        del distances

    kernel = pairwise_kernels(
        matrix,
        metric=sklearn_metric,
        n_jobs=1,
        **kwargs,
    )
    return np.asarray(kernel, dtype=kernel_dtype)


def _center_kernel_python(kernel: np.ndarray, biased: bool) -> np.ndarray:
    """Return a centred copy of a kernel using the Python backend.

    Parameters
    ----------
    kernel : numpy.ndarray
        Square kernel matrix.
    biased : bool
        Select ordinary double centring when ``True`` or U-centring when
        ``False``.

    Returns
    -------
    numpy.ndarray
        Newly allocated centred kernel.
    """
    centered = kernel.copy()
    return _center_kernel_inplace_python(centered, biased)


def _center_kernel_inplace_python(kernel: np.ndarray, biased: bool) -> np.ndarray:
    r"""Centre an exclusively owned kernel in place using NumPy.

    Biased centring applies ``H K H``, where
    ``H = I - 11.T / n``. Unbiased centring applies, for ``i != j``,

    ``K[i,j] - K[i,:].sum()/(n-2) - K[:,j].sum()/(n-2)``
    ``+ K.sum()/((n-1)(n-2))``

    and sets the diagonal to zero.

    Parameters
    ----------
    kernel : numpy.ndarray
        Writable square kernel owned by the caller.
    biased : bool
        Whether to use biased centring.

    Returns
    -------
    numpy.ndarray
        The same array object, modified in place.
    """
    sample_count = kernel.shape[0]

    if biased:
        # Expanding H @ K @ H gives a row adjustment, column adjustment, and
        # grand-mean correction. Applying each broadcast in place avoids an
        # additional n-by-n temporary.
        row_means = kernel.sum(axis=1) / sample_count
        column_means = kernel.sum(axis=0) / sample_count
        grand_mean = kernel.sum() / (sample_count * sample_count)
        kernel -= row_means[:, None]
        kernel -= column_means[None, :]
        kernel += grand_mean
        return kernel

    # U-centring uses n-2 for row/column corrections and forces the diagonal
    # to zero. This is the unbiased estimator used by normalized HSIC.
    row_terms = kernel.sum(axis=1) / (sample_count - 2)
    column_terms = kernel.sum(axis=0) / (sample_count - 2)
    grand_term = kernel.sum() / ((sample_count - 1) * (sample_count - 2))
    kernel -= row_terms[:, None]
    kernel -= column_terms[None, :]
    kernel += grand_term
    np.fill_diagonal(kernel, 0.0)
    return kernel


@njit(cache=True, nogil=True)
def _center_kernel_numba(kernel: np.ndarray, biased: bool) -> np.ndarray:
    """Return a centred copy of a kernel using the Numba backend.

    Parameters
    ----------
    kernel : numpy.ndarray
        Square kernel matrix.
    biased : bool
        Whether to use biased centring.

    Returns
    -------
    numpy.ndarray
        Newly allocated centred kernel.
    """
    centered = kernel.copy()
    return _center_kernel_inplace_numba(centered, biased)


@njit(cache=True, nogil=True)
def _center_kernel_inplace_numba(kernel: np.ndarray, biased: bool) -> np.ndarray:
    """Centre an exclusively owned kernel in place using Numba.

    Parameters
    ----------
    kernel : numpy.ndarray
        Writable square kernel matrix.
    biased : bool
        Whether to use biased centring.

    Returns
    -------
    numpy.ndarray
        The same array object, modified in place.
    """
    sample_count = kernel.shape[0]
    row_sums = np.empty(sample_count, dtype=kernel.dtype)
    column_sums = np.empty(sample_count, dtype=kernel.dtype)
    total = 0.0

    # Calculate all marginals before overwriting any matrix entry. Once these
    # O(n) summaries exist, the original values are no longer needed after
    # their corresponding centred entry is written.
    for i in range(sample_count):
        row_total = 0.0
        column_total = 0.0
        for j in range(sample_count):
            row_total += kernel[i, j]
            column_total += kernel[j, i]
        row_sums[i] = row_total
        column_sums[i] = column_total
        total += row_total

    if biased:
        denominator = sample_count
        grand_term = total / (sample_count * sample_count)
    else:
        denominator = sample_count - 2
        grand_term = total / ((sample_count - 1) * (sample_count - 2))

    # Explicit scalar loops compile efficiently under Numba and avoid the
    # array-sized broadcast temporaries produced by a vectorized expression.
    for i in range(sample_count):
        for j in range(sample_count):
            value = (
                kernel[i, j]
                - row_sums[i] / denominator
                - column_sums[j] / denominator
                + grand_term
            )
            kernel[i, j] = 0.0 if not biased and i == j else value

    return kernel


_center_kernel = numba_dispatch(_center_kernel_numba)(_center_kernel_python)
_center_kernel_inplace = numba_dispatch(_center_kernel_inplace_numba)(
    _center_kernel_inplace_python
)


def center_kernel(kernel: np.ndarray, biased: bool) -> np.ndarray:
    """Validate and centre a precomputed kernel.

    Parameters
    ----------
    kernel : numpy.ndarray
        Square precomputed kernel matrix.
    biased : bool
        Select biased centring when ``True`` or unbiased U-centring when
        ``False``.

    Returns
    -------
    numpy.ndarray
        Centred copy of ``kernel`` using float32 storage for floating inputs
        no wider than 32 bits and float64 storage otherwise.

    Raises
    ------
    ValueError
        If the kernel is not square, contains no samples, or has fewer than
        four samples when unbiased centring is requested.

    Notes
    -----
    The active Python or Numba backend is selected through pySPoC's global
    Numba settings. The caller's kernel is not modified.
    """
    source = np.asarray(kernel)
    matrix = np.asarray(source, dtype=_kernel_float_dtype(source.dtype))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A precomputed kernel must be a square matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("A precomputed kernel must contain at least one sample.")
    if not biased and matrix.shape[0] < 4:
        raise ValueError("Unbiased HSIC requires at least four samples.")
    return _center_kernel(matrix, biased)


def _hsic_from_centered_python(
    centered_x: np.ndarray,
    squared_norm_x: float,
    centered_y: np.ndarray,
    squared_norm_y: float,
    biased: bool,
) -> float:
    """Calculate normalized HSIC using NumPy operations.

    Parameters
    ----------
    centered_x, centered_y : numpy.ndarray
        Equally shaped centred kernel matrices.
    squared_norm_x, squared_norm_y : float
        Precomputed squared Frobenius norms of the centred kernels.
    biased : bool
        Whether to apply the biased statistic's final square root and
        non-negative boundary.

    Returns
    -------
    float
        Normalized HSIC statistic.
    """
    # The Frobenius inner product is the only matrix-sized computation still
    # required after preparation. Both self-products were cached beforehand.
    covariance = float(np.sum(centered_x * centered_y))
    # A zero centred norm represents a degenerate kernel for which normalized
    # dependence is undefined; match the reference behaviour by returning 0.
    if squared_norm_x <= 0.0 or squared_norm_y <= 0.0:
        return 0.0
    if biased and covariance <= 0.0:
        return 0.0

    statistic = covariance / np.sqrt(squared_norm_x * squared_norm_y)
    return float(np.sqrt(statistic)) if biased else float(statistic)


@njit(cache=True, nogil=True)
def _hsic_from_centered_numba(
    centered_x: np.ndarray,
    squared_norm_x: float,
    centered_y: np.ndarray,
    squared_norm_y: float,
    biased: bool,
) -> float:
    """Calculate normalized HSIC using a GIL-free Numba loop.

    Parameters
    ----------
    centered_x, centered_y : numpy.ndarray
        Equally shaped centred kernel matrices.
    squared_norm_x, squared_norm_y : float
        Precomputed squared Frobenius norms of the centred kernels.
    biased : bool
        Whether to apply the biased statistic's final square root and
        non-negative boundary.

    Returns
    -------
    float
        Normalized HSIC statistic.

    Notes
    -----
    ``nogil=True`` permits separate executor threads to evaluate disjoint
    kernel pairs concurrently.
    """
    # Accumulate directly into a scalar: allocating centered_x * centered_y
    # would add an avoidable n-by-n temporary to every active worker.
    covariance = 0.0
    for i in range(centered_x.shape[0]):
        for j in range(centered_x.shape[1]):
            covariance += centered_x[i, j] * centered_y[i, j]

    if squared_norm_x <= 0.0 or squared_norm_y <= 0.0:
        return 0.0
    if biased and covariance <= 0.0:
        return 0.0

    statistic = covariance / np.sqrt(squared_norm_x * squared_norm_y)
    return np.sqrt(statistic) if biased else statistic


hsic_from_centered = numba_dispatch(_hsic_from_centered_numba)(
    _hsic_from_centered_python
)


def _squared_norm_python(matrix: np.ndarray) -> float:
    """Calculate a squared Frobenius norm using NumPy.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix whose squared entries should be summed.

    Returns
    -------
    float
        ``sum(matrix[i, j] ** 2)`` over all entries.

    Notes
    -----
    ``einsum`` avoids materializing ``matrix * matrix``.
    """
    return float(np.einsum("ij,ij->", matrix, matrix))


@njit(cache=True, nogil=True)
def _squared_norm_numba(matrix: np.ndarray) -> float:
    """Calculate a squared Frobenius norm using a Numba loop.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix whose squared entries should be summed.

    Returns
    -------
    float
        Squared Frobenius norm.
    """
    squared_norm = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            squared_norm += matrix[i, j] * matrix[i, j]
    return squared_norm


_squared_norm = numba_dispatch(_squared_norm_numba)(_squared_norm_python)


def prepare_kernel(kernel: np.ndarray, biased: bool) -> PreparedKernel:
    """Prepare an externally owned kernel for repeated HSIC comparisons.

    Parameters
    ----------
    kernel : numpy.ndarray
        Square precomputed kernel matrix. It is not modified.
    biased : bool
        Whether to apply biased or unbiased centring.

    Returns
    -------
    PreparedKernel
        Read-only centred kernel and its squared Frobenius norm.

    Notes
    -----
    Caching the norm avoids recomputing the self-product every time this
    kernel participates in another HSIC pair.
    """
    centered = center_kernel(kernel, biased)
    centered.flags.writeable = False
    squared_norm = _squared_norm(centered)
    return centered, squared_norm


def _prepare_owned_kernel(kernel: np.ndarray, biased: bool) -> PreparedKernel:
    """Prepare an internally owned kernel without copying it.

    Parameters
    ----------
    kernel : numpy.ndarray
        Newly allocated, writable kernel owned exclusively by this pipeline.
    biased : bool
        Whether to apply biased or unbiased centring.

    Returns
    -------
    PreparedKernel
        Read-only, in-place-centred kernel and its squared Frobenius norm.

    Notes
    -----
    This private path is essential to the memory bound: retaining separate
    raw and centred matrices would add another ``n**2`` floats during every
    preparation.
    """
    centered = _center_kernel_inplace(kernel, biased)
    centered.flags.writeable = False
    return centered, _squared_norm(centered)


def compare_prepared_kernels(
    first: PreparedKernel,
    second: PreparedKernel,
    biased: bool,
) -> float:
    """Calculate normalized HSIC from two prepared kernels.

    Parameters
    ----------
    first, second : PreparedKernel
        Centred kernels paired with their cached squared Frobenius norms.
    biased : bool
        Whether to use the biased statistic's final transformation.

    Returns
    -------
    float
        Normalized HSIC statistic.
    """
    centered_x, squared_norm_x = first
    centered_y, squared_norm_y = second
    return hsic_from_centered(
        centered_x,
        squared_norm_x,
        centered_y,
        squared_norm_y,
        biased,
    )


def hsic_from_kernels(
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    *,
    biased: bool = False,
) -> float:
    """Calculate normalized HSIC from two precomputed kernels.

    Parameters
    ----------
    kernel_x, kernel_y : numpy.ndarray
        Square kernel matrices with identical shapes.
    biased : bool, default=False
        Use biased centring and the biased normalized statistic when ``True``.

    Returns
    -------
    float
        Normalized HSIC statistic between the represented variables.

    Raises
    ------
    ValueError
        If the kernel shapes differ or either kernel fails centring
        validation.

    Notes
    -----
    This convenience function prepares both kernels independently. For many
    pairwise comparisons, :func:`pairwise_hsic` is more efficient because it
    reuses each prepared anchor kernel within a barrier phase.
    """
    if np.shape(kernel_x) != np.shape(kernel_y):
        raise ValueError("HSIC kernel matrices must have matching shapes.")
    return compare_prepared_kernels(
        prepare_kernel(kernel_x, biased),
        prepare_kernel(kernel_y, biased),
        biased,
    )


def pairwise_hsic(
    data: np.ndarray,
    data_b: np.ndarray | None = None,
    *,
    metric: KernelMetric = "rbf",
    biased: bool = False,
    kernel_kwargs: Mapping[str, Any] | None = None,
    max_workers: int | None = None,
    memory_fraction: float = 0.5,
) -> np.ndarray:
    
    """Calculate pairwise HSIC with bounded-memory barrier scheduling.

    Parameters
    ----------
    data : numpy.ndarray
        First data matrix with shape ``(n_samples, n_variables_a)``. Every
        column is treated as a separate random variable.
    data_b : numpy.ndarray or None, optional
        Optional second matrix with shape ``(n_samples, n_variables_b)``.
        Supplying it computes all cross-matrix comparisons; omitting it uses
        the symmetric comparisons within ``data``.
    metric : KernelMetric, default="rbf"
        Similarity kernel used to represent each variable.
    biased : bool, default=False
        Use the biased normalized HSIC estimator when ``True``. The unbiased
        estimator requires at least four samples.
    kernel_kwargs : Mapping[str, Any] or None, optional
        Additional parameters forwarded to :func:`compute_kernel`.
    max_workers : int or None, optional
        Requested worker ceiling. ``None`` resolves through pySPoC settings
        and the logical processors available to the process.
    memory_fraction : float, default=0.5
        Maximum fraction of currently available physical memory budgeted for
        prepared kernels and the output matrix.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_variables_a, n_variables_a)`` matrix when ``data_b`` is
        omitted, otherwise a rectangular
        ``(n_variables_a, n_variables_b)`` matrix.

    Raises
    ------
    ValueError
        If either input is not two-dimensional, their sample counts differ,
        they contain no samples, or they contain fewer than four samples for
        unbiased HSIC.

    Notes
    -----
    A conventional pairwise executor can allow every worker to construct two
    unrelated ``n_samples`` by ``n_samples`` kernels simultaneously, giving a
    worst-case input footprint near ``2 * k * n_samples**2`` floats for ``k``
    workers. This function instead supplies a preparation callback and a
    comparison callback to :func:`execute_pairwise_barrier`.

    The barrier executor retains at most ``a`` prepared anchor kernels and
    streams one shared kernel through the worker group. All comparisons using
    that shared kernel complete before the next shared kernel is constructed.
    Persistent prepared-kernel storage is therefore bounded near
    ``(a + 1) * n_samples**2`` floats. Later variables may be reconstructed
    for subsequent anchor blocks; this deliberate recomputation trades CPU
    work for a predictable memory ceiling.

    The memory estimate determines ``a`` after reserving the output matrix.
    The effective worker count is ``min(requested_workers, a)``; therefore a
    serial run can still cache and reuse several anchors.
    """
    matrix = np.asarray(data)
    second_matrix = None if data_b is None else np.asarray(data_b)
    
    if matrix.ndim != 2:
        raise ValueError("HSIC input must be a two-dimensional data matrix.")
    if second_matrix is not None and second_matrix.ndim != 2:
        raise ValueError("Second HSIC input must be a two-dimensional data matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("HSIC input must contain at least one sample.")
    if second_matrix is not None and matrix.shape[0] != second_matrix.shape[0]:
        raise ValueError("HSIC inputs must contain the same number of samples.")
    if not biased and matrix.shape[0] < 4:
        raise ValueError("Unbiased HSIC requires at least four samples.")

    kwargs = dict(kernel_kwargs or {})

    # Every prepared value owns one dense n-by-n centred kernel in the
    # effective float32 or float64 computation dtype. The
    # output matrix has not yet been allocated by the executor, so reserve its
    # bytes explicitly before converting the remaining budget into workers.
    kernel_dtype = _kernel_float_dtype(
        matrix.dtype
        if second_matrix is None
        else np.result_type(matrix.dtype, second_matrix.dtype)
    )
    kernel_bytes = estimate_dense_array_bytes(
        matrix.shape[0], dtype=kernel_dtype)
    second_variable_count = (
        matrix.shape[1] if second_matrix is None else second_matrix.shape[1]
    )
    result_bytes = (
        matrix.shape[1]
        * second_variable_count
        * np.dtype(np.float64).itemsize
    )

    # Memory determines cache capacity independently of CPU concurrency. This
    # permits one worker to reuse many anchors rather than reconstructing a
    # kernel merely because no additional thread is available.
    requested_workers = resolve_worker_limit(max_workers)
    memory_anchor_count = maximum_workers_permitted_by_memory(
        kernel_bytes,
        memory_fraction=memory_fraction,
        shared_prepared_values=1,
        reserved_bytes=result_bytes,
    )
    anchor_count = min(memory_anchor_count, matrix.shape[1])
    selected_workers = min(requested_workers, anchor_count)

    def prepare(column: np.ndarray) -> PreparedKernel:
        """Build and centre the representation owned by one executor slot."""
        kernel = compute_kernel(column, metric, **kwargs)
        # The kernel was allocated immediately above and has no external
        # observers, so centring may safely reuse that allocation in place.
        return _prepare_owned_kernel(kernel, biased)

    def compare(first: PreparedKernel, second: PreparedKernel) -> float:
        """Compare two read-only prepared values inside a worker thread."""
        # Kernel construction, centring, and self-normalization have already
        # happened. The worker performs only a GIL-free Frobenius product and
        # scalar normalization, so no matrix-sized worker temporary is needed.
        return compare_prepared_kernels(first, second, biased)

    # The executor owns the synchronization policy. Within each phase every
    # worker reads the same streamed kernel and a different anchor. Waiting at
    # the phase barrier is what makes it safe to release that shared kernel
    # before the next one is prepared.
    executor_kwargs = {
        "max_workers": selected_workers,
        "anchor_count": anchor_count,
    }
    if second_matrix is None:
        return execute_pairwise_barrier(
            matrix,
            prepare,
            compare,
            **executor_kwargs,
        )
    return execute_pairwise_barrier(
        matrix,
        prepare,
        compare,
        second_matrix,
        **executor_kwargs,
    )


__all__ = [
    "KernelMetric",
    "SUPPORTED_KERNELS",
    "center_kernel",
    "compare_prepared_kernels",
    "compute_kernel",
    "hsic_from_centered",
    "hsic_from_kernels",
    "pairwise_hsic",
    "prepare_kernel",
]

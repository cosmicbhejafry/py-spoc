"""HSIC statistic with bounded-memory internal pairwise execution."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from typing import Any, Literal

from pyspoc.component.statistics.base import Statistic
from pyspoc.settings import settings
from pyspoc._tag import Tag, Type, Signature
from .func import SUPPORTED_KERNELS, pairwise_hsic


class HilbertSchmidtIndependenceCriterion(Statistic):
    r"""Calculate pairwise normalized Hilbert-Schmidt independence.

    Parameters
    ----------
    biased : bool, default=False
        Select the biased normalized HSIC estimator. The unbiased estimator
        requires at least four samples along the non-comparison axis.
    metric : KernelMetric, default="rbf"
        Kernel used to construct sample similarity matrices. Supported names
        are listed in :data:`pyspoc.statistics.distance.hsic.SUPPORTED_KERNELS`.
    downsample : bool, default=True
        If the data matrix provided is 64-bit floating point type, downsamples
        to 32-bit floating point type. This reduces memory footprint and allows
        for higher multi-core processing thresholds on smaller workstations.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.

        ### Additional Kernel Key-Word Parameters

    kernel_kwargs : Mapping[str, Any] or None, optional
        Additional parameters forwarded to scikit-learn's kernel function.

    Notes
    -----
    Let :math:`K^x` and :math:`K^y` be the :math:`n \times n` kernel matrices
    for two variables. For the biased estimator, define

    .. math::

        H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^{\mathsf T}, \qquad
        C^x = H K^x H, \qquad C^y = H K^y H.

    Equivalently, each centred entry is

    .. math::

        C^x_{ij} = K^x_{ij}
        - \frac{1}{n}\sum_s K^x_{is}
        - \frac{1}{n}\sum_s K^x_{sj}
        + \frac{1}{n^2}\sum_{s,t}K^x_{st},

    with the analogous expression for :math:`C^y`. For the unbiased
    estimator, the diagonal is set to zero and, for :math:`i \ne j`,

    .. math::

        C^x_{ij} = K^x_{ij}
        - \frac{1}{n-2}\sum_s K^x_{is}
        - \frac{1}{n-2}\sum_s K^x_{sj}
        + \frac{1}{(n-1)(n-2)}\sum_{s,t}K^x_{st}.

    The implementation returns normalized HSIC. Using the Frobenius inner
    product :math:`\langle A,B\rangle_F = \sum_{i,j}A_{ij}B_{ij}`, define

    .. math::

        R_{\mathrm{HSIC}}(X,Y) =
        \frac{\langle C^x,C^y\rangle_F}
        {\sqrt{\langle C^x,C^x\rangle_F
                     \langle C^y,C^y\rangle_F}}.

    The unbiased statistic returns :math:`R_{\mathrm{HSIC}}`; the biased
    statistic returns :math:`\sqrt{R_{\mathrm{HSIC}}}` after applying the
    reference implementation's non-negative boundary. Degenerate centred
    kernels with zero norm return zero.

    HSIC centring, normalization, and pairwise execution are implemented
    internally by pySPoC rather than delegated to an external HSIC library.
    Scikit-learn is used only to construct the selected kernel matrices.

    Pairwise computation uses barrier-synchronized thread phases. At most
    :math:`a` memory-permitted centred anchor kernels and one shared centred
    kernel are retained persistently. All workers finish comparing the current
    shared kernel before it is released and the next is constructed. The
    anchor kernels are prepared concurrently, up to the selected worker limit;
    subsequently streamed kernels are prepared one at a time. The
    prepared-kernel footprint is consequently bounded near
    :math:`(a+1)n^2` floating-point values. The worker count is independently
    bounded by :math:`a`, allowing even one worker to reuse several cached
    kernels. The
    result matrix, small per-kernel summaries, and temporary packed distances
    used to select a default RBF bandwidth are additional allocations.

    At construction, the component snapshots
    ``settings.current.max_worker_threads`` and
    ``settings.current.max_memory_fraction``. The runtime worker count is the
    smaller of the configured processor limit and the limit estimated to fit
    within that memory fraction.

    This class calculates statistics only. It does not perform a permutation
    test, calculate p-values, or apply hyppo's chi-squared null approximation.
    """

    _long_name = "Hilbert-Schmidt Independence Criterion"
    _identifier = "hsic"
    _tags = (Tag.ASSOCIATION, Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.SYMMETRICAL)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.NONLINEARITY,)

    def __init__(
            self,
            biased: bool = False,
            metric: Literal[
                "additive_chi2",
                "chi2",
                "linear",
                "poly",
                "polynomial",
                "rbf",
                "gaussian",
                "laplacian",
                "sigmoid",
                "cosine"] = "rbf",
            downsample = True,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            *,
            kernel_kwargs: Mapping[str, Any] | None = {}):
                        
        self._biased = biased
        self._metric = metric
        self._downsample = downsample
        self._kernel_kwargs = dict(kernel_kwargs or {})

        if biased:
            self._identifier += ".biased"

        if metric in ("gaussian", "rbf"):
            self._tags = self._get_modified_tags(
                self._tags,
                add_tags=Tag.GAUSSIAN)

        if metric == "linear":
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.NONLINEAR)

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            thread_safety="safe",
            internal_parallelism="parallel")


    @staticmethod
    def get_available_kernels() -> tuple[str, ...]:
        return tuple(SUPPORTED_KERNELS)

    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Calculate the pairwise normalized HSIC matrix.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional data with observations on axis zero and variables
            on axis one.

        Returns
        -------
        numpy.ndarray
            Symmetric square matrix over the axis selected by ``dim``.

        Raises
        ------
        ValueError
            If the selected orientation is invalid for HSIC or provides fewer
            than four samples to the unbiased estimator.
        """

        mtype = data.dtype

        # Downsample data copy if required.
        if np.issubdtype(mtype, np.floating):
            mtypebits = np.dtype(mtype).itemsize * 8

            if mtypebits > 32 and self._downsample:
                data = data.astype(np.float32)
        
        error_policy = settings.current.pairwise_error_policy
        throw_first_error = error_policy == "first"

        result, exceptions = pairwise_hsic(
            data,
            metric=self._metric,
            biased=self._biased,
            kernel_kwargs=self._kernel_kwargs,
            throw_first_error=throw_first_error)

        if exceptions and error_policy == "all":
            raise ExceptionGroup(
                "One or more exceptions occurred during "
                f"{self.__class__.__qualname__}.compute()",
                exceptions)

        return result

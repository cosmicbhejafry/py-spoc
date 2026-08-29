"""HSIC statistic with bounded-memory internal pairwise execution."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from typing import Any, Literal

from pyspoc.statistics.base import Statistic
from .func import KernelMetric, pairwise_hsic


class HilbertSchmidtIndependenceCriterion(Statistic):
    r"""Calculate pairwise normalized Hilbert-Schmidt independence.

    Parameters
    ----------
    dim : {"n", "p"}, default="p"
        Axis whose elements are compared. ``"p"`` compares variables
        (columns) and returns a ``p`` by ``p`` matrix. ``"n"`` compares
        observations by transposing the input first.
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

    _name = "Hilbert-Schmidt Independence Criterion"
    _identifier = "hsic"
    _labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(
        self,
        dim: Literal["n", "p"] = "p",
        biased: bool = False,
        metric: KernelMetric = "rbf",
        downsample = True,
        *,
        kernel_kwargs: Mapping[str, Any] | None = {},
    ) -> None:
        
        """Initialize an HSIC statistic from explicit and global settings."""
        self._dim = dim
        self._biased = biased
        self._metric = metric
        self._downsample = downsample
        self._kernel_kwargs = dict(kernel_kwargs or {})

        if biased:
            self._identifier += ".biased"

        super().__init__(thread_safety="safe", internal_parallelism="parallel")

    @property
    def name(self) -> str:
        """Return the human-readable statistic name.

        Returns
        -------
        str
            ``"Hilbert-Schmidt Independence Criterion"``.
        """
        return self._name

    @property
    def identifier(self) -> str:
        """Return the stable configuration identifier.

        Returns
        -------
        str
            ``"hsic"`` for the unbiased estimator or ``"hsic.biased"`` for
            the biased estimator.
        """
        return self._identifier

    @property
    def labels(self) -> list[str]:
        """Return descriptive labels for filtering this statistic.

        Returns
        -------
        list[str]
            Labels describing the statistic's sign, ordering, linearity, and
            directionality properties.
        """
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray:
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

        # Typically idempotent as data is always an array.
        matrix = np.asarray(data)

        if self._dim == "n":
            matrix = matrix.T

        mtype = matrix.dtype

        # Downsample data copy if required.
        if np.issubdtype(mtype, np.floating):
            mtypebits = np.dtype(mtype).itemsize * 8

            if mtypebits > 32 and self._downsample:
                matrix = matrix.astype(np.float32)
        
        return pairwise_hsic(
            matrix,
            metric=self._metric,
            biased=self._biased,
            kernel_kwargs=self._kernel_kwargs
        )

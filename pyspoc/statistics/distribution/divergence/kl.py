import numpy as np

from typing import Any, Literal
from sklearn.neighbors import KernelDensity

from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic


class KullbackLeiblerDivergence(PairwiseStatistic):


    def __init__(
            self,
            kde_kernel: Literal[
                'gaussian',
                 'tophat',
                 'epanechnikov',
                 'exponential',
                 'linear',
                 'cosine'] = "gaussian",
            kde_bw: float | Literal["scott", "silverman"] = "scott",
            kl_resolution: int = 1000,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=False,
            symmetry_type=None,
            internal_parallelism="serial")

        self._kde_kernel = kde_kernel
        self._kde_bw = kde_bw
        self._kl_resolution = kl_resolution
        self._support = None
        self._support_grid = None
        self._log_densities = {}


    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:

        self._log_densities.clear()
        # Fit each column once before constructing the common grid. In
        # particular, Scott/Silverman are rules, not numeric bandwidths; sklearn
        # exposes their resolved values on the fitted estimator as bandwidth_.
        kdes = [
            KernelDensity(kernel=self._kde_kernel, bandwidth=self._kde_bw).fit(
                data[:, i:i + 1])
            for i in range(data.shape[1])
        ]
        max_bandwidth = max(kde.bandwidth_ for kde in kdes)

        # Compact kernels vanish outside one bandwidth. Infinite-support kernels
        # need wider padding: Gaussian tails past 8 bandwidths and exponential
        # tails past 32 bandwidths have very small remaining probability mass.
        # This is still a finite-grid approximation, not exact tail integration.
        tail_width = (
            8.0 if self._kde_kernel == "gaussian"
            else 32.0 if self._kde_kernel == "exponential"
            else 1.0
        )
        padding = tail_width * max_bandwidth
        tot_min, tot_max = data.min(), data.max()
        self._support = (tot_min - padding, tot_max + padding)
        self._support_grid = np.linspace(
            self._support[0], self._support[1], self._kl_resolution)

        # Only evaluated arrays are needed during pair comparisons. Preparing
        # them here avoids refitting KDEs and keeps writes out of worker tasks.
        kde_grid = self._support_grid.reshape(-1, 1)
        self._log_densities.update({
            i: kde.score_samples(kde_grid) for i, kde in enumerate(kdes)
        })
        del kdes
        return super()._compute(data)
    

    def _pairwise_compute_indexed(
            self,
            x: np.ndarray[tuple[Any, ...], np.dtype[Any]],
            y: np.ndarray[tuple[Any, ...], np.dtype[Any]],
            index: tuple[int, int]
    ) -> np.floating:

        i, j = index

        if self._support_grid is None:
            raise RuntimeError("Support grid is not initialized. "
                "Call _compute() before computing pairwise KL divergence.")

        log_p = self._log_densities[i]
        log_q = self._log_densities[j]
        if np.isnan(log_p).any() or np.isnan(log_q).any():
            raise ValueError("KDE evaluation produced NaN log densities.")

        positive_p = np.isfinite(log_p)
        # A zero target density wherever the source is positive makes KL
        # infinite. Keep this distinct from the zero-source convention below.
        if np.any(positive_p & np.isneginf(log_q)):
            return np.float64(np.inf)

        # For p=0 the contribution is zero, including where q=0 as well. Mask
        # BEFORE subtracting logs so -inf - -inf and 0 * inf are never evaluated.
        integrand = np.zeros_like(log_p)
        integrand[positive_p] = np.exp(log_p[positive_p]) * (
            log_p[positive_p] - log_q[positive_p])
        estimate = np.trapezoid(integrand, x=self._support_grid)

        # Approximate continuous KL divergence by quadrature over the KDEs.
        return estimate.mean()

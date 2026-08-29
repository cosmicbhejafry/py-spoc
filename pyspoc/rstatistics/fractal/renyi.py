import numpy as np

from numba import njit
from typing import Literal

from pyspoc._numba import fallback_loader as fload
from pyspoc._argchecking import check_float
from .base import FractalMeasureBase
from . import _funcs_numba as fnb

# ---------------------------------------------------------------------------
# Implementation note
# ---------------------------------------------------------------------------
#
# `GeneralizedEntropy` is implemented as a thin, stateful interface over
# the stateless computational kernels defined in the `func` module.
#
# All core numerical algorithms have been moved into standalone functions in
# `func`. This separation follows the common scientific-computing pattern of
# isolating pure computational kernels from the object-oriented API layer.
#
# The benefits of this design are:
#
#   * Stateless kernels are easier to unit test in isolation.
#   * Computational code can be accelerated independently (e.g. via Numba).
#   * The class remains responsible only for configuration and orchestration.
#   * Algorithmic logic is not tightly coupled to object state.
#
# In practice this class is responsible for:
#
#   * storing configuration parameters (e.g. q, tolerances)
#   * validating inputs and managing object state
#   * dispatching calls to the appropriate function in `func`
#
# The `func` module therefore contains the functional core of the algorithm,
# while this class provides the public stateful interface.
# ---------------------------------------------------------------------------


class RenyiEntropy(FractalMeasureBase):

    """
    Generalized fractal dimension estimator using Rényi entropy over box-counting
    scales, with piecewise regression based elbow trimming of the scaling curve.

    Given a static point dataset of shape ``(n, p)``, the method discretizes
    the state space into non-overlapping boxes across a range of scales and
    computes the entropy of the induced natural density at each scale. A scaling
    region is then identified and used to estimate the generalized fractal
    dimension.


    Definition
    ----------------------------
    For a box covering of size :math:`\\varepsilon`, let :math:`p_i` denote the
    proportion of points in the :math:`i`-th occupied box, with
    :math:`i = 1, \\ldots, M`, where :math:`M` is the number of occupied boxes.
    The Rényi entropy of order :math:`q` is defined as

    .. math::

        H_q(\\varepsilon) =
        \\frac{1}{1-q}
        \\log \\left( \\sum_{i=1}^{M} p_i^q \\right).

    With :math:`q = 0`, this reduces to the box-counting or capacity dimension,
    and in the limit :math:`q \\to 1`, this reduces to the Shannon entropy,

    .. math::

        H_1(\\varepsilon) = -\\sum_{i=1}^{M} p_i \\log p_i.

    Using the Rényi entropy as its basis, the generalized fractal dimension is
    defined by

    .. math::

        \\Delta_q^{(H)} =
        \\lim_{N \\to \\infty}
        \\lim_{\\varepsilon \\to 0}
        \\left(
            \\frac{-H_q(\\varepsilon)}{\\log \\varepsilon}
        \\right),

    where :math:`N` is the number of sampled points and :math:`\\varepsilon` is
    the box size. In practice, this quantity is estimated from the scaling law

    .. math::

        H_q(\\varepsilon) \\sim -\\Delta_q^{(H)} \\log \\varepsilon,

    so that the fractal dimension corresponds to the slope of the approximately
    linear scaling region in a plot of :math:`H_q(\\varepsilon)` against
    :math:`-\\log(\\varepsilon)`.


    Parameters
    ----------
    slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
        Method used to select the final scaling-curve slope. ``"ols"`` uses
        ordinary least squares, ``"deshmukh"`` prefers the slope-ensemble
        estimate, and ``"hybrid"`` selects between them using ``r2_thresh``.

    q : float, default=0, requires: >=0
        Rényi entropy order. Special cases include :math:`q = 0` for box-counting
        entropy and :math:`q \\to 1` for Shannon entropy.

    r2_thresh : float, default=0.9, requires: >=0, <=1
        Minimum :math:`R^2` score required from an ordinary least squares (OLS) fit to use
        its slope as the fractal dimension estimate. The process is attempted twice.
        See __Algorithm__ for more information on the parameter's use in the algorithm.

    monotonic_tol : float, default=0.1, requires: >=0, <=1
        Proportional tolerance applied to monotonic detection of the smoothed scaling
        curve for elbow detection and removal. The scaling curve is considered monotonic
        if and only if the proportion of adjacent points along the curve that violate
        monotonicity is less than or equal to this value. A value of `1`
        disables the monotonicity check entirely.

    deshmukh_reg_proportion : float, default=0.25, requires: >0, <=0.5
        Minimum relative subinterval length used when constructing the ensemble of
        linear regressions over the retained scaling region.

    minimum_scaling_region : float, default=0.1, requires: >0, <=1
        Minimum proportion of both the original scaling-curve points and its
        log-scale span that an elbow-trimmed region must retain.

    minimum_scaling_points : int or None, default=None, requires: >=20
        Absolute minimum number of points retained after elbow trimming. If
        ``None``, this is 40% of the data-dependent scale length, with a
        minimum of 20 points.

    use_adaptive_scaling : bool, default=True
        Whether to iteratively adapt the initial scale range to exclude
        uninformative saturated regions.

    scale_method : {"datseries", "log-10"}, default="datseries"
        Algorithm used to generate the initial box scales.

    scale_length : int or None, default=None, requires: >=50
        Number of intervals in the generated scale range. If ``None``, this is
        resolved from dimensionality :math:`p` as
        ``max(50, ceil(100 * log(p)))``. Scale generators may return one more
        scale value than the resolved interval count.

    scale_adaption_iters : int, default=20, requires: >=1
        Maximum number of scale-range expansion iterations when adaptive scaling
        is enabled.


    Returns
    -------
    float
        Estimated generalized fractal dimension.


    Notes
    -----
    This implementation uses a box-counting approximation to the natural density
    and computes :math:`H_q(\\varepsilon)` over a default set of initial scales.
    The final estimate depends on the quality of the observed scaling region and
    may be sensitive to sample size, scale selection, and the elbow-detection
    tolerances.


    Algorithm
    ---------------
    1. Generate the initial scaling region using `adaptive`, `datseries`, or
        `log base-10` methods.
    2. For each scale:
            2a. Partition the data space into hypercubes of the given scale.
            2b. Compute the Rényi entropy of the data over the hypercubes.
    3. Collect the negative log of the scaling region and the respective Rényi
        entropies as the scaling curve.
    4. Apply OLS to the entire scaling curve.
    5. Attempt elbow detection and removal. Accept the trimmed region only if
        it retains the required number and proportion of points and the required
        proportion of the original log-scale span, then refit OLS.
    6. If:
        OLS :math:`R^2` `>= r2_thresh` requirement and `slope_estimation_method == 'hybrid'`:
            Return the OLS slope as the fractal dimension estimate.
        `slope_estimation_method == 'ols``:
            Return the OLS slope as the fractal dimension estimate.
        
    7. Compute the Deshmukh slope estimate.
    8. If Deshmukh slope estimate is valid:
            Return Deshmukh slope estimate.
    9. Otherwise, return the last computed OLS slope estimate.


    Elbow detection
    ---------------
    A typical feature of the scaling curve is a sigmoidal shape due to entropy
    saturation at the outer scales. With the true scaling region commonly found at
    the centre of the curve, elbow-detection may be applied to reduce the influence
    of unstable outer-scale behaviour.

    The method is applied for detecting upper and/or lower elbows, and therefore also
    accounts for simpler curves with only a single elbow.


    Deshmukh Slope Estimation
    ---------------
    The method is defined in [3], which involves fitting an ensemble of linear
    regressions computed over subintervals of the retained scaling curve. These slope
    estimates are then averaged to produce the final fractal dimension estimate.


    References
    ----------
    .. [1] Cui, T., & Wang, T. (2024). Exact box-counting and temporal sampling algorithms for
        fractal dimension estimation with applications to animal behavior analysis.
        Results in Engineering.
        (Some code has been integrated into this class and can be found at
        https://github.com/wanglab-georgetown/fractal.)

    .. [2] Datseris et al. (2023). Estimating fractal dimensions: A comparative review
        and open source implementations. Chaos.

    .. [3] V. Deshmukh, E. Bradley, J. Garland, and J. D. Meiss (2021). Toward automated
        extraction and characterization of scaling regions in dynamical systems. Chaos.

    """

    _name = "Rényi Generalized Entropy"
    _identifier = "rge"
    _labels = ["fractal"]
    _SHANNON_ENTROPY_TOL = 1e-6

    def __init__(self,
                 slope_estimation_method: Literal["hybrid", "ols", "deshmukh"] = "hybrid",
                 q: float = 0,
                 r2_thresh: float = 0.9,
                 monotonic_tol: float = 0.1,
                 deshmukh_reg_proportion: float = 0.25,
                 minimum_scaling_region: float = 0.1,
                 minimum_scaling_points: int | None = None,
                 use_adaptive_scaling: bool = True,
                 scale_method: Literal["datseries", "log-10"] = "datseries",
                 scale_length: int | None = None,
                 scale_adaption_iters: int = 20,
                 **kwargs):
        
        """Initialize a Rényi generalized-dimension statistic.

        Parameters
        ----------
        slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
            Strategy used to select the final scaling-curve slope.
        q : float, default=0, requires: >=0
            Rényi entropy order. ``0`` gives the box-counting entropy and values
            sufficiently close to ``1`` use the Shannon-entropy limit.
        r2_thresh : float, default=0.9, requires: >=0, <=1
            Minimum OLS :math:`R^2` accepted by the hybrid strategy.
        monotonic_tol : float, default=0.1, requires: >=0, <=1
            Maximum proportion of adjacent monotonicity violations allowed by
            elbow detection.
        deshmukh_reg_proportion : float, default=0.25, requires: >0, <=0.5
            Minimum relative length of scaling-curve subintervals included in
            the Deshmukh slope ensemble.
        minimum_scaling_region : float, default=0.1, requires: >0, <=1
            Minimum proportion of the original point count and log-scale span
            retained by an accepted elbow-trimmed region.
        minimum_scaling_points : int or None, default=None, requires: >=20
            Absolute minimum number of retained scaling-curve points. If
            omitted, this is 40% of the resolved scale length, with a minimum
            of 20 points.
        use_adaptive_scaling : bool, default=True
            Whether to iteratively adapt the generated scale range.
        scale_method : {"datseries", "log-10"}, default="datseries"
            Method used to generate the initial box scales.
        scale_length : int or None, default=None, requires: >=50
            Number of intervals in the generated scale range. If omitted, this
            is ``max(50, ceil(100 * log(p)))`` for dimensionality ``p``.
        scale_adaption_iters : int, default=20, requires: >=1
            Maximum adaptive-scale expansion iterations.

        Raises
        ------
        TypeError
            If a checked numeric argument has an incompatible type.
        ValueError
            If a checked numeric argument lies outside its documented bounds.
        """
                
        self._q = check_float(q, minimum=0, arg_name="q")
        super().__init__(slope_estimation_method=slope_estimation_method,
                         r2_thresh=r2_thresh,
                         monotonic_tol=monotonic_tol,
                         deshmukh_reg_proportion=deshmukh_reg_proportion,
                         minimum_scaling_region=minimum_scaling_region,
                         minimum_scaling_points=minimum_scaling_points,
                         use_adaptive_scaling=use_adaptive_scaling,
                         scale_method=scale_method,
                         scale_length=scale_length,
                         scale_adaption_iters=scale_adaption_iters,
                         **kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def labels(self) -> list[str]:
        return self._labels
    
    @property
    def q(self) -> float:
        return self._q
    
    def _compute_density_estimate(
            self,
            data: np.ndarray,
            scales: np.ndarray) -> np.ndarray:

        """
        Compute Renyi entropy according to the formula

        :math:`H_q = \\frac{1}{1 - q} log (\\sum_{i=1}^M p_i^q)`

        for :math:`M` non-overlapping boxes covering the set of points and
        :math:`p_i` are the proportion of points within that box.

        :math:`q` is specified as an instance property, defined on instantiation.

        Parameters
        ----------
        data : np.ndarray
            Array of points (n, p).
            
        scales : np.ndarray
            Array of box scales to compute the density over.
            
        Returns
        ----------
                
        np.ndarray
            Array of density estimates at different scales.
        """

        return _get_renyi_entropy(
            self.q, data, scales, self._SHANNON_ENTROPY_TOL)


@njit
def _get_renyi_entropy_numba(
        q: float,
        data: np.ndarray,
        scales: np.ndarray,
        shannon_entropy_tol: float = 1e-6) -> np.ndarray:
    
    """
    Numba implementation of the Rényi entropy according to the formula

    :math:`H_q = \\frac{1}{1 - q} log (\\sum_{i=1}^M p_i^q)`

    for :math:`M` non-overlapping boxes covering the set of points and
    :math:`p_i` are the proportion of points within that box.
    
    Parameters
    ----------
    q : float
        Rényi entropy order. Special cases include :math:`q = 0` for box-counting
        entropy and :math:`q \\to 1` for Shannon entropy.
            
    data : np.ndarray
        Array of points (n, p).
        
    scales : np.ndarray
        Array of box scales to compute the density over.

    shannon_entropy_tol : float, default = 1e-6
        Defines tolerance level for judging q = 0.
        Specifically, if abs(q) < shannon_entropy_tol, take q = 0.
        With q = 0, Rényi reverts to box-counting measure.
        
    Returns
    ----------
            
    np.ndarray
        Array of density estimates at different scales.
    """
    
    # Initialising.
    n_scales = scales.shape[0]
    H = np.empty(shape=n_scales, dtype=np.float64)
    n_observations = data.shape[0]
    scales_are_nonincreasing = np.all(scales[:-1] >= scales[1:])
    scales_are_nondecreasing = np.all(scales[:-1] <= scales[1:])

    for k in range(n_scales):
        s = scales[k]

        # Get boxes containing each point (already sorted thanks to sorted data).
        box_ids = np.floor(data / s).astype(np.int32)
        
        
        hashes = fnb.get_row_hashes_numba(box_ids)


        cnts = fnb.get_box_tallies_numba(box_ids, hashes)


        cnts = cnts[cnts > 0]
        n_occupied_boxes = cnts.shape[0]

        # At the fine-scale limit, every observation has its own box and all
        # probabilities are 1 / n. Finer scales therefore have entropy log(n),
        # regardless of q, and need no further integer box identifiers.
        if n_occupied_boxes == n_observations:
            saturated_entropy = np.log(n_observations)
            H[k] = saturated_entropy

            if scales_are_nonincreasing:
                H[k:] = saturated_entropy
                break

            continue

        # At the coarse-scale limit, the sole occupied box has probability 1,
        # so its entropy and that of every remaining coarser scale are zero.
        if n_occupied_boxes == 1:
            H[k] = 0.0

            if scales_are_nondecreasing:
                H[k:] = 0.0
                break

            continue


        if q == 0:
            H[k] = np.log(n_occupied_boxes)
            continue

        probs = cnts / cnts.sum()

        if abs(q-1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H


@fload.numba_dispatch(_get_renyi_entropy_numba)
def _get_renyi_entropy(
        q: float,
        data : np.ndarray,
        scales: np.ndarray,
        shannon_entropy_tol: float = 1e-6) -> np.ndarray:
    
    """
    Python implementation of the Rényi entropy.

    Rényi entropy is computed according to the formula:

    :math:`H_q = \\frac{1}{1 - q} log (\\sum_{i=1}^M p_i^q)`

    for :math:`M` non-overlapping boxes covering the set of points and
    :math:`p_i` are the proportion of points within that box.

    Parameters
    ----------
    q : float
        Rényi entropy order. Special cases include :math:`q = 0` for box-counting
        entropy and :math:`q \\to 1` for Shannon entropy.
            
    data : np.ndarray
        Array of points (n, p).
        
    scales : np.ndarray
        Array of box scales to compute the density over.

    shannon_entropy_tol : float, default = 1e-6
        Defines tolerance level for judging q = 0.
        Specifically, if abs(q) < shannon_entropy_tol, take q = 0.
        With q = 0, Rényi reverts to box-counting measure.
        
    Returns
    ----------
            
    np.ndarray
        Array of density estimates at different scales.
    """

    n_scales = scales.shape[0]
    H = np.empty(shape=n_scales)
    n_observations = data.shape[0]
    scales_are_nonincreasing = np.all(scales[:-1] >= scales[1:])
    scales_are_nondecreasing = np.all(scales[:-1] <= scales[1:])

    for k in range(H.shape[0]):
        s = scales[k]
        box_ids = np.floor(data / s).astype(np.int32)
        _, cnts = np.unique(box_ids, axis=0, return_counts=True)
        n_occupied_boxes = cnts.shape[0]

        # Avoid constructing still-finer box identifiers once the entropy has
        # saturated at log(n), which also prevents integer overflow at scales
        # too small to provide any additional information.
        if n_occupied_boxes == n_observations:
            saturated_entropy = np.log(n_observations)
            H[k] = saturated_entropy

            if scales_are_nonincreasing:
                H[k:] = saturated_entropy
                break

            continue

        # Likewise, all scales coarser than a one-box partition have zero
        # entropy and can be filled without further box-count calculations.
        if n_occupied_boxes == 1:
            H[k] = 0.0

            if scales_are_nondecreasing:
                H[k:] = 0.0
                break

            continue

        if q == 0:
            H[k] = np.log(n_occupied_boxes)
            continue

        probs = cnts / cnts.sum()

        if abs(q - 1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H
import numpy as np

from numba import njit
from typing import Literal, Iterable

from pyspoc.rstatistics.fractal.base import FractalMeasureBase

from . import _funcs_py as f
from . import _funcs_numba as fnumba

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


class CorrelationSum(FractalMeasureBase):

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
    q : float, default=0
        Rényi entropy order. Special cases include :math:`q = 0` for box-counting
        entropy and :math:`q \\to 1` for Shannon entropy.

    adj_r2_tol : float, default=0.01, requires: >0, <=1
        Minimum proportional improvement in adjusted :math:`R^2` required to
        accept an additional linear segment during piecewise regression model
        selection.

    elbow_tol : float, default=0.15, requires: >0, <=1
        Proportional tolerance used to identify outer segments as elbows. If the
        slope of an outer segment is below this tolerance multiplied by the average slope
        of the inner segments, that outer segment is removed.

    deshmukh_reg_proportion : float, default=0.25, requires: >0, <=1
        Minimum relative subinterval length used when constructing the ensemble of
        linear regressions over the retained scaling region.


    Returns
    -------
    np.ndarray
        Estimated generalized fractal dimension. For static input data this is
        returned as a scalar-like NumPy value.


    Notes
    -----
    This implementation uses a box-counting approximation to the natural density
    and computes :math:`H_q(\\varepsilon)` over a default set of initial scales.
    The final estimate depends on the quality of the observed scaling region and
    may be sensitive to sample size, scale selection, and the elbow-detection
    tolerances.


    Elbow detection
    ---------------
    To reduce the influence of unstable outer-scale behaviour, this class applies
    a piecewise linear regression model selection procedure to the estimated
    entropy scaling curve. The procedure begins with a standard linear regression
    model and then incrementally increases the number of linear segments. New
    segments are accepted only while the adjusted :math:`R^2` score improves by at
    least ``adj_r2_tol`` relative to the previous fit.

    Once the selected piecewise linear model has been obtained, the slopes of the
    two outermost segments are compared with the average slope of the inner
    segments. An outer segment is treated as an elbow and removed if its slope is
    less than ``elbow_tol`` times the mean slope of the inner region. This
    trimming step is intended to remove edge effects caused by finite sampling,
    discretization, or departures from the main scaling regime.

    After trimming, the final fractal dimension estimate is obtained based on the method
    defined in [3], which involves fitting an ensemble of linear regressions computed
    over subintervals of the retained scaling region. These slope estimates are then
    aggregated using a mode-weighted averaging procedure.


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

    _name = "Correlation Sum"
    _identifier = "corrsum"
    _labels = ["fractal"]

    def __init__(self,
                 q: float = 0,
                 adj_r2_thresh: float = 0.005,
                 elbow_thresh: float = 0.25,
                 deshmukh_reg_proportion: float = 0.25,
                 scale_generator: Literal["datseries", "base-10"] = "datseries",
                 scale_length: int = 50,
                 debug_numba: Iterable[Literal["ignore", "warn", "raise", "bounds"]] = "ignore"):
                
        self._q = q
        super().__init__(r2_thresh=adj_r2_thresh,
                         monotonic_tol=elbow_thresh,
                         deshmukh_reg_proportion=deshmukh_reg_proportion,
                         scale_method=scale_generator,
                         scale_length=scale_length,
                         debug_numba=debug_numba)

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
            self.q, data, scales, self._SHANNON_ENTROPY_TOL, debug_numba = self._debug_numba)
    

def _assign_boxes(data: np.ndarray):
    data - data.min(axis=1)





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
    
    n_scales = scales.shape[0]
    n = data.shape[0]
    p = data.shape[1]
    H = np.empty(shape=n_scales, dtype=np.float64)

    for k in range(n_scales):
        #cnts = np.zeros((n, p+1), dtype=np.int32)
        cnts = np.zeros(n+1, dtype=np.int32)
        s = scales[k]

        # Get boxes containing each point.
        box_ids = np.floor(data / s).astype(np.int32)

        # Sort boxes.
        box_ids = fnumba._sort_data(box_ids)

        # Initialise previous box array.
        prev_box_id = np.zeros(shape=p).astype(np.int32)
        j = 0

        # Loop through all points.
        for i in range(n):

            # Get the box containing the point.
            box_id = box_ids[i]
            
            # If a different box than before, move on
            # to the next box.
            if np.any(prev_box_id != box_id):
                prev_box_id[:] = box_id
                j += 1
            
            # Add a point count to the current box.
            cnts[j] += 1
                
        cnts = cnts[cnts > 0]

        if q == 0:
            H[k] = np.log(cnts.shape[0])
            continue

        probs = cnts / cnts.sum()

        if abs(q-1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H


# Bounds-check enabled version for testing and debugging.
_get_renyi_entropy_numba_bc = njit(boundscheck=True)(_get_renyi_entropy_numba)

# Fast version closer to C-perform for production use.
_get_renyi_entropy_numba_fast = njit(boundscheck=False)(_get_renyi_entropy_numba)


def _get_renyi_entropy_py(
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

    points = np.expand_dims(data, axis=2)
    results = (points / scales).astype(int)
    H : np.ndarray = np.empty(shape=results.shape[2], dtype=float)

    for k in range(H.shape[0]):
        _, cnts = np.unique(results[:,:,k], axis=0, return_counts=True)

        if q == 0:
            H[k] = np.log(cnts.shape[0])
            continue

        probs = cnts / cnts.sum()

        if abs(q - 1) < shannon_entropy_tol:
            H[k] = -np.sum(probs * np.log(probs))
            continue

        H[k] = np.log(np.sum(probs ** q)) / (1 - q)

    return H


@f._numba_exec(_get_renyi_entropy_numba_bc, _get_renyi_entropy_numba_fast)
def _get_renyi_entropy(
        q: float,
        data : np.ndarray,
        scales: np.ndarray,
        shannon_entropy_tol: float = 1e-6,
        *,
        debug_numba: \
            Iterable[Literal["ignore", "warn", "raise", "bounds"]] = "ignore") -> np.ndarray:
    """
    Hybrid implementation of the Rényi entropy. Defaults to Numba computation,
    falling back to Python implemtnation if required.

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

    return _get_renyi_entropy_py(q, data, scales, shannon_entropy_tol)

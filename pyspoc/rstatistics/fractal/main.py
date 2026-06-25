import numpy as np
import warnings

from types import FunctionType

from pyspoc import ReducedStatistic

from . import func as f
from . import func_numba as fnumba

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


class GeneralizedEntropy(ReducedStatistic):

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

    adj_r2_tol : float, default=0.02
        Minimum proportional improvement in adjusted :math:`R^2` required to
        accept an additional linear segment during piecewise regression model
        selection.

    elbow_tol : float, default=0.15
        Proportional tolerance used to identify outer segments as elbows. If the
        slope of an outer segment is below this tolerance times the average slope
        of the inner segments, that outer segment is removed.

    deshmukh_reg_proportion : float, default=0.25
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
    defined in Deshmukh et al. (2021), which involves fitting an ensemble of
    linear regressions computed over subintervals of the retained scaling region.
    These slope estimates are then aggregated using a mode-weighted averaging procedure.

    References
    ----------
    .. [1] Cui, T., & Wang, T. (2024). Exact box-counting and temporal sampling algorithms for
        fractal dimension estimation with applications to animal behavior analysis.
        Results in Engineering.
        (Some code has been integrated into this class and can be found at
        https://github.com/wanglab-georgetown/fractal.)

    .. [2] Datseris et al. (2023). Estimating fractal dimensions: A comparative review
        and open source implementations. Chaos.

    """

    _name = "Generalized Entropy"
    _identifier = "nde"
    _labels = ["fractal"]
    _NAN_RETURN_WARNING = "No scaling could be determined for the provided dataset. " \
        "Returning NaN result."
    _SHANNON_ENTROPY_TOL = 1e-6

    def __init__(self,
                 q: float = 0,
                 adj_r2_tol: float = 0.02,
                 elbow_tol: float = 0.15,
                 deshmukh_reg_proportion: float = 0.25,
                 debug_numba: bool = False):
        
        self._q = q
        self._deshmukh_reg_proportion = deshmukh_reg_proportion
        self._adj_r2_tol = adj_r2_tol
        self._elbow_tol = elbow_tol
        self._debug_numba = debug_numba
        super().__init__()

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
    
    @property
    def adj_r2_tol(self) -> float:
        return self._adj_r2_tol
    
    @property
    def elbow_tol(self) -> float:
        return self._elbow_tol

    @property
    def deshmukh_reg_proportion(self) -> float:
        return self._deshmukh_reg_proportion
        
    def _fallback_execute(self,
                          func: FunctionType,
                          fallback_func: FunctionType,
                          *args,
                          **kwargs):
        try:
            result = func(*args, **kwargs)
            
            if self._debug_numba:
                print(f"Numba ran {func.__name__} successfully.")

            return result
        except:
            if self._debug_numba:
                print(f"Numba failed to run {func.__name__}.")

            return fallback_func(*args, **kwargs)

    def compute(self, data: np.ndarray) -> np.ndarray | float:

        nan_array = np.array(np.nan)
        
        scales = self._fallback_execute(
            fnumba._get_default_init_scale,
            f._get_default_init_scale,
            data
        )

        # Return NaN array if no scales found.
        if scales is None:
            warnings.warn(self._NAN_RETURN_WARNING)
            return nan_array
        
        # Fractal estimator.
        H = self._fallback_execute(
            fnumba._get_renyi_entropy,
            f._get_renyi_entropy,
            self.q,
            data,
            scales,
            self._SHANNON_ENTROPY_TOL
        )

        # Return NaN array if no result.
        if H is None:
            warnings.warn(self._NAN_RETURN_WARNING)
            return nan_array
        
        neg_log_scales = -np.log(scales)
        best_fit = f._get_best_parsimonous_model_fit(neg_log_scales,
                                                     H,
                                                     self.adj_r2_tol)
        
        alphas, breakpoints = f._get_pieces(best_fit,
                                            neg_log_scales)

        trimmed_scales, trimmed_H = self._fallback_execute(
            fnumba._trim_elbows,
            f._trim_elbows,
            alphas,
            breakpoints,
            neg_log_scales,
            H,
            self.elbow_tol)
       
        slopes, weights = f._compute_slope_ensemble(trimmed_scales,
                                                    trimmed_H,
                                                    self.deshmukh_reg_proportion)
        
        fd = f._return_modal_average(slopes, weights)
        return fd

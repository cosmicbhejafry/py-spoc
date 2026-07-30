import numpy as np

from abc import ABC, abstractmethod
from typing import Literal, Iterable

from pyspoc import ReducedStatistic

from . import funcs_py as fpy
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


class FractalMeasureBase(ReducedStatistic, ABC):

    """
    Base class for fractal dimension estimators.

    Concrete children must implement method compute_density_estimate() that computes a
    point cloud density estimate over multiple box scales, returned as a numpy array.
    For more detailed instructions, see the abstract method below.

    Given a static point dataset of shape ``(n, p)``, the method discretizes
    the state space into non-overlapping boxes across a range of scales and,
    using the implemented compute_density_estimate() method, computes the induced
    natural density at each scale. Elbow detection and a scaling region are
    then identified to estimate the fractal dimension.

    
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


    Parameters
    ----------
    adj_r2_tol : float, default=0.02, requires: >0, <=1
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
    This implementation uses a box-counting approximation over a default set of initial
    scales. The final estimate depends on the quality of the observed scaling region and
    may be sensitive to sample size, scale selection, and the elbow-detection tolerances.


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

    _WARNING_INTERVAL_BOUNDARIES = "{var_name} expects value greater than {min} " \
        "and less than or equal to {max}, got {got}. Value has been clipped at {clipped}."

    _ERROR_DENSITY_MISSING = "No densities could be determined for the provided dataset. " \
        "Returning NaN result."

    _ERROR_DENSITY_CARDINARITY = "Incorrect number of densities computed. " \
        "Expected {expected}, got {actual}. Returning NaN result."
    
    _ERROR_FRACTAL_DIM_FAILURE = "No fractal dimension could be estimated. Both OLS and Deshmukh " \
        "methods have failed. Please check input data and if issue persists, raise with authors."

    def __init__(self,
                 *,
                 adj_r2_thresh: float = 0.9,
                 elbow_thresh: float = 0.25,
                 deshmukh_reg_proportion: float = 0.25,
                 use_adaptive_scaling: bool = True,
                 scale_method: Literal["datseries", "base-10"] = "datseries",
                 scale_length: int = 50,
                 scale_adaption_iters: int = 20,
                 debug_numba: Iterable[Literal["ignore", "warn", "raise", "bounds"]] = "ignore",
                 **kwargs):
                        
        self._deshmukh_reg_proportion = self._handle_interval_variables(
            deshmukh_reg_proportion,
            "deshmukh_reg_proportion",
            0,
            0.5)
        
        self._adj_r2_tol = self._handle_interval_variables(adj_r2_thresh, "adj_r2_thresh")
        self._elbow_tol = self._handle_interval_variables(elbow_thresh, "elbow_thresh")
        self._debug_numba = [debug_numba] if \
            isinstance(debug_numba, (str, bytes)) or not isinstance(debug_numba, Iterable) \
            else debug_numba
        self._use_adaptive_scaling = use_adaptive_scaling
        self._scale_method = scale_method
        self._scale_length = scale_length
        self._scale_adaption_iters = scale_adaption_iters
        self._kwargs = kwargs
        super().__init__()

    @property
    def adj_r2_tol(self) -> float:
        return self._adj_r2_tol
    
    @property
    def elbow_tol(self) -> float:
        return self._elbow_tol

    @property
    def deshmukh_reg_proportion(self) -> float:
        return self._deshmukh_reg_proportion
    
    @property
    def use_adaptive_scaling(self) -> bool:
        return self._use_adaptive_scaling
    
    @property
    def scale_method(self) -> str:
        return self._scale_method
    
    @property
    def scale_length(self) -> int:
        return self._scale_length
    
    @property
    def scale_adaption_iters(self) -> int:
        return self._scale_adaption_iters

    def _handle_interval_variables(
            self,
            var: float,
            var_name: str,
            min: float = 0,
            max: float = 1):
        
        clipped = None
        
        if var <= 0:
            clipped = 0.01

        if var > 1:
            clipped = 1

        if clipped:
            print(self._WARNING_INTERVAL_BOUNDARIES.format(
                var_name = var_name,
                got = var,
                min = min,
                max = max,
                clipped = clipped))
            return clipped

        return var

    @abstractmethod
    def _compute_density_estimate(
            self,
            data: np.ndarray,
            scales: np.ndarray) -> np.ndarray:
        pass

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        
        # Get boxes of different scales.
        if self.use_adaptive_scaling:
            scales = fpy._get_adaptive_scales(
                data,
                method=self.scale_method,
                k=self.scale_length,
                max_iter=self.scale_adaption_iters,
                **self._kwargs)
            
        else:
            if self.scale_method == "datseries":
                scales = fpy._get_datseries_scales(
                    data,
                    k=self.scale_length,
                    **self._kwargs
                )
            else:
                scales = fpy._get_log10_scales(
                    data,
                    k=self.scale_length,
                    **self._kwargs
                )
        
        scales = scales[::-1]
        
        # Estimate density at each box scale.
        densities = self._compute_density_estimate(data, scales)

        # Throw error if no result.
        if densities is None:
            raise ValueError(self._ERROR_DENSITY_MISSING)
        
        # Throw error if scales and density counts are different.
        if densities.shape != scales.shape:
            raise ValueError(self._ERROR_DENSITY_CARDINARITY.format(
                expected=scales.shape,
                actual=densities.shape))
        
        # Log transform for linear relationship.
        neg_log_scales = -np.log(scales)

        # Perform OLS regression of the scaling curve as starting point.
        slope, adj_r2 = fpy._compute_ols_results(neg_log_scales, densities)

        if adj_r2 < self.adj_r2_tol:

            elbow_idxs = fpy._find_elbow_idx(
                neg_log_scales,
                densities,
                direction="increasing",
                polyorder=3,
                monotonic_tolerance=0.1)

            if elbow_idxs:
                elbow_filter = slice(*elbow_idxs)
                neg_log_scales = neg_log_scales[elbow_filter]
                densities = densities[elbow_filter]

       # Compute fractal dimension using Deshmukh method.
        fd = fnb._compute_deshmukh_slope_estimate(
            neg_log_scales,
            densities,
            self.deshmukh_reg_proportion)
        
        if fd:
            # Return fractal dimension estimate if available.
            return fd

        # Return OLS slope if Deshmukh method fails.
        if slope:
            return slope
        
        # Raise error if both failed
        raise ValueError(self._ERROR_FRACTAL_DIM_FAILURE)
                
        
        

import numpy as np

from abc import ABC, abstractmethod
from typing import Literal

from pyspoc import ReducedStatistic
from pyspoc._argchecking import (
    RuntimeTypeCheckedMixin,
    check_float,
    check_integer,
    check_natural_number,
)

from . import _funcs_py as fpy
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


class FractalMeasureBase(RuntimeTypeCheckedMixin, ReducedStatistic, ABC):

    """
    Base class for fractal dimension estimators.

    Concrete children must implement ``_compute_density_estimate()``, which computes a
    point cloud density estimate over multiple box scales, returned as a numpy array.
    For more detailed instructions, see the abstract method below.

    Given a static point dataset of shape ``(n, p)``, the method discretizes
    the state space into non-overlapping boxes across a range of scales and,
    using the implemented ``_compute_density_estimate()`` method, computes the induced
    natural density at each scale. Elbow detection and a scaling region are
    then identified to estimate the fractal dimension.

    
    Parameters
    ----------
    slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
        Method used to select the final scaling-curve slope. ``"ols"`` always
        returns the ordinary least-squares estimate when available,
        ``"deshmukh"`` prefers the ensemble estimate described by Deshmukh
        et al., and ``"hybrid"`` returns the OLS estimate when its coefficient
        of determination meets ``r2_thresh`` and otherwise uses the Deshmukh
        estimate.

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
        Absolute minimum number of points that an elbow-trimmed scaling region
        must retain. If ``None``, this is 40% of the data-dependent scale
        length, with a minimum of 20 points.

    use_adaptive_scaling : bool, default=True
        Whether to iteratively adapt the initial scale range to the informative
        box-counting region. If ``False``, the scale range produced directly by
        ``scale_method`` is used.

    scale_method : {"datseries", "log-10"}, default="datseries"
        Algorithm used to generate the initial box scales. ``"datseries"``
        uses the data-dependent scale construction, while ``"log-10"`` uses a
        base-10 logarithmic range.

    scale_length : int or None, default=None, requires: >=50
        Number of intervals in the generated scale range. If ``None``, this is
        resolved from dimensionality :math:`p` as
        ``max(50, ceil(100 * log(p)))``. Scale generators may consequently
        return ``scale_length + 1`` scale values.

    scale_adaption_iters : int, default=20, requires: >=1
        Maximum number of iterations used to expand an adaptively generated
        scale range. This argument is used only when
        ``use_adaptive_scaling=True``.

    **kwargs : Any, optional
        Additional keyword arguments forwarded to the selected scale-generation
        function.


    Returns
    -------
    float
        Estimated fractal dimension.


    Notes
    -----
    This implementation uses a box-counting approximation over a default set of initial
    scales. The final estimate depends on the quality of the observed scaling region and
    may be sensitive to sample size, scale selection, and the elbow-detection tolerances.


    Algorithm
    ---------------
    1. Generate the initial scaling region using `adaptive`, `datseries`, or
        `log-10` methods.
    2. For each scale:
            2a. Partition the data space into hypercubes of the given scale.
            2b. Compute the densities of the data over the hypercubes.
    3. Collect the negative log of the scaling region and the respective densities
        as the scaling curve.
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
    of unstable outer-scale behaviour. The curve is smoothed before candidate
    elbows are identified.

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

    _ERROR_DENSITY_MISSING = "No densities could be determined for the provided dataset. " \
        "Returning NaN result."

    _ERROR_DENSITY_CARDINARITY = "Incorrect number of densities computed. " \
        "Expected {expected}, got {actual}. Returning NaN result."
    
    _ERROR_FRACTAL_DIM_FAILURE = "No fractal dimension could be estimated. Both OLS and Deshmukh " \
        "methods have failed. Please check input data and if issue persists, raise with authors."

    def __init__(self,
                 *,
                 slope_estimation_method: Literal["hybrid", "ols", "deshmukh"] = "hybrid",
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
        """Initialize the shared fractal-dimension estimation workflow.

        Parameters
        ----------
        slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
            Strategy used to select the final slope estimate. ``"hybrid"``
            accepts a sufficiently strong OLS fit and otherwise prefers the
            Deshmukh estimate.
        r2_thresh : float, default=0.9, requires: >=0, <=1
            Minimum OLS :math:`R^2` accepted by the hybrid strategy.
        monotonic_tol : float, default=0.1, requires: >=0, <=1
            Maximum proportion of adjacent monotonicity violations allowed by
            elbow detection. A value of ``1`` disables this rejection criterion.
        deshmukh_reg_proportion : float, default=0.25, requires: >0, <=0.5
            Minimum relative length of scaling-curve subintervals included in
            the Deshmukh slope ensemble.
        minimum_scaling_region : float, default=0.1, requires: >0, <=1
            Minimum proportion of the original point count and log-scale span
            that an elbow-trimmed region must retain.
        minimum_scaling_points : int or None, default=None, requires: >=20
            Absolute minimum number of retained scaling-curve points. If
            omitted, this is 40% of the resolved scale length, with a minimum
            of 20 points.
        use_adaptive_scaling : bool, default=True
            Whether to iteratively adapt the generated scale range.
        scale_method : {"datseries", "log-10"}, default="datseries"
            Method used to construct the initial box scales.
        scale_length : int or None, default=None, requires: >=50
            Number of intervals in the generated scale range. If omitted, this
            is ``max(50, ceil(100 * log(p)))`` for dimensionality ``p``.
        scale_adaption_iters : int, default=20, requires: >=1
            Maximum adaptive-scale expansion iterations.
        **kwargs : Any, optional
            Additional arguments forwarded to the selected scale generator.

        Raises
        ------
        TypeError
            If a checked numeric argument has an incompatible type.
        ValueError
            If a checked numeric argument lies outside its documented bounds.
        """
                        
        self._slope_estimation_method = slope_estimation_method
        self._deshmukh_reg_proportion = check_float(
            deshmukh_reg_proportion,
            exclusive_minimum=0,
            maximum=0.5,
            arg_name="deshmukh_reg_proportion")
        
        self._r2_thresh = check_float(
            r2_thresh, minimum=0, maximum=1, arg_name="r2_thresh")
        self._monotonic_tol = check_float(
            monotonic_tol, minimum=0, maximum=1, arg_name="monotonic_tol")
        self._minimum_scaling_region = check_float(
            minimum_scaling_region,
            exclusive_minimum=0,
            maximum=1,
            arg_name="minimum_scaling_region")
        self._minimum_scaling_points = (
            20
            if minimum_scaling_points is None
            else check_integer(
                minimum_scaling_points,
                minimum=20,
                arg_name="minimum_scaling_points"))
        self._use_adaptive_scaling = use_adaptive_scaling
        self._scale_method = scale_method
        self._scale_length = (
            None
            if scale_length is None
            else check_integer(scale_length, minimum=50, arg_name="scale_length"))
        self._scale_adaption_iters = check_natural_number(
            scale_adaption_iters, arg_name="scale_adaption_iters")
        self._kwargs = kwargs
        super().__init__()

    @property
    def slope_estimation_method(self) -> str:
        return self._slope_estimation_method

    @property
    def r2_thresh(self) -> float:
        return self._r2_thresh
    
    @property
    def monotonic_tol(self) -> float:
        return self._monotonic_tol

    @property
    def deshmukh_reg_proportion(self) -> float:
        return self._deshmukh_reg_proportion

    @property
    def minimum_scaling_region(self) -> float:
        return self._minimum_scaling_region

    @property
    def minimum_scaling_points(self) -> int | None:
        return self._minimum_scaling_points
    
    @property
    def use_adaptive_scaling(self) -> bool:
        return self._use_adaptive_scaling
    
    @property
    def scale_method(self) -> str:
        return self._scale_method
    
    @property
    def scale_length(self) -> int | None:
        return self._scale_length
    
    @property
    def scale_adaption_iters(self) -> int:
        return self._scale_adaption_iters

    def _resolve_scale_length(self, data: np.ndarray) -> int:
        """Resolve the configured or dimensionality-based scale length."""
        if self.scale_length is not None:
            return self.scale_length

        dimensions = data.shape[1]
        return max(50, int(np.ceil(100 * np.log(dimensions))))

    def _resolve_minimum_scaling_points(self, scale_length: int) -> int:
        proportional_minimum = int(np.ceil(
            self.minimum_scaling_region * scale_length))
        required_points = max(
            self._minimum_scaling_points,
            proportional_minimum)
        return required_points

    @abstractmethod
    def _compute_density_estimate(
            self,
            data: np.ndarray,
            scales: np.ndarray) -> np.ndarray:
        pass

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        scale_length = self._resolve_scale_length(data)
        
        # Get boxes of different scales.
        if self.use_adaptive_scaling:
            scales = fpy.get_adaptive_scales(
                data,
                method=self.scale_method,
                k=scale_length,
                max_iter=self.scale_adaption_iters,
                **self._kwargs)
            
        else:
            if self.scale_method == "datseries":
                scales = fpy.get_datseries_scales(
                    data,
                    k=scale_length,
                    **self._kwargs
                )
            else:
                scales = fpy.get_log10_scales(
                    data,
                    k=scale_length,
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
        slope, r2 = fpy.compute_ols_results(neg_log_scales, densities)

        elbow_idxs = fpy.find_elbow_idx(
            neg_log_scales,
            densities,
            direction="increasing",
            polyorder=3,
            monotonic_tolerance=self.monotonic_tol)

        if elbow_idxs:
            elbow_filter = slice(*elbow_idxs)
            candidate_scales = neg_log_scales[elbow_filter]
            candidate_densities = densities[elbow_filter]
            required_points = self._resolve_minimum_scaling_points(neg_log_scales.shape[0])
            retains_enough_points = candidate_scales.shape[0] >= required_points
            retains_enough_span = (
                np.ptp(candidate_scales)
                >= self.minimum_scaling_region * np.ptp(neg_log_scales))

            if retains_enough_points and retains_enough_span:
                neg_log_scales = candidate_scales
                densities = candidate_densities
                slope, r2 = fpy.compute_ols_results(
                    neg_log_scales,
                    densities)

        # Return the OLS slope if it certain conditions are met.
        if slope:
            if (self.slope_estimation_method == "ols"
                or (self.slope_estimation_method == "hybrid" and r2 >= self._r2_thresh)):
                return slope

       # Compute fractal dimension using Deshmukh method.
        fd = fnb.compute_deshmukh_slope_estimate(
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

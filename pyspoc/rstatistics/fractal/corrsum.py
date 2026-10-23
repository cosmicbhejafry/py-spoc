import numpy as np

from typing import Any, Literal

from pyspoc._tag import Tag, Type, Signature
from .base import FractalMeasureStatistic
from . import _funcs_py as fpy

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


class CorrelationSum(FractalMeasureStatistic):

    """
    <Correlation Sum description>

    Given a static point dataset of shape ``(n, p)``, the method discretizes
    the state space into non-overlapping boxes across a range of scales and
    computes the entropy of the induced natural density at each scale. A scaling
    region is then identified and used to estimate the generalized fractal
    dimension.


    Definition
    ----------------------------
    <Mathematical definition of correlation sum>
   

    Parameters
    ----------
    slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
        Method used to select the final scaling-curve slope. ``"ols"`` uses
        ordinary least squares, ``"deshmukh"`` prefers the slope-ensemble
        estimate, and ``"hybrid"`` selects between them using ``r2_thresh``.

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
        Estimated correlation sum fractal dimension.


    Notes
    -----
    <Correlation sum implementation specific notes>

    The final estimate depends on the quality of the observed scaling region and
    may be sensitive to sample size, scale selection, and the elbow-detection
    tolerances.


    Algorithm
    ---------------
    1. Generate the initial scaling region using `datseries`, or
    `log base-10` methods.
    2. Refine the scaling region until the unique point pair counts at each scale
        have a range of 90% of the unique point pairs (`0.9 * n * (n-1) / 2`) by:
        - Computing the Euclidean distance between each unique point pair.
        - For each radius scale, tally the number of point pairs with distance less
            than or equal to the scale.
    3. Retain the final counts from the refinement above as the correlation sum.
    4. Collect the negative log of the scaling region and the respective correlation
        sum as the scaling curve.
    5. Apply OLS to the entire scaling curve.
    6. Attempt elbow detection and removal. Accept the trimmed region only if
    it retains the required number and proportion of points and the required
    proportion of the original log-scale span, then refit OLS.
    7. If OLS :math:`R^2` `>= r2_thresh` requirement and `slope_estimation_method == 'hybrid'`:
       - Return the OLS slope as the fractal dimension estimate.
       Else `slope_estimation_method == "ols"`:
       - Return the OLS slope as the fractal dimension estimate.
    8. Compute the Deshmukh slope estimate.
    9. If Deshmukh slope estimate is valid:
        - Return Deshmukh slope estimate.
    10. Otherwise, return the last computed OLS slope estimate.


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
    The method is defined in [2], which involves fitting an ensemble of linear
    regressions computed over subintervals of the retained scaling curve. These slope
    estimates are then averaged to produce the final fractal dimension estimate.


    References
    ----------
    .. [1] Datseris et al. (2023). Estimating fractal dimensions: A comparative review
        and open source implementations. Chaos.

    .. [2] V. Deshmukh, E. Bradley, J. Garland, and J. D. Meiss (2021). Toward automated
        extraction and characterization of scaling regions in dynamical systems. Chaos.

    """

    _long_name = "Correlation Sum"
    _identifier = "fractal-cs"
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.DIMENSIONALITY,)
    _tags = (Tag.FRACTAL, Tag.NONLINEAR, Tag.NONPARAMETRIC)

    def __init__(
            self,
            slope_estimation_method: Literal[
                "hybrid", "ols", "deshmukh"] = "hybrid",
            r2_thresh: float = 0.9,
            monotonic_tol: float = 0.1,
            deshmukh_reg_proportion: float = 0.25,
            minimum_scaling_region: float = 0.1,
            minimum_scaling_points: int | None = None,
            initial_scale_method: Literal[
                "datseries", "log-10"] = "datseries",
            scale_length: int | None = None,
            scale_adaption_iters: int = 20,
            **kwargs):
        
        """Initialize a Rényi generalized-dimension statistic.

        Parameters
        ----------
        slope_estimation_method : {"hybrid", "ols", "deshmukh"}, default="hybrid"
            Strategy used to select the final scaling-curve slope.
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
        initial_scale_method : {"datseries", "log10"}, default="datseries"
            The method for generating the initial radii scale to be refined.
        scale_length : int or None, default=None, requires: >=50
            Number of radii scales in the generated scale range. If omitted, this
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
                
        super().__init__(slope_estimation_method=slope_estimation_method,
                         r2_thresh=r2_thresh,
                         monotonic_tol=monotonic_tol,
                         deshmukh_reg_proportion=deshmukh_reg_proportion,
                         minimum_scaling_region=minimum_scaling_region,
                         minimum_scaling_points=minimum_scaling_points,
                         initial_scale_method=initial_scale_method,
                         scale_length=scale_length,
                         scale_adaption_iters=scale_adaption_iters,
                         **kwargs)


    def _compute(self, data: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> float | np.floating:
        non_negative_data = data - data.min()
        scale_length = self._resolve_scale_length(non_negative_data)

        scales, counts = fpy.get_adaptive_corr_sum_results(
            non_negative_data,
            method=self.scale_method,
            k=scale_length,
            max_iter=self.scale_adaption_iters,
            **self._kwargs)
        
        counts = counts.astype(np.float64)
        zero_count_mask = counts > 0
        counts = counts[zero_count_mask]
        scales = scales[zero_count_mask]
        densities = np.log(counts)
        return self._compute_fd(scales, densities)

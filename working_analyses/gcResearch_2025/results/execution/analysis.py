from __future__ import annotations

import numpy as np

from functools import partial
from typing import Any


def fit_multivariate_gaussian_pdf(
        data: np.ndarray,
        noise: float) -> partial[Any]:
    
    """Fit an arbitrary-dimensional multivariate Gaussian PDF.

    Parameters
    ----------
    data : numpy.ndarray
        Finite numeric array with shape
        ``(n_observations, n_dimensions)``. At least two observations and one
        dimension are required.
    noise : float
        Finite non-negative value added to every diagonal entry of the sample
        covariance matrix.

    Returns
    -------
    functools.partial
        PDF callable accepting one point or a compatible array of points. Its
        fixed ``mean`` and ``cov`` keyword arguments contain the fitted
        Gaussian parameters.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array, values are non-numeric, or ``noise``
        is not a real number.
    ValueError
        If the input rank or size is invalid, values are non-finite, or
        ``noise`` is negative or non-finite.

    Notes
    -----
    The returned partial is directly compatible with
    ``similarity.gaussian_group_similarity``. Singular empirical covariance
    matrices are permitted, which is particularly useful when dimensions
    outnumber observations or some features are linearly dependent.
    """
    from scipy.stats import multivariate_normal

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if isinstance(noise, bool) or not isinstance(
            noise, (int, float, np.integer, np.floating)):
        raise TypeError("noise must be a real number.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and non-negative.")
    if data.ndim != 2:
        raise ValueError(
            "data must have shape (n_observations, n_dimensions)."
        )
    if data.shape[0] < 2:
        raise ValueError("data must contain at least two observations.")
    if data.shape[1] < 1:
        raise ValueError("data must contain at least one dimension.")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must contain numeric values.")

    matrix = np.asarray(data, dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError("data must contain only finite values.")

    dimensionality = matrix.shape[1]
    mean = matrix.mean(axis=0)
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    fitted_covariance = (
        covariance
        + float(noise) * np.eye(dimensionality)
    )
    return partial(
        multivariate_normal.pdf,
        mean=mean,
        cov=fitted_covariance,
        allow_singular=True,
    )


def evaluate_gaussian_density_grid(
        data: np.ndarray,
        noise: float,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        grid_step: float = 0.05) -> tuple[np.ndarray, partial[Any]]:
    
    """Fit a bivariate Gaussian and evaluate its density on a regular grid.

    The Gaussian mean is the sample centroid. Its covariance is the sample
    covariance regularized by adding ``noise * I``. Grid limits span three
    marginal standard deviations either side of the fitted mean. Each bound is
    rounded outwards to the nearest multiple of ``grid_step`` so independently
    fitted Gaussians lie on the same coordinate lattice.

    Parameters
    ----------
    data : numpy.ndarray
        Finite numeric array with shape ``(n_samples, 2)``. Rows represent
        observations and columns represent the x and y variables.
    noise : float
        Finite, non-negative diagonal covariance regularization.
    xlim : tuple of float or None, optional
        Explicit x-axis limits. When omitted, use the fitted mean plus or minus
        three marginal standard deviations.
    ylim : tuple of float or None, optional
        Explicit y-axis limits. When omitted, use the fitted mean plus or minus
        three marginal standard deviations.
    grid_step : float, default=0.05
        Positive spacing between adjacent x and y coordinates. Grid coordinates
        are integer multiples of this value.

    Returns
    -------
    grid : numpy.ndarray
        Array with shape ``(n_x, n_y, 3)``. ``grid[..., 0]`` contains the x
        coordinates, ``grid[..., 1]`` contains the y coordinates, and
        ``grid[..., 2]`` contains the fitted Gaussian density. Grid sizes depend
        on the fitted spread and ``grid_step``.
    pdf : functools.partial
        Fitted bivariate Gaussian PDF, also suitable for direct point-density
        evaluation and similarity computation.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array or a numeric argument is not a real
        number.
    ValueError
        If the data do not have shape ``(n_samples, 2)``, contain fewer than
        two observations, contain non-finite values, or ``noise`` is negative
        or non-finite, or ``grid_step`` is not finite and positive.

    Notes
    -----
    The standard deviations used for the grid limits include the supplied
    noise because they are derived from the regularized fitted covariance.
    ``allow_singular=True`` permits density evaluation when ``noise`` is zero
    and the empirical covariance is singular.
    """
    pdf = fit_multivariate_gaussian_pdf(data, noise)

    if isinstance(grid_step, bool) or not isinstance(
            grid_step, (int, float, np.integer, np.floating)):
        raise TypeError("grid_step must be a real number.")
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be finite and positive.")
    if data.shape[1] != 2:
        raise ValueError("data must have shape (n_samples, 2).")
    mean = np.asarray(pdf.keywords["mean"], dtype=float)
    fitted_covariance = np.asarray(pdf.keywords["cov"], dtype=float)
    standard_deviations = np.sqrt(np.diag(fitted_covariance))

    def make_discretized_axis(
            centroid: float,
            standard_deviation: float,
            lim: tuple[float, float] | None) -> np.ndarray:
        """Return outward-rounded three-sigma coordinates on one lattice."""
        step = float(grid_step)

        if lim is not None:
            lower_scaled, upper_scaled = lim[0] / step, lim[1] / step
        else:
            lower_scaled = (centroid - 3 * standard_deviation) / step
            upper_scaled = (centroid + 3 * standard_deviation) / step

        # The tolerance prevents an exact mathematical lattice point from
        # being moved by one extra step solely because of floating-point noise.
        lower_tolerance = 1e-12 * max(1.0, abs(lower_scaled))
        upper_tolerance = 1e-12 * max(1.0, abs(upper_scaled))
        lower_index = int(np.floor(lower_scaled + lower_tolerance))
        upper_index = int(np.ceil(upper_scaled - upper_tolerance))

        # A zero-variance axis otherwise has only one coordinate, which cannot
        # form image cells or establish grid spacing for downstream blending.
        if lower_index == upper_index:
            lower_index -= 1
            upper_index += 1

        return np.arange(lower_index, upper_index + 1, dtype=float) * step

    x_points = make_discretized_axis(mean[0], standard_deviations[0], xlim)
    y_points = make_discretized_axis(mean[1], standard_deviations[1], ylim)
    x_grid, y_grid = np.meshgrid(x_points, y_points, indexing="ij")
    grid_points = np.stack((x_grid, y_grid), axis=-1)

    density = pdf(x=grid_points)
    return np.concatenate((grid_points, density[..., np.newaxis]), axis=-1), pdf

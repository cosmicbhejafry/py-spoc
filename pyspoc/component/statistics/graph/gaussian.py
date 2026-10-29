from __future__ import annotations

import numpy as np
from typing import Literal
from collections.abc import Collection

from pyspoc.component.statistics.association.correlation import PartialCorrelation
from pyspoc._tag import Tag, Signature
from ._base import GraphStatistic



class GGM(GraphStatistic):
    """Fit a Gaussian graphical model using standardised partial correlation estimated from data.

    Parameters
    ----------
    inversion_method : {"pseudo", "graphical_lasso", "ledoit_wolf"}, default="graphical_lasso"
        Method used to estimate the precision matrix. ``"graphical_lasso"`` estimates a sparse
        precision matrix, selecting its penalty by cross-validation when ``alpha`` is ``None``.
        ``"pseudo"`` applies a Hermitian Moore-Penrose pseudoinverse to the empirical covariance,
        producing a saturated graphical model. ``"ledoit_wolf"`` uses analytically selected
        covariance shrinkage.
    absolute : bool, default=False
        Replace signed coefficients by their magnitudes.
    threshold : float or None, optional
        Set edges with absolute weight below this value to zero.
    include_self : bool, default=False
        Retain unit-weight self edges on the diagonal.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.
    alpha : float or None, optional or requires: >0
        Graphical LASSO penalty coefficient. If ``None``, use
        :class:`sklearn.covariance.GraphicalLassoCV`.

        ### Cross-Validation Parameters

    cv_alphas : int or Collection[float], requires: >=1 for int, >=0 for floats.
        If an integer is given, it fixes the number of points on the grids of alpha to be used.
        If a collection is given, it gives the grid to be used.
    cv_folds : int, default=5, requires: >=2
        Number of cross-validation folds used when Graphical LASSO selects
        ``alpha`` automatically.
    cv_n_refinements : int, default=4, requires: >=1
        The number of times the grid is refined. Not used if explicit values of alphas are passed.
    cv_tol : float, default=1e-4, requires: >0
        The tolerance to declare convergence. If the dual gap goes below this value,
        iterations are stopped.
    cv_enet_tol : float, default=1e-4, requires: >0
        The tolerance for the elastic net solver used to calculate the descent direction.
        This parameter controls the accuracy of the search direction for a given column update,
        not of the overall parameter estimate.
    cv_max_iter : int, default=100, requires: >=1
        Maximum number of cross-valiation iterations.
    cv_mode : {"cd", "lars"}, default="cd"
        The Lasso solver to use: coordinate descent or LARS. Use `"lars"` for very sparse underlying
        graphs, where number of features is greater than number of samples. Otherwise, `"cd"` is
        preferred due to greater numerical stability.
    cv_eps : float, default=`np.finfo(np.float64).eps`
        The machine-precision regularization in the computation of the Cholesky diagonal factors.
        Increase this for very ill-conditioned systems.
    """

    _long_name = "Gaussian Graphical Model Graph"
    _identifier = "graph-ggm"
    _signatures = (Signature.DENSITY, Signature.TOPOLOGY)
    _tags = (Tag.PARAMETRIC, Tag.SIGNED, Tag.DISTRIBUTION, Tag.GRAPH_WEIGHTED, Tag.GRAPH_DIRECTED)

    def __init__(
            self,
            inversion_method: Literal[
                "pseudo",
                "pseudoinverse",
                "graphical_lasso",
                "ledoit_wolf"] = "graphical_lasso",
            ensure_adjacency_type: Literal[
                "binary",
                "binary-weighted",
                "bipolar",
                "bipolar-weighted",
                "absolute",
                None] = None,
            adjacency_scaling: Literal[
                "minmax",
                "clip",
                None
            ] = None,
            zero_threshold: float | None = None,
            self_value: float | None = 0,
            assume_centered: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            *,
            alpha: float | None = None,
            cv_alphas : int | Collection[float] = 4,
            cv_folds: int = 5,
            cv_n_refinements: int = 4,
            cv_tol: float = 1e-4,
            cv_enet_tol: float = 1e-4,
            cv_max_iter: int = 100,
            cv_mode: Literal["cd", "lars"] = "cd",
            cv_eps: float = np.finfo(np.float64).eps):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            ensure_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold,
            self_value=self_value)
        
        if adjacency_scaling in ("binary", "binary-weighted", "absolute"):
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=[Tag.GRAPH_DIRECTED, Tag.SIGNED])
        
        self._nan_policy = nan_policy
        self._base_stat = PartialCorrelation(
            squared=False,
            dim=dim,
            inversion_method=inversion_method,
            nan_policy=nan_policy,
            assume_centered=assume_centered,
            alpha=alpha,
            cv_alphas=cv_alphas,
            cv_folds=cv_folds,
            cv_n_refinements=cv_n_refinements,
            cv_tol=cv_tol,
            cv_enet_tol=cv_enet_tol,
            cv_max_iter=cv_max_iter,
            cv_mode=cv_mode,
            cv_eps=cv_eps)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Return the ``(n_features, n_features)`` correlation adjacency."""
        graph_mat = self._base_stat._compute(data)

        if self._adjacency_type:
            from sklearn.preprocessing import MinMaxScaler
            graph_mat = MinMaxScaler(feature_range=(-1,1)).fit_transform(graph_mat)

        return graph_mat

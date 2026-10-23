"""Association-based feature graph Statistics."""
from __future__ import annotations

import numpy as np

from typing import Literal, Any
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

import pyspoc._argchecking as argchk

from pyspoc._tag import Tag, Type, Signature
from pyspoc.statistics.association.correlation import (
    Pearson,
    SpearmanR,
    KendallTau,
    PartialCorrelation)
from pyspoc.statistics.association.covariance import (
    Covariance)
from pyspoc.settings import settings

from ._base import GraphStatistic



class CorrelationGraph(GraphStatistic):
    """Construct a feature graph from pairwise correlation coefficients.

    Parameters
    ----------
    estimator : {"Pearson", "SpearmanR", "KendallTau", "Partial"}, default="Pearson"
        Correlation coefficient used for each pair of feature columns.
    ensure_adjacency-type : {"binary", "binary-weighted", "bipolar", \
        "bipolar-weighted", "absolute", `None`}
        <Insert description here>
    adjacency_scaling : {"minmax", "clip", None}
        <Insert description here>
    zero_threshold : float or None, optional
        Set edges with absolute weight below this value to zero. If `"binary"` or `"bipolar"`
        are selected as the `ensure_adjacency_type`, this value defaults to `0.5`.
    self_value : float or None, default=0
        Set the self edges on the diagonal of the graph adjacency to this value. If `None`
        is passed then no change is made to the original adjacency matrix.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.

    Notes
    ----------
    For a partial correlation based approach, see the :mod:`pyspoc.statistics.graph.gaussian`
    module.

    """

    _long_name = "Correlation Graph"
    _identifier = "graph-corr"
    _tags = (Tag.SYMMETRICAL, Tag.GRAPH_WEIGHTED, Tag.SIGNED, Tag.ASSOCIATION)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.TOPOLOGY,)

    def __init__(
            self,
            estimator: Literal[
                "Pearson",
                "SpearmanR",
                "KendallTau",
                "Partial"] = "Pearson",
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
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            *, # Partial correlation parameters.
            assume_centered: bool = False):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            ensure_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold,
            self_value=self_value)

        self._method = estimator
        self._nan_policy = nan_policy

        if ensure_adjacency_type in ("binary", "binary-weighted", "absolute"):
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        match estimator:
            case "Pearson":
                self._base_stat = Pearson(
                    dim=self.dim,
                    squared=False,
                    nan_policy=nan_policy)
            case "Spearman":
                self._base_stat = SpearmanR(
                    dim=self.dim,
                    squared=False,
                    nan_policy=nan_policy)
                self._tags = self._get_modified_tags(
                    self._tags,
                    add_tags=Tag.NONLINEAR)
            case "KendallTau":
                self._base_stat = KendallTau(
                    dim=self.dim,
                    squared=False,
                    nan_policy=nan_policy)
                self._tags = self._get_modified_tags(
                    self._tags,
                    add_tags=Tag.NONLINEAR)
            case "Partial":
                self._base_stat = PartialCorrelation(
                    dim=self.dim,
                    squared=False,
                    inversion_method="pseudo",
                    nan_policy=nan_policy,
                    assume_centered=assume_centered)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Return the ``(n_features, n_features)`` correlation adjacency."""

        # Added if compute() is called directly by user.
        return self._base_stat._compute(data)

    

class CovarianceGraph(GraphStatistic):
    """Construct a graph whose edge weights are estimated feature covariances.
    
    
    Parameters
    ----------
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.
    
    """

    _long_name = "Covariance Graph"
    _identifier = "graph-cov"
    _tags = (Tag.SYMMETRICAL, Tag.GRAPH_WEIGHTED, Tag.SIGNED, Tag.ASSOCIATION)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.TOPOLOGY,)

    def __init__(
            self,
            estimator: Literal[
                "EmpiricalCovariance",
                "EllipticEnvelope",
                "GraphicalLassoCV",
                "MinCovDet",
                "OAS",
                "LedoitWolf",
                "ShrunkCovariance"] = "EmpiricalCovariance",
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
            dim: Literal["n", "p"] = "p",
            assume_centered: bool = False,
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            **estimator_kwargs: Any):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            ensure_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold,
            self_value=self_value)

        if ensure_adjacency_type in ("binary", "binary-weighted", "absolute"):
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        self._assume_centered = assume_centered
        self._estimator = estimator
        self._estimator_kwargs = estimator_kwargs
        self._set_base_statistic(estimator)


    def _set_base_statistic(self, estimator: str):
        self._base_stat = Covariance(
            squared=False,
            invert=False,
            assume_centered=self._assume_centered,
            dim=self._dim,
            nan_policy=self._nan_policy,
            estimator=self._estimator,
            estimator_kwargs=self._estimator_kwargs)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Estimate covariance and return it as a feature adjacency matrix."""
        return self._base_stat._compute(data)

   
class RegressionGraph(GraphStatistic):
    """Regress each feature on all others to form a neighbourhood graph.

    Before optional symmetrization, entry ``(i, j)`` is the coefficient for
    feature ``i`` when predicting feature ``j``. This convention makes columns
    regression targets and rows candidate predictors.


    Parameters
    ----------
    
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.

    """

    _long_name = "Regression Graph"
    _identifier = "graph-regression"
    _tags = (Tag.GRAPH_WEIGHTED, Tag.SIGNED, Tag.ASSOCIATION, Tag.PARAMETRIC)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.TOPOLOGY,)

    def __init__(
            self,
            estimator: Literal["ols", "lasso", "ridge", "elastic_net"] = "lasso",
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
            symmetrize: Literal["mean", "maximum", "minimum"] | None = "mean",
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            *,
            max_iter: int = 1000,
            alpha: float = 1.0,
            l1_ratio: float = 0.5):

        self._alpha = argchk.check_float(alpha, minimum=0, arg_name="alpha")
        self._l1_ratio = argchk.check_float(
            l1_ratio,
            minimum=0,
            maximum=1,
            arg_name="l1_ratio")
        self._max_iter = argchk.check_natural_number(max_iter, "max_iter")
        
        self._estimator = estimator
        self._symmetrize = symmetrize
        self._absolute = False

        if symmetrize:
            self._tags = self._get_modified_tags(
                self._tags,
                add_tags=Tag.SYMMETRICAL)

        if ensure_adjacency_type in ("binary", "binary-weighted", "absolute"):
            # Graph adjacency will be absolute if any of these are selected (*1)
            self._absolute = True
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            ensure_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold,
            self_value=self_value)


    def _make_estimator(self):
        """Create the configured scikit-learn regression estimator."""
        if self._estimator == "ols":
            return LinearRegression()
        
        if self._estimator == "lasso":
            return Lasso(alpha=self._alpha, max_iter=self._max_iter)
        
        if self._estimator == "ridge":
            return Ridge(alpha=self._alpha, max_iter=self._max_iter)
        
        return ElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            max_iter=self._max_iter)
    

    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Fit one regression per target feature and return its coefficient graph."""
        matrix = data
        
        if self._adjacency_type:
            from sklearn.preprocessing import StandardScaler
            matrix = StandardScaler().fit_transform(matrix)

        n_features = matrix.shape[1]
        graph = np.zeros((n_features, n_features), dtype=float)
        feature_indices = np.arange(n_features)

        for target in range(n_features):
            predictors = feature_indices != target
            if not predictors.any():
                continue

            model = self._make_estimator().fit(matrix[:, predictors], matrix[:, target])
            graph[predictors, target] = np.asarray(model.coef_).reshape(-1)

        # (*1) If absolute, set adjacency absolute prior to symmetrize.
        if self._absolute:
            graph = np.abs(graph)

        if self._symmetrize == "mean":
            graph = (graph + graph.T) / 2

        elif self._symmetrize == "maximum":
            graph = np.maximum(graph, graph.T)
            
        elif self._symmetrize == "minimum":
            graph = np.minimum(graph, graph.T)

        return graph

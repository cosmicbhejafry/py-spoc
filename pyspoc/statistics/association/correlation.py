from __future__ import annotations

import numpy as np
import scipy as sp
import mne_connectivity as mnec

from scipy.linalg import pinvh
from typing import Union, Literal
from collections.abc import Collection

from pyspoc._tag import Tag, Type, Signature
from pyspoc.statistics.base import Statistic, PairwiseStatistic
from pyspoc.statistics.association.covariance import (
    GraphicalLasso,
    LedoitWolf
)


class Pearson(Statistic):
    # Setting the name internally.
    _long_name = "Pearson's Correlation"

    # Setting the identifier internally.
    _identifier = "corr-pearson"

    _signatures = (Signature.DIMENSIONALITY,)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.SYMMETRICAL, Tag.SIGNED)


    def __init__(self,
                 dim: str = "p",
                 nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
                 squared: bool = False,
                 alternative: Literal["two-sided", "less", "greater"] = "two-sided"):

        # Storing the squared argument.
        self._squared = squared
        self._alternative = alternative

        if squared:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        # Call the base class initialiser with required arguments.
        super().__init__(dim=dim, nan_policy=nan_policy)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        corr = np.corrcoef(data.T)

        # Square results if required.
        if self._squared:
            return np.square(corr)

        # Return value.
        return corr



class SpearmanR(PairwiseStatistic):

    _long_name = "Spearman's Rank Correlation"
    _identifier = "corr-spearmanr"
    _signatures = (Signature.DIMENSIONALITY,)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.SYMMETRICAL, Tag.SIGNED, Tag.NONLINEAR, Tag.NONPARAMETRIC)

    def __init__(
            self,
            dim: str = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            squared: bool = False,
            alternative: Literal["two-sided", "less", "greater"] = "two-sided"):
        
        self._squared = squared
        self._alternative = alternative
        
        if squared:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        super().__init__(dim=dim,
                         nan_policy=nan_policy,
                         is_ordered=False,
                         symmetry_type="exact")


    def _pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        corr = sp.stats.spearmanr(
            x, y, nan_policy=self._nan_policy).statistic

        if self._squared:
            return corr ** 2

        return corr


class KendallTau(PairwiseStatistic):

    _long_name = "Kendall's Tau"
    _identifier = "corr-kendalltau"
    _signatures = (Signature.DIMENSIONALITY,)
    _type = Type.DESCRIPTIVE
    #_tags = (Tag.SYMMETRICAL, Tag.SIGNED, Tag.NONLINEAR, Tag.NONPARAMETRIC)

    def __init__(
            self,
            dim: str = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            squared: bool = False,
            alternative: Literal["two-sided", "less", "greater"] = "two-sided"):
        
        self._squared = squared
        self._nan_policy = nan_policy
        self._alternative = alternative

        if squared:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        super().__init__(dim=dim,
                         nan_policy=nan_policy,
                         is_ordered=False,
                         symmetry_type="exact")


    def _pairwise_compute(self,
                         x: np.ndarray,
                         y: np.ndarray) -> Union[np.ndarray, float]:

        corr = sp.stats.kendalltau(
            x, y, nan_policy=self._nan_policy).correlation

        if self._squared:
            return corr ** 2

        return corr



class PowerEnvelopeCorrelation(Statistic):
    # Setting the name internally.
    _long_name = "Power Envelope Correlation"

    # Setting the identifier internally.
    _identifier = "corr-pec"

    _signatures = (Signature.DIMENSIONALITY,)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.SYMMETRICAL, Tag.SIGNED, Tag.NONLINEAR, Tag.SIGNAL_PROCESSING)


    def __init__(
            self,
            dim: str = "p",
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            orth: bool = False,
            log: bool = False,
            absolute: bool = False):

        self._orth = "pairwise" if orth else False

        # Store the log argument for use in the compute method.
        self._log = log

        # Store the absolute argument for use in the compute method.
        self._absolute = absolute

        if absolute:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        # Call the base class initialiser.
        super().__init__(dim=dim, nan_policy=nan_policy)


    # data: Full dataset with n observations, p variables as a numpy array.
    def _compute(self, data: np.ndarray) -> np.ndarray:

        # Expand so we have a single epoch.
        expanded_data = np.reshape(data, (data.shape[0], data.shape[1], 1)).T

        # Utilizing the envelop_correlation function provided by the MNE-Connectivity library.
        env_corr = mnec.envelope_correlation(
            # Passing through the stored arguments from the initialiser.
            expanded_data,
            orthogonalize=self._orth,
            log=self._log,
            absolute=self._absolute
        )

        corr = env_corr.get_data()

        # Squeezing the result to remove any excess dimensions.
        adj = np.squeeze(corr)

        # Filling self/auto-correlations with 1s.
        np.fill_diagonal(adj, 1)

        # Returning the p by p matrix as numpy array.
        return adj



class PartialCorrelation(Statistic):
    """Estimate pairwise correlations conditional on all remaining variables.

    Partial correlations are obtained by normalizing a precision matrix
    ``Omega`` according to

    ``rho[i, j] = -Omega[i, j] / sqrt(Omega[i, i] * Omega[j, j])``.

    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    dim : {"p", "n"}, default="p"
        Axis whose elements become variables in the returned matrix. ``"p"``
        compares data columns; ``"n"`` compares data rows.
    inversion_method : {"pseudo", "graphical_lasso", "ledoit_wolf"}, default="pseudo"
        Method used to estimate the precision matrix. ``"pseudo"`` applies a
        Hermitian Moore--Penrose pseudoinverse to the empirical covariance.
        ``"graphical_lasso"`` estimates a sparse precision matrix, selecting
        its penalty by cross-validation when ``alpha`` is ``None``.
        ``"ledoit_wolf"`` uses analytically selected covariance shrinkage.
    nan_policy : {"propagate", "raise", "omit"}, default="omit"
        Treatment of missing observations. ``"omit"`` removes complete rows
        containing a NaN after orienting the requested variables; ``"propagate"``
        returns an all-NaN matrix of the appropriate shape.
    assume_centered : bool, default=False
        If true, covariance estimators do not center the supplied data.

        ### Graphical Lasso parameters

    alpha : float or None, optional
        Strictly positive Graphical LASSO penalty. If ``None``, use
        :class:`sklearn.covariance.GraphicalLassoCV`.
    cv : int, default=5
        Number of cross-validation folds used when Graphical LASSO selects
        ``alpha`` automatically.
    max_iter : int, default=100
        Maximum Graphical LASSO optimization iterations.
    tol : float, default=1e-4
        Strictly positive convergence tolerance for Graphical LASSO.

    Notes
    -----
    Variables whose estimated precision diagonal is numerically zero have
    undefined partial correlations. Their off-diagonal entries are returned as
    zero, while all diagonal entries retain the correlation convention of one.
    """

    _long_name = "Partial Correlation"
    _identifier = "corr-partial"

    _signatures = (Signature.DIMENSIONALITY,)
    #_type = Type.DESCRIPTIVE
    _tags = (Tag.SYMMETRICAL, Tag.SIGNED)

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            squared: bool = False,
            nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise",
            assume_centered: bool = False,
            inversion_method: Literal[
                "pseudo",
                "pseudoinverse",
                "graphical_lasso",
                "ledoit_wolf"] = "pseudo",
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

        self._squared = squared
        self._dim = dim
        self._inversion_method = (
            "pseudo"
            if inversion_method in {"pseudo", "pseudoinverse"}
            else inversion_method)
        self._nan_policy = nan_policy
        self._assume_centered = assume_centered
        self._alpha = alpha
        self._cv_alphas = cv_alphas
        self._cv_folds = cv_folds
        self._cv_n_refinements = cv_n_refinements
        self._cv_tol = cv_tol
        self._cv_enet_tol = cv_enet_tol
        self._cv_max_iter = cv_max_iter
        self._cv_mode = cv_mode
        self._cv_eps = cv_eps
        
        if squared:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)

        super().__init__(dim=dim, nan_policy=nan_policy)


    def _compute(self, data: np.ndarray) -> np.ndarray:
        """Estimate and return the requested partial-correlation matrix."""
        precision = self._estimate_precision(data)
        precision_diagonal = np.diag(precision)
        nonnegative_diagonal = np.clip(precision_diagonal, a_min=0.0, a_max=None)
        scale = np.sqrt(np.outer(nonnegative_diagonal, nonnegative_diagonal))

        partial = np.divide(
            -precision,
            scale,
            out=np.zeros_like(precision, dtype=float),
            where=scale > np.finfo(float).eps,
        )

        # Numerical estimators can introduce tiny asymmetries or values just
        # outside the theoretical correlation interval.
        partial = (partial + partial.T) / 2
        partial = np.clip(partial, -1.0, 1.0)
        np.fill_diagonal(partial, 1.0)

        if self._squared:
            partial = np.square(partial)

        return partial
    

    def _estimate_precision(self, data: np.ndarray) -> np.ndarray:
        """Estimate a precision matrix using the configured inversion method."""
        if self._inversion_method == "pseudo":
            centered = data if self._assume_centered else data - np.mean(data, axis=0)
            covariance = np.atleast_2d(centered.T @ centered / data.shape[0])
            return pinvh(covariance)  # pyright: ignore[reportReturnType]

        if self._inversion_method == "ledoit_wolf":
            estimator = LedoitWolf(
                dim=self._dim,
                squared=False,
                nan_policy=self._nan_policy,
                invert=True,
                assume_centered=self._assume_centered)
        else:
            estimator = GraphicalLasso(
                dim=self._dim,
                squared=False,
                nan_policy=self._nan_policy,
                invert=True,
                assume_centered=self._assume_centered,
                alpha=self._alpha,
                cv_alphas=self._cv_alphas,
                cv_folds=self._cv_folds,
                cv_n_refinements=self._cv_n_refinements,
                cv_tol=self._cv_tol,
                cv_enet_tol=self._cv_enet_tol,
                cv_max_iter=self._cv_max_iter,
                cv_mode=self._cv_mode,
                cv_eps=self._cv_eps)
        
        return estimator._compute(data)

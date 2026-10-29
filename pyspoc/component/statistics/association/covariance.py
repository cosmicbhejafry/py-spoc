from __future__ import annotations

import numpy as np

from sklearn import covariance as skcov
from sklearn.covariance import EmpiricalCovariance as EmpCov
from typing import Literal, Any
from collections.abc import Collection

import pyspoc._argchecking as argchk
from pyspoc.component.statistics.base import Statistic
from pyspoc._tag import Tag, Type, Signature
from pyspoc.settings import settings



class Covariance(Statistic):
    """
    Base covariance estimator class used by concrete implementations.
    Information on covariance estimators can be found at: https://scikit-learn.org/stable/modules/covariance.html

    Parameters
    ----------
    squared : bool
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool
        If True, returns the precision matrix instead.
    assume_centered : bool
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    estimator : <Insert description here>
    dim : {"n", "p"}
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;

    Concrete Classes
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant, providing robust estimator of covariance.
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.
    """

    _long_name = "Covariance"
    _identifier = "cov"
    _signatures = (Signature.DIMENSIONALITY,)
    _type = Type.DESCRIPTIVE
    _tags = (Tag.ASSOCIATION, Tag.SYMMETRICAL, Tag.SIGNED)


    def __init__(
            self,
            squared: bool,
            invert: bool,
            assume_centered: bool,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"],
            estimator: Literal[
                "EmpiricalCovariance",
                "EllipticEnvelope",
                "GraphicalLasso",
                "GraphicalLassoCV",
                "MinCovDet",
                "OAS",
                "LedoitWolf",
                "ShrunkCovariance",
                None] = None,
            **estimator_kwargs: Any):
        
        if squared:
            self._tags = self._get_modified_tags(
                self._tags,
                remove_tags=Tag.SIGNED)
            self._identifier += "-sq"

        self._is_squared = squared
        self._invert = invert
        self._assume_centered = assume_centered
        self._estimator = type(self).__qualname__ if estimator is None else estimator
        self._estimator_kwargs = estimator_kwargs
        super().__init__(dim=dim, nan_policy=nan_policy)


    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray:
        """
        Compute the covariance matrix for the given point cloud data matrix.

        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional point cloud data matrix with observations along the rows
            and features along the columns.

        Returns
        -------
        np.ndarray[int, int]
            Estimated covariance matrix.
        """
        cov_obj = self._fit(data)
        result = cov_obj.covariance_ if not self._invert else cov_obj.precision_

        if self._is_squared:
            result = np.square(result)

        return result
    

    def _fit(self, data: np.ndarray) -> EmpCov:
        cov_class: EmpCov = getattr(skcov, self._estimator)(
            store_precision=True,
            assume_centered=self._assume_centered,
            **self._estimator_kwargs)
        cov_obj = cov_class.fit(data)
        return cov_obj



class EmpiricalCovariance(Covariance):
    """
    Maximum likelihood covariance estimator.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
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
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;
        

    See Also
    --------
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant (robust estimator of covariance).
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.

    """

    _name = "Empirical Covariance"
    _identifier = "cov-empirical"

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            squared=squared,
            invert=invert,
            assume_centered=assume_centered,
            dim=dim,
            nan_policy=nan_policy)



class EllipticEnvelope(Covariance):
    """
    Calculates robust covariance estimate for outlier detection in Gaussian distributed datasets
    
    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    support_fraction : float, default=None, requires: >0, <1
        The proportion of points to be included in the support of the raw MCD estimate. If `None`,
        the minimum value of support_fraction will be used within the algorithm:
        `(n_samples + n_features + 1) / 2 * n_samples`.
    contamination : float, default=0.1, requires: >0, <=1
        The amount of contamination of the data set, i.e. the proportion of outliers in the data
        set.
    random_seed : int or None, default=None
        Determines the pseudo random number generator for shuffling the data. Pass an int for
        specific reproducible needs or `None` to use the library-wide setting.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;

    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant (robust estimator of covariance).
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.

        
    Notes
    ----------
    Outlier detection from covariance estimation may break or not perform well in
    high-dimensional settings. In particular, one will always take care to work with
    `n_samples > n_features ** 2`.


    References
    ----------
    .. [1] Rousseeuw, P.J., Van Driessen, K. “A fast algorithm for the minimum covariance
        determinant estimator” Technometrics 41(3), 212 (1999)

    """

    _name = "Elliptic Envelope"
    _identifier = "cov-elliptic"

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            support_fraction: float | None = None,
            contamination: float = 0.1,
            random_seed: int | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        support_fraction = (argchk.check_float(
                support_fraction,
                exclusive_minimum=0,
                exclusive_maximum=1,
                arg_name="support_fraction")
            if support_fraction is not None
            else None)
        
        contamination = argchk.check_float(
            contamination,
            exclusive_minimum=0,
            maximum=0.5,
            arg_name="contamination")
        
        super().__init__(
            dim=dim,
            squared=squared,
            nan_policy=nan_policy,
            invert=invert,
            assume_centered=assume_centered,
            support_fraction=support_fraction,
            contamination=contamination,
            random_state=self.random_seed
        )

        self._tags = self._get_modified_tags(
            self._tags,
            add_tags=[Tag.PARAMETRIC, Tag.DISTRIBUTION])



class GraphicalLasso(Covariance):
    """Estimate pairwise correlations conditional on all remaining variables.

    Partial correlations are obtained by normalizing a precision matrix
    ``Omega`` according to

    ``rho[i, j] = -Omega[i, j] / sqrt(Omega[i, i] * Omega[j, j])``.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    alpha : float or None, optional or requires: >0
        Graphical LASSO penalty coefficient. If ``None``, use
        :class:`sklearn.covariance.GraphicalLassoCV`.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;

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


    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant (robust estimator of covariance).
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.


    Notes
    -----
    Search for the optimal penalization parameter `alpha` is done on an iteratively refined grid:
    first the cross-validated scores on a grid are computed, then a new refined grid is centered
    around the maximum, and so on.

    One of the challenges which is faced here is that the solvers can fail to converge to a
    well-conditioned estimate. The corresponding values of alpha then come out as missing values,
    but the optimum may be close to these missing values.

    In fit, once the best parameter alpha is found through cross-validation, the model is fit again
    using the entire training set.


    References
    ------


    """

    _long_name = "Graphical Lasso"
    _identifier = "cov-glasso"
    _tags = (Tag.ASSOCIATION, Tag.SYMMETRICAL, Tag.SIGNED, Tag.PARAMETRIC)

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            dim: Literal["p", "n"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
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
        self._nan_policy = nan_policy
        self._assume_centered = assume_centered
        self._alpha = alpha

        if alpha is None: # Cross-validation used.
            self._cv_folds = argchk.check_integer(cv_folds, minimum=2, arg_name="cv_folds")
            cv_n_refinements = argchk.check_natural_number(cv_n_refinements, "cv_n_refinements")
            cv_max_iter = argchk.check_natural_number(cv_max_iter, arg_name="cv_max_iter")
            cv_tol = argchk.check_float(cv_tol, exclusive_minimum=0, arg_name="cv_tol")
            cv_enet_tol = argchk.check_float(
                cv_enet_tol, exclusive_minimum=0, arg_name="cv_enet_tol")
            cv_eps = argchk.check_float(cv_eps, exclusive_minimum=0, arg_name="cv_eps")
            cv_alphas = (
                argchk.check_natural_number(cv_alphas, "cv_alphas")
                if isinstance(cv_alphas, int)
                else tuple(
                    argchk.check_float(alpha, exclusive_minimum=0, arg_name="cv_alphas")
                    for alpha in cv_alphas)
                )
            super().__init__(
                dim=dim,
                squared=squared,
                nan_policy=nan_policy,
                invert=invert,
                estimator="GraphicalLassoCV",
                assume_centered=assume_centered,
                alphas=cv_alphas,
                cv=self._cv_folds,
                n_refinements=cv_n_refinements,
                max_iter=cv_max_iter,
                mode=cv_mode,
                tol=cv_tol,
                enet_tol=cv_enet_tol,
                eps=cv_eps,
                n_jobs = settings.current.max_worker_threads)
        else:
            alpha = argchk.check_float(alpha, exclusive_minimum=0, arg_name="alpha")
            super().__init__(
                dim=dim,
                squared=squared,
                nan_policy=nan_policy,
                invert=invert,
                estimator="GraphicalLasso",
                assume_centered=assume_centered,
                alpha=alpha)
    

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray:
        """Estimate and return the requested partial-correlation matrix."""
        
        n = data.shape[0] if self._dim == "p" else data.shape[1]
        
        if (self._alpha is None
            and self._cv_folds > n):
            raise ValueError(
                "cv_folds must not exceed the number of observations when using"
                "cross-validated Graphical LASSO.")

        return super()._compute(data)
    


class OAS(Covariance):

    """
    Oracle Approximating Shrinkage covariance estimator with shrinkage.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
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
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;


    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant, providing robust estimator of covariance.
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.


    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.


    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`

    """

    _long_name = "Oracle Approximating Shrinkage"
    _identifier = "cov-oas"
    _tags = (Tag.ASSOCIATION, Tag.SYMMETRICAL, Tag.SIGNED, Tag.PARAMETRIC)

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            dim=dim,
            squared=squared,
            nan_policy=nan_policy,
            invert=invert,
            assume_centered=assume_centered)



class LedoitWolf(Covariance):

    """
    Ledoit-Wolf covariance estimator with shrinkage, where the shrinkage coefficient is computed
    using O. Ledoit and M. Wolf's formula as described in [1]_.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    block_size : int, default=1000, requires:>=1
        Size of blocks into which the covariance matrix will be split during its Ledoit-Wolf
        estimation. This is purely a memory optimization and does not affect results.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;

    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant, providing robust estimator of covariance.
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.

    
    Notes
    --------
    The regularised covariance is:
    
    `(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)`

    where `mu = trace(cov) / n_features` and shrinkage is given by the Ledoit and Wolf formula
    (see References)


    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`

    """

    _long_name = "Ledoit-Wolf Shrinkage"
    _identifier = "cov-lw"

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            block_size: int = 1000,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            dim=dim,
            squared=squared,
            nan_policy=nan_policy,
            invert=invert,
            assume_centered=assume_centered,
            block_size=block_size)



class MinCovDet(Covariance):

    """
    Minimum Covariance Determinant, providing robust estimator of covariance.

    The Minimum Covariance Determinant covariance estimator is to be applied on Gaussian-distributed
    data, but could still be relevant on data drawn from a unimodal, symmetric distribution.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    support_fraction : float, default=None, requires: >0, <1
        The proportion of points to be included in the support of the raw MCD estimate. If `None`,
        the minimum value of support_fraction will be used within the algorithm:
        `(n_samples + n_features + 1) / 2 * n_samples`.
    random_seed : int or None, default=None
        Determines the pseudo random number generator for shuffling the data. Pass an int for
        specific reproducible needs or `None` to use the library-wide setting.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN.
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN;
        

    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`ShrunkCovariance` : Covariance estimator with shrinkage.

    
    Notes
    -------
    It is not meant to be used with multi-modal data (the algorithm used to fit a MinCovDet object
    is likely to fail in such a case). One should consider projection pursuit methods to deal with
    multi-modal datasets.


    References
    ----------
    .. [1] P. J. Rousseeuw. Least median of squares regression. J. Am Stat Ass, 79:871, 1984.
    .. [2] A Fast Algorithm for the Minimum Covariance Determinant Estimator, 1999,
        American Statistical Association and the American Society for Quality, TECHNOMETRICS
    .. [3] R. W. Butler, P. L. Davies and M. Jhun, Asymptotics For The Minimum Covariance
        Determinant Estimator, The Annals of Statistics, 1993, Vol. 21, No. 3, 1385-1400
    """

    _long_name = "Minimum Covariance Determinant"
    _identifier = "cov-mcd"

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            support_fraction: float | None = None,
            random_seed: int | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        support_fraction = argchk.check_float(
            support_fraction,
            exclusive_minimum=0,
            exclusive_maximum=1,
            arg_name="support_fraction")

        super().__init__(
            dim=dim,
            squared=squared,
            nan_policy=nan_policy,
            invert=invert,
            assume_centered=assume_centered,
            support_fraction=support_fraction,
            random_seed=self.random_seed)


class ShrunkCovariance(Covariance):
    """
    Covariance estimator with shrinkage.

    More information can be found at:
        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html


    Parameters
    ----------
    squared : bool, default=False
        Square every partial-correlation coefficient, discarding its sign.
    invert : bool, default=False
        If True, returns the precision matrix instead.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly zero.
        If False (default), data will be centered before computation.
    shrinkage : float, default=0.1, requires: >=0, <=1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.


    See Also
    --------
    :class:`EmpiricalCovariance` : Maximum likelihood covariance estimator.
    :class:`EllipticEnvelope` : An object for detecting outliers in a Gaussian distributed dataset.
    :class:`GraphicalLasso` : Sparse inverse covariance estimation with an l1-penalized estimator.
    :class:`OAS` : Oracle Approximating Shrinkage covariance estimator with shrinkage.
    :class:`LedoitWolf` : Ledoit-Wolf covariance estimator with shrinkage.
    :class:`MinCovDet` : Minimum Covariance Determinant, providing robust estimator of covariance.


    Notes
    --------
    The regularised covariance is:
    
    `(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)`

    where `mu = trace(cov) / n_features` and shrinkage is given by constructor argument `shrinkage`
    """

    _long_name = "Shrunk Covariance"
    _identifier = "cov-sc"

    def __init__(
            self,
            squared: bool = False,
            invert: bool = False,
            assume_centered: bool = False,
            shrinkage: float = 0.1,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        shrinkage = argchk.check_float(
            shrinkage,
            minimum=0,
            maximum=1,
            arg_name="shrinkage")

        super().__init__(
            dim=dim,
            squared=squared,
            nan_policy=nan_policy,
            invert=invert,
            assume_centered=assume_centered,
            shrinkage=shrinkage)

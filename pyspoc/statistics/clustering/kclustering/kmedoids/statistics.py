"""Concrete Statistics derived from fitted K-Medoids cluster centres."""
from __future__ import annotations

from typing import Literal
from numpy import ndarray

from pyspoc.settings import settings
from pyspoc._tag import Type, Tag
from ._base import KMedoidsStatistic, KMedoidsEstimator
from .._mixin import KClusteringSimilarityMixin



class KMedoidsSimilarity(KMedoidsStatistic, KClusteringSimilarityMixin):
    """Calculate pairwise similarities between fitted K-Medoids medoids.

    Each row and column of the result represents one fitted cluster. The
    resulting matrix is symmetric, has a unit diagonal, and contains
    similarities in the interval ``[0, 1]``.

    Parameters
    ----------
    k : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations for a single K-Medoids run.
    random_seed : int or `None`, optional
        Per-estimator random-seed override. If ``None``, use the active
        library-wide seed.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    metric : string, default="euclidean"
        What distance metric to use. See :obj:`AVAILABLE_DISTANCES` for a list of
        available distance metrics.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    kernel : {"rbf", "mahalanobis_rbf", "laplacian", \
                "inverse_distance", "mahalanobis_inverse_distance", \
                "cosine", "correlation"}, default="rbf"
        Similarity measure applied to fitted cluster centres.
    gamma : float or None, optional
        Strictly positive similarity decay coefficient. If ``None``, use
        ``1 / n_features`` when computation begins. Gamma is ignored by
        cosine and correlation similarities.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=:attr:`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.


    Raises
    ------
    TypeError
        If a runtime-checked argument has an incompatible type.
    ValueError
        If ``k`` is less than two or ``gamma`` is not strictly positive.

    
    Notes
    -----
    RBF, Laplacian, and inverse-distance similarities measure spatial
    proximity. Cosine and correlation similarities instead compare the
    direction or feature profile of centres.

    Singular covariance matrices are supported by discarding covariance
    eigenvalues below a numerical tolerance. Consequently, Mahalanobis
    similarities ignore directions containing no independently measurable
    variance.
    """

    _long_name = "KMedoids - Cluster Similarity"
    _identifier = "kmedoids-cs"
    _type = Type.DESCRIPTIVE
    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING, Tag.SYMMETRICAL)


    def __init__(
            self,
            k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal["k-medoids++", "random", "heuristic", "build"] = "heuristic",
            metric: Literal["euclidean",
                "nan_euclidean",
                "manhattan",
                "cosine",
                "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "dice",
                "hamming",
                "jaccard",
                "haversine",
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalsneath",
                "sqeuclidean",
                "yule"] = "euclidean",
            algorithm: Literal["alternate", "pam"] = "alternate",
            kernel: Literal[
                "rbf",
                "mahalanobis_rbf",
                "laplacian",
                "inverse_distance",
                "mahalanobis_inverse_distance",
                "cosine",
                "correlation",
            ] = "rbf",
            gamma: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        KMedoidsStatistic.__init__(
            self=self,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            initializer=initializer,
            metric=metric,
            algorithm=algorithm,
            dim=dim,
            nan_policy=nan_policy)
        KClusteringSimilarityMixin.__init__(
            self=self,
            kernel=kernel,
            gamma=gamma)
    
    def _get_result(self, data: ndarray, fitted_estimator: KMedoidsEstimator) -> ndarray | float:
        return KClusteringSimilarityMixin._get_result(self, data, fitted_estimator)

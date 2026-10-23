"""Concrete Statistics derived from fitted K-Means cluster centres."""

from __future__ import annotations

import numpy as np
from typing import Literal

from pyspoc.settings import settings
from pyspoc._tag import Tag, Type
from ._base import KMeansEstimator
from ._base import KMeansStatistic
from .._mixin import KClusteringSimilarityMixin



class KMeansSimilarity(KMeansStatistic, KClusteringSimilarityMixin):
    """Calculate pairwise similarities between fitted K-Means centres.

    Each row and column of the result represents one fitted cluster. The
    resulting matrix is symmetric, has a unit diagonal, and contains
    similarities in the interval ``[0, 1]``.

    Parameters
    ----------
    k : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used to initialize cluster centroids.
        
        "k-means++": selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        "random": choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Which algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan"
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).
    kernel : {"rbf", "mahalanobis_rbf", "laplacian", \
              "inverse_distance", "mahalanobis_inverse_distance", \
              "cosine", "correlation"}, default="rbf"
        Similarity measure applied to fitted cluster centres. Mahalanobis
        variants first whiten centres using the pseudo-inverse square root of
        the empirical feature covariance matrix.
    gamma : float or None, optional
        Strictly positive similarity decay coefficient. If ``None``, use
        ``1 / n_features`` for the dataset supplied to :meth:`compute`.
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

    _name = "KMeans - Cluster Similarity"
    _identifier = "kmeans-cs"
    _type = Type.DESCRIPTIVE
    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING, Tag.SYMMETRICAL)

    def __init__(
            self,
            k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            tol: float = 0.0001,
            initializer: Literal["k-means++", "random"] = "k-means++",
            algorithm: Literal["lloyd", "elkan"] = "lloyd",
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

        KMeansStatistic.__init__(
            self=self,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            tol=tol,
            initializer=initializer,
            algorithm=algorithm,
            dim=dim,
            nan_policy=nan_policy)
        KClusteringSimilarityMixin.__init__(
            self=self,
            kernel=kernel,
            gamma=gamma)

    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        
        return KClusteringSimilarityMixin._get_result(self, data, fitted_estimator)

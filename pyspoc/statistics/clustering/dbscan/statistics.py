"""Concrete Statistics derived from fitted DBSCAN cluster centres."""
from __future__ import annotations

import numpy as np

from typing import Literal

from pyspoc.settings import settings
from pyspoc.types import DistanceMetric
from pyspoc._tag import Tag, Type
from ._base import DBScanEstimator, DBScanStatistic
from .._mixin import ClusteringSimilarityMixin


# TODO: This should be merged with the KClustering Similarity statistics
# as the only difference is line 132 (KCluster models have property cluster_centers_
# while DBScan has components_). It may be worth doing this for other statistics
# between the clustering methods as well.



class DBScanSimilarity(
    DBScanStatistic,
    ClusteringSimilarityMixin):
    """Calculate pairwise similarities between fitted DBScan.

    Each row and column of the result represents one fitted cluster. The
    resulting matrix is symmetric, has a unit diagonal, and contains
    similarities in the interval ``[0, 1]``.


    Parameters
    ----------
    eps : float, requires: >0
        The maximum distance between two samples for one to be considered as in the neighborhood of
        the other. This is not a maximum bound on the distances of points within a cluster. This is
        the most important DBSCAN parameter to choose appropriately for your data set and distance
        function. Smaller values generally lead to more clusters.
    min_samples : int, requires: >=2, <=`n` with `n` the number of observations in the data matrix \
        provided

        The number of samples (or total weight) in a neighborhood for a point to be considered as a
        core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN
        will find denser clusters, whereas if it is set to a lower value, the found clusters will be
        more sparse.
    n_results : int, requires: >=1
        The number of fits to perform over the sweep. `min_eps` and `max_eps` determine the range,
        while `n_results` will determine the number of fits within that range.
    min_samples : int, default=5, requires: >=2, <=`n` with `n` the number of observations in the \
        data matrix provided

        The number of samples (or total weight) in a neighborhood for a point to be considered as a
        core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN
        will find denser clusters, whereas if it is set to a lower value, the found clusters will be
        more sparse.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        The algorithm to be used by the NearestNeighbors module to compute pointwise distances and
        find nearest neighbors. `"auto"` will attempt to decide the most appropriate algorithm based
        on the values passed to fit method. See
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
        documentation for details.
    leaf_size: int, default=30, requires: >=1
        Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and
        query, as well as the memory required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used in the KMedoids algorithm's
        determination of selected medoids. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    minkowski_p : float or `None`, default=None, required if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    n_jobs : int, default=`settings.current.max_worker_threads`
        The number of parallel jobs to run. Non-positive values will use the library-wide setting
        `settings.current.max_worker_threads`.
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
    nan_policy : {"raise", "row-omit", "col-omit"}, default=`settings.current.data_nan_policy`
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

    _name = "DBScan - Cluster Similarity"
    _identifier = "dbscan-cs"
    _type = Type.DESCRIPTIVE
    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING, Tag.SYMMETRICAL)


    def __init__(
            self,
            eps: float,
            min_samples: int = 5,
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"] = "auto",
            leaf_size: int = 30,
            metric: DistanceMetric = "euclidean",
            minkowski_p: float | None = None,
            n_jobs: int = settings.current.max_worker_threads,
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

        DBScanStatistic.__init__(
            self=self,
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs,
            dim=dim,
            nan_policy=nan_policy)
        ClusteringSimilarityMixin.__init__(
            self=self,
            kernel=kernel,
            gamma=gamma)

    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: DBScanEstimator
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        
        return ClusteringSimilarityMixin._get_result(self, data, fitted_estimator)

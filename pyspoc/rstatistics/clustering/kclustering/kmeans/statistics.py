import numpy as np

from typing import Literal

from pyspoc.settings import settings
from pyspoc.statistics.clustering.kclustering.kmeans._base import KMeansEstimator
from ._base import KMeansSweepStatistic
from .._mixin import (
    KClusteringErrorsMixin,
    KClusteringSILScoresMixin,
    KClusteringDispersionSummaryMixin)



class KMeansErrors(
    KMeansSweepStatistic,
    KClusteringErrorsMixin):
    """
    Returns a series of errors from KMeans fits with a sweep of `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used to initialize cluster centroids.
        
        `"k-means++"`: selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        `"random"`: choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Which algorithm to use. The classical EM-style algorithm is `"lloyd"`. The `"elkan"`
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).
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
        If a numeric argument or component is outside its accepted range.

    """

    _long_name = "KMeans - Error Distribution"
    _identifier = "kmeans-error"


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator) -> float:
                
        return KClusteringErrorsMixin._get_result(
            self, data, fitted_estimator)


class KMeansSILScores(
    KMeansSweepStatistic,
    KClusteringSILScoresMixin):
    """
    Returns a series of Silhouette scores from KMeans fits with a sweep of
    `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used to initialize cluster centroids.
        
        `"k-means++"`: selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        `"random"`: choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Which algorithm to use. The classical EM-style algorithm is `"lloyd"`. The `"elkan"`
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).
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
        If a numeric argument or component is outside its accepted range.

    """

    _long_name = "KMeans - Silhouette Score Distribution"
    _identifier = "kmeans-sil"

    # TODO: Add metric to docstring.
    
    def __init__(
            self,
            min_k: int,
            max_k: int,
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
            max_iter: int = 300,
            random_seed: int | None = None,
            algorithm: Literal["lloyd", "elkan"] = "lloyd",
            initializer: Literal["k-means++", "random"] = "k-means++",
            tol: float = 0.0001,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
    
            KMeansSweepStatistic.__init__(
                self,
                min_k=min_k,
                max_k=max_k,
                max_iter=max_iter,
                random_seed=random_seed,
                algorithm=algorithm,
                initializer=initializer,
                tol=tol,
                dim=dim,
                nan_policy=nan_policy)
    
            KClusteringSILScoresMixin.__init__(
                self,
                metric=metric)


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator) -> float:
                
        return KClusteringSILScoresMixin._get_result(
            self, data, fitted_estimator)


class KMeansDispersionSummaries(
    KMeansSweepStatistic,
    KClusteringDispersionSummaryMixin):
    """
    Returns a series of cluster dispersion metrics from KMeans fits with a sweep of
    `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used to initialize cluster centroids.
        
        `"k-means++"`: selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        `"random"`: choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Which algorithm to use. The classical EM-style algorithm is `"lloyd"`. The `"elkan"`
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).
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
        If a numeric argument or component is outside its accepted range.

    """

    _long_name = "KMeans - Dispersion Summary Distribution"
    _identifier = "kmeans-dispersion"

    # TODO: Need to add summary_type and distance_type to docstring.

    def __init__(
        self,
        min_k: int,
        max_k: int,
        summary_type: Literal[
                "mean",
                "median",
                "min",
                "max",
                "stdev",
                "iqr",
                "range"],
        distance_type: Literal[
            "ptc", # Point-to-cluster distance summary
            "ctc", # Cluster-to-centroid distance summary
            "ctm", # Cluster-to-medoid distance summary
            "ptc-ctc-ratio", # Point-to-cluster cluster-to-centroid distance summary ratio
            "ptc-ctm-ratio"], # Point-to-cluster cluster-to-medoid distance summary ratios
        max_iter: int = 300,
        random_seed: int | None = None,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
        initializer: Literal["k-means++", "random"] = "k-means++",
        tol: float = 0.0001,
        dim: Literal["n", "p"] = "p",
        nan_policy: Literal[
            "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        KMeansSweepStatistic.__init__(
            self,
            min_k=min_k,
            max_k=max_k,
            max_iter=max_iter,
            random_seed=random_seed,
            algorithm=algorithm,
            initializer=initializer,
            tol=tol,
            dim=dim,
            nan_policy=nan_policy)

        KClusteringDispersionSummaryMixin.__init__(
            self,
            summary_type=summary_type,
            distance_type=distance_type)


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator
    ) -> float | np.floating:
                
        return KClusteringDispersionSummaryMixin._get_result(
            self, data, fitted_estimator)

import numpy as np

from typing import Literal

from pyspoc.settings import settings
from pyspoc.types import DistanceMetric, SummaryType
from pyspoc.statistics.clustering.kclustering.kmeans._base import KMeansEstimator
from ._base import KMeansSweepStatistic
from ..._mixin import (
    ClusteringSILScoresMixin,
    ClusteringDispersionSummaryMixin)

from .._mixin import KClusteringInertiaMixin



class KMeansInertia(
    KMeansSweepStatistic,
    KClusteringInertiaMixin):
    """
    Returns a series of inertia scores from KMeans fits with a sweep of `k` parameters.
    Inertia is the sum of distances between points and their assigned cluster centroids.
    Minkowski distance is used, which with `p=2` produces Euclidean distances that are the
    standard for K-Means inertia.

    
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
    tol : float, default=0.0001, requires>0
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
        of two consecutive iterations to declare convergence.
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
        If a numeric argument or component is outside its accepted range,
        or if required components are empty.
    """

    _long_name = "KMeans - Inertia Distribution"
    _identifier = "kmeans-inertia"


    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal["k-means++", "random"] = "k-means++",
            algorithm: Literal["lloyd", "elkan"] = "lloyd",
            tol: float = 0.0001,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

            super().__init__(
                min_k=min_k,
                max_k=max_k,
                max_iter=max_iter,
                random_seed=random_seed,
                algorithm=algorithm,
                initializer=initializer,
                tol=tol,
                dim=dim,
                nan_policy=nan_policy)

    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator) -> float | np.floating:
                
        return KClusteringInertiaMixin._get_result(
            self, data, fitted_estimator)
    


class KMeansSILScores(
    KMeansSweepStatistic,
    ClusteringSILScoresMixin):
    """
    Returns a series of Silhouette scores from KMeans fits with a sweep of `k` parameters.


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
    tol : float, default=0.0001, requires>0
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
        of two consecutive iterations to declare convergence.
    silhouette_metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used to compute the Silhouette score.
        See :class:`DistanceMetric` for the list of supported distance metrics and their formulae.
    silhouette_minkowski_p : float or `None`, default=`None", requires: >0 \
        if `silhouette_metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance. Note that for `p<1`, Minkowski distance
        is no longer a true distance metric as it can violate the triangle inequality.
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
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal["k-means++", "random"] = "k-means++",
            algorithm: Literal["lloyd", "elkan"] = "lloyd",
            tol: float = 0.0001,
            silhouette_metric: DistanceMetric = "euclidean",
            silhouette_minkowski_p: float | None = None,
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
    
            ClusteringSILScoresMixin.__init__(
                self,
                silhouette_metric=silhouette_metric,
                silhouette_minkowski_p=silhouette_minkowski_p)


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator) -> float | np.floating:
                
        return ClusteringSILScoresMixin._get_result(
            self, data, fitted_estimator)


class KMeansDispersionSummaries(
    KMeansSweepStatistic,
    ClusteringDispersionSummaryMixin):
    """
    Returns a series of dispersion measures from KMedoids fits with a sweep of `k` parameters.
    Distances between points are computed using Minkowski distance, with Minkowski's `p` available
    as a constructor parameter. Multiple dispersion measures are available in addition to ratios
    between different measures.


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
    tol : float, default=0.0001, requires>0
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
        of two consecutive iterations to declare convergence.
    dispersion_measure : {"within-cluster" "across-cluster", "within-across-ratio"}, \
            default="within-cluster"
        The type of dispersion measure to compute.
        - `"within-cluster"` : Within cluster distance summary.
        - `"across-cluster"` : Across cluster distance summary.
        - `"within-across-ratio"` : Within vs. across distance summary ratio.
    dispersion_minkowski_p : float, default=2, requires: >0
        Minkowski _p_ parameter used for Minkowski distance in dispersion calculation, with the
        default of `2` producing Euclidean distances. Note that for `p<1`, Minkowski distance is
        no longer a true distance metric as it can violate the triangle inequality.
    inner_summary : SummaryType, default="mean"
        The method of summarizing accumulated distances. See :class:`SummaryType` for the list
        of supported summaries.
    outer_summary : SummaryType, default="mean"
        The method of further summarizing distance summaries. See :class:`SummaryType` for the list
        of supported summaries.
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


    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal["k-means++", "random"] = "k-means++",
            algorithm: Literal["lloyd", "elkan"] = "lloyd",
            tol: float = 0.0001,
            dispersion_measure: Literal[
                "within-cluster", # Within cluster distance summary
                "across-cluster", # Across cluster distance summary
                "within-across-ratio" # Within vs. across cluster distance summary ratio
            ] = "within-cluster",
            dispersion_minkowski_p: float = 1,
            inner_summary: SummaryType = "mean",
            outer_summary: SummaryType = "mean",
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

        ClusteringDispersionSummaryMixin.__init__(
            self,
            inner_summary=inner_summary,
            outer_summary=outer_summary,
            dispersion_minkowski_p=dispersion_minkowski_p)

        self._dispersion_measure = dispersion_measure


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMeansEstimator
    ) -> float | np.floating:
                
        return ClusteringDispersionSummaryMixin._get_result(
            self,
            data=data,
            fitted_estimator=fitted_estimator,
            dispersion_measure=self._dispersion_measure)

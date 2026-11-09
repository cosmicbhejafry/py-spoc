import numpy as np

from typing import Literal

from pyspoc.types import DistanceMetric, SummaryType
from pyspoc.settings import settings
from pyspoc.statistics.clustering.dbscan._base import DBScanEstimator
from pyspoc._utils.numerical import summarize
from ._base import DBScanSweepStatistic
from .._mixin import (
    ClusteringSILScoresMixin,
    ClusteringDispersionSummaryMixin)


class DBScanNoiseRatio(
    DBScanSweepStatistic):
    """
    Returns a series of ratios of the number of noise points to total points from DBScan fits
    with a sweep over the `eps` parameter.


    Parameters
    ----------
    min_eps : float, requires: >0
        Minimum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
    max_eps : float, requires: >=`min_eps`
        Maximum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
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
    minkowski_p : float or `None`, default=None, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    n_jobs : int, default=`settings.current.max_worker_threads`
        The number of parallel jobs to run. Non-positive values will use the library-wide setting
        `settings.current.max_worker_threads`.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=`settings.current.data_nan_policy`
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.

        
    Notes
    --------
    `eps` defines the maximum distance between two samples for one to be considered as in the
    neighborhood of the other. This is not a maximum bound on the distances of points within a
    cluster. This is the most important DBSCAN parameter to choose appropriately for your data set
    and distance function. Smaller values generally lead to more clusters.


    Raises
    ---------
    TypeError
        If a runtime-checked argument has an incompatible type.
    ValueError
        If a numeric argument or component is outside its accepted range.

    """

    _long_name = "DBScan - Inertia Distribution"
    _identifier = "dbscan-inertia"


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: DBScanEstimator) -> float | np.floating:
        
        labels = fitted_estimator._predict(data)
        return (labels == -1).sum() / labels.shape[0]



class DBScanSILScores(
    DBScanSweepStatistic,
    ClusteringSILScoresMixin):
    """
    Returns a series of Silhouette scores from DBScan fits with a sweep over the `eps` parameter.


    Parameters
    ----------
    min_eps : float, requires: >0
        Minimum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
    max_eps : float, requires: >=`min_eps`
        Maximum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
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
    minkowski_p : float or `None`, default=None, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    n_jobs : int, default=`settings.current.max_worker_threads`
        The number of parallel jobs to run. Non-positive values will use the library-wide setting
        `settings.current.max_worker_threads`.
    silhouette_metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used to compute the Silhouette score.
        See :class:`DistanceMetric` for the list of supported distance metrics and their formulae.
    silhouette_minkowski_p : float or `None`, default=`None", requires: >0 if `metric = "minkowski"`
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

        
    Notes
    --------
    `eps` defines the maximum distance between two samples for one to be considered as in the
    neighborhood of the other. This is not a maximum bound on the distances of points within a
    cluster. This is the most important DBSCAN parameter to choose appropriately for your data set
    and distance function. Smaller values generally lead to more clusters.


    Raises
    ---------
    TypeError
        If a runtime-checked argument has an incompatible type.
    ValueError
        If a numeric argument or component is outside its accepted range.
    """

    _long_name = "DBScan - Silhouette Score Distribution"
    _identifier = "dbscan-sil"


    def __init__(
            self,
            min_eps: float,
            max_eps: float,
            n_results: int,
            min_samples: int,
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            metric: DistanceMetric,
            minkowski_p: float | None,
            n_jobs: int = settings.current.max_worker_threads,
            silhouette_metric: DistanceMetric = "euclidean",
            silhouette_minkowski_p: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        DBScanSweepStatistic.__init__(
            self,
            min_eps=min_eps,
            max_eps=max_eps,
            n_results=n_results,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs,
            dim=dim,
            nan_policy=nan_policy)

        ClusteringSILScoresMixin.__init__(
            self,
            silhouette_metric=silhouette_metric,
            silhouette_minkowski_p=silhouette_minkowski_p)


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: DBScanEstimator) -> float | np.floating:
        
        clusters = fitted_estimator._predict(data)
        signal_data, signal_clusters = fitted_estimator._get_signal_data(data, clusters)
        
        return ClusteringSILScoresMixin._get_result(
            self,
            data=signal_data,
            fitted_estimator=fitted_estimator,
            clusters=signal_clusters)
    


class DBScanDispersionSummaries(
    DBScanSweepStatistic,
    ClusteringDispersionSummaryMixin):

    """
    Returns a series of dispersion measures from DBSCAN fits with a sweep over the `eps` parameter.
    Distances between points are computed using Minkowski distance, with Minkowski's `p` available
    as a constructor parameter. Multiple dispersion measures are available in addition to ratios
    between different measures.


    Parameters
    ----------
    min_eps : float, requires: >0
        Minimum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
    max_eps : float, requires: >=`min_eps`
        Maximum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
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
    minkowski_p : float or `None`, default=None, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    n_jobs : int, default=`settings.current.max_worker_threads`
        The number of parallel jobs to run. Non-positive values will use the library-wide setting
        `settings.current.max_worker_threads`.
    dispersion_measure : {"within-cluster" "across-cluster", "within-across-ratio", "noise" \
        "signal", "noise-signal-ratio"}, default="within-cluster"
        The type of dispersion measure to compute.
        - `"within-cluster"` : Within cluster distance summary.
        - `"across-cluster"` : Across cluster distance summary.
        - `"within-across-ratio"` : Within vs. across distance summary ratio.
        - `"signal"` : Distance summary for all clustered points.
        - `"noise"` : Distance summary for non-clustered noise points.
        - `"signal-noise-ratio"` : Signal vs. noise distance summary ratio.
    dispersion_minkowski_p : float, default=2, requires: >0
        Minkowski _p_ parameter used for Minkowski distance in dispersion calculation, with the
        default of `1` producing Euclidean distances. Note that for `p<1`, Minkowski distance is
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
        If a numeric argument or component is outside its accepted range,
        or if ``components`` is empty.

        """

    _long_name = "DBScan - Dispersion Summary Distribution"
    _identifier = "dbscan-dispersion"

    def __init__(
        self,
        min_eps: float,
        max_eps: float,
        n_results: int,
        min_samples: int,
        algorithm: Literal[
            "auto", "ball_tree", "kd_tree", "brute"],
        leaf_size: int,
        metric: DistanceMetric,
        minkowski_p: float | None,
        n_jobs: int = settings.current.max_worker_threads,
        dispersion_measure: Literal[
            "within-cluster", # Within cluster distance summary
            "across-cluster", # Across cluster distance summary
            "within-across-ratio", # Within vs. across cluster distance summary ratio
            "signal", # Clustered points distance summary
            "noise", # Noise points distance summary
            "signal-noise-ratio" # Clustered vs. noise distance summary ratio
        ] = "within-cluster",
        dispersion_minkowski_p: float = 2,
        inner_summary: SummaryType = "mean",
        outer_summary: SummaryType = "mean",
        dim: Literal["n", "p"] = "p",
        nan_policy: Literal[
            "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        DBScanSweepStatistic.__init__(
            self,
            min_eps=min_eps,
            max_eps=max_eps,
            n_results=n_results,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs,
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
            fitted_estimator: DBScanEstimator
    ) -> float | np.floating:
        
        if self._dispersion_measure in (
            "within-cluster",
            "across-cluster",
            "within-across-ratio"):
            return ClusteringDispersionSummaryMixin._get_result(
                self,
                data=data,
                fitted_estimator=fitted_estimator,
                dispersion_measure=self._dispersion_measure)

        clusters = fitted_estimator._predict(data)
        
        if self._dispersion_measure in ("noise", "signal-noise-ratio"):
            noise_distances = fitted_estimator._compute_noise_minkowski_distances(
                data, minkowski_p=self._dispersion_minkowski_p, clusters=clusters)
            
            noise_summary = (np.nan
                if noise_distances.size == 0
                else summarize(
                    noise_distances, summary_type=self._outer_summary))

            if self._dispersion_measure == "noise":
                return float(noise_summary)
            
        signal_distances = fitted_estimator._compute_signal_minkowski_distances(
            data, minkowski_p=self._dispersion_minkowski_p, clusters=clusters)

        if signal_distances.size == 0:
            return np.nan
        
        signal_summary = summarize(
            signal_distances, summary_type=self._outer_summary)
        
        if self._dispersion_measure == "signal":
            return float(signal_summary)
        
        if signal_summary == 0:
            if noise_summary == 0:
                return 1.0
            
            return 0.0

        if np.isnan(noise_summary):
            noise_summary = 0

        return signal_summary / noise_summary

import numpy as np

from typing import Literal

from pyspoc.types import DistanceMetric, SummaryType
from pyspoc.statistics.clustering.kclustering.kmedoids._base import KMedoidsEstimator
from pyspoc.settings import settings
from ._base import KMedoidsSweepStatistic
from ..._mixin import (
    ClusteringSILScoresMixin,
    ClusteringDispersionSummaryMixin)

from .._mixin import KClusteringInertiaMixin


class KMedoidsInertia(
    KMedoidsSweepStatistic,
    KClusteringInertiaMixin):
    """
    Returns a series of inertia scores from KMedoids fits with a sweep of `k` parameters.
    Inertia is the sum of distances between points and their assigned cluster medoids.
    Minkowski distance is used, which with `p=1` produces Manhattan distances that are the
    standard for K-Medoids inertia.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300, requires >1
        Maximum number of K-Medoids iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. `"alternate"` is faster while `"pam"` is more accurate.
        Corresponds to the `method` argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used in the KMedoids algorithm's
        determination of selected medoids. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    minkowski_p : float or `None`, default=`None`, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
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

    _long_name = "KMedoids - Inertia Distribution"
    _identifier = "kmedoids-inertia"

    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal[
                "k-medoids++", "random", "heuristic", "build"] = "heuristic",
            algorithm: Literal["alternate", "pam"] = "alternate",
            metric: DistanceMetric = "euclidean",
            minkowski_p: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
    
            KMedoidsSweepStatistic.__init__(
                self,
                min_k=min_k,
                max_k=max_k,
                max_iter=max_iter,
                random_seed=random_seed,
                initializer=initializer,
                algorithm=algorithm,
                metric=metric,
                minkowski_p=minkowski_p,
                dim=dim,
                nan_policy=nan_policy)
    
            KClusteringInertiaMixin.__init__(
                self)

    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMedoidsEstimator) -> float | np.floating:
                
        return KClusteringInertiaMixin._get_result(
            self, data, fitted_estimator)



class KMedoidsSILScores(
    KMedoidsSweepStatistic,
    ClusteringSILScoresMixin):
    """
    Returns a series of Silhouette scores from KMedoids fits with a sweep of `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300, requires >1
        Maximum number of K-Medoids iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. `"alternate"` is faster while `"pam"` is more accurate.
        Corresponds to the `method` argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used in the KMedoids algorithm's
        determination of selected medoids. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    minkowski_p : float or `None`, default=`None`, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    silhouette_metric : DistanceMetric, optional
        A string identifier for a supported distance metric used to compute the Silhouette score.
        If not supplied, `metric` is used. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    silhouette_minkowski_p : float or `None`, default=`None`, requires: >0 if \
        `silhouette_metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the Sihouette distance
        `silhouette_metric`. Note that for `p<1`, Minkowski distance is no longer a true distance
        metric as it can violate the triangle inequality.
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

    _long_name = "KMedoids - Silhouette Score Distribution"
    _identifier = "kmedoids-sil"

    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal[
                "k-medoids++", "random", "heuristic", "build"] = "heuristic",
            algorithm: Literal["alternate", "pam"] = "alternate",
            metric: DistanceMetric = "euclidean",
            minkowski_p: float | None = None,
            silhouette_metric: DistanceMetric | None = None,
            silhouette_minkowski_p: float | None = None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
    
            if silhouette_metric is None:
                 silhouette_metric = metric
            
            KMedoidsSweepStatistic.__init__(
                self,
                min_k=min_k,
                max_k=max_k,
                max_iter=max_iter,
                random_seed=random_seed,
                initializer=initializer,
                algorithm=algorithm,
                metric=metric,
                minkowski_p=minkowski_p,
                dim=dim,
                nan_policy=nan_policy)
    
            ClusteringSILScoresMixin.__init__(
                self,
                silhouette_metric=silhouette_metric,
                silhouette_minkowski_p=silhouette_minkowski_p)


    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMedoidsEstimator) -> float | np.floating:
                
        return ClusteringSILScoresMixin._get_result(
            self, data, fitted_estimator)


class KMedoidsDispersionSummaries(
    KMedoidsSweepStatistic,
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
    max_iter : int, default=300, requires >1
        Maximum number of K-Medoids iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. `"alternate"` is faster while `"pam"` is more accurate.
        Corresponds to the `method` argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : DistanceMetric, default="euclidean"
        A string identifier for a supported distance metric used in the KMedoids algorithm's
        determination of selected medoids. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    minkowski_p : float or `None`, default=`None`, requires: >0 if `metric = "minkowski"`
        Minkowski _p_ parameter used for Minkowski distance if selected as the KMedoids distance
        `metric`. Note that for `p<1`, Minkowski distance is no longer a true distance metric as it
        can violate the triangle inequality.
    dispersion_measure : {"within-cluster" "across-cluster", "within-across-ratio"}, \
            default="within-cluster"
        The type of dispersion measure to compute.
        - `"within-cluster"` : Within cluster distance summary.
        - `"across-cluster"` : Across cluster distance summary.
        - `"within-across-ratio"` : Within vs. across distance summary ratio.
    dispersion_minkowski_p : float, default=1, requires: >0
        Minkowski _p_ parameter used for Minkowski distance in dispersion calculation, with the
        default of `1` producing Manhattan distances. Note that for `p<1`, Minkowski distance is
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

    _long_name = "KMedoids - Dispersion Summary Distribution"
    _identifier = "kmedoids-dispersion"

    # TODO: Need to add summary_type and distance_type to docstring.

    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal[
                "k-medoids++", "random", "heuristic", "build"] = "heuristic",
            algorithm: Literal["alternate", "pam"] = "alternate",
            metric: DistanceMetric = "euclidean",
            minkowski_p: float | None = None,
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

        KMedoidsSweepStatistic.__init__(
            self,
            min_k=min_k,
            max_k=max_k,
            max_iter=max_iter,
            random_seed=random_seed,
            algorithm=algorithm,
            initializer=initializer,
            metric=metric,
            minkowski_p=minkowski_p,
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
            fitted_estimator: KMedoidsEstimator
    ) -> float | np.floating:
                
        return ClusteringDispersionSummaryMixin._get_result(
            self,
            data=data,
            fitted_estimator=fitted_estimator,
            dispersion_measure=self._dispersion_measure)

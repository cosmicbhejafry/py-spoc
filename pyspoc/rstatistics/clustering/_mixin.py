import numpy as np

from typing import Literal
from sklearn.metrics.cluster import silhouette_score

import pyspoc._argchecking as argchk

from pyspoc._tag import Tag, Type, Signature
from pyspoc._utils.numerical import summarize
from pyspoc.types import DistanceMetric, SummaryType
from pyspoc.statistics.clustering._mixin import ClusteringMixin
from pyspoc.statistics.clustering._base import ClusteringEstimator



class ClusteringSILScoresMixin(ClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.DESCRIPTIVE

    def __init__(
            self,
            silhouette_metric: DistanceMetric,
            silhouette_minkowski_p: float | None = None):
        
        if silhouette_metric == "minkowski":
            if silhouette_minkowski_p is None:
                raise ValueError("When using the 'minkowski' distance metric, "
                    "the 'silhouette_minkowski_p' parameter must be provided.")

            self._silhouette_metric_p = argchk.check_float(
                silhouette_minkowski_p, exclusive_minimum=0, arg_name="silhouette_minkowski_p")
        else:
            self._silhouette_metric_p = None
        
        self._silhouette_metric = silhouette_metric
        super().__init__()
    
    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: ClusteringEstimator,
            clusters: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None
    ) -> float | np.floating:

        if not fitted_estimator._are_clusters_valid(clusters, data.shape[0]):
            clusters = fitted_estimator._predict(data)

        n_clusters = np.unique(clusters).shape[0]

        if n_clusters <= 1:
            return np.nan

        if self._silhouette_metric == "minkowski":
            return float(silhouette_score(
                data, clusters, metric=self._silhouette_metric, p=self._silhouette_metric_p))
        
        return float(silhouette_score(
            data, clusters, metric=self._silhouette_metric))



class ClusteringDispersionSummaryMixin(ClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.DESCRIPTIVE

    # Distances are Minkowski with w = 1
    def __init__(
            self,
            inner_summary: SummaryType,
            outer_summary: SummaryType,
            dispersion_minkowski_p: float):

        self._inner_summary = inner_summary
        self._outer_summary = outer_summary
        self._dispersion_minkowski_p = argchk.check_float(
            dispersion_minkowski_p, exclusive_minimum=0, arg_name="dispersion_minkowski_p")

    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: ClusteringEstimator,
            dispersion_measure: Literal[
                "within-cluster", # Within cluster distance summary
                "across-cluster", # Across cluster distance summary
                "within-across-ratio"] # Within vs. across cluster distance summary ratio
    ) -> np.floating | float:
        
        labels = fitted_estimator._predict(data)
        
        if dispersion_measure in ("within-cluster", "within-across-ratio"):
            within_distances = fitted_estimator._compute_within_cluster_minkowski_distances(
                data,
                minkowski_p=self._dispersion_minkowski_p,
                clusters=labels)

            if not within_distances: # No clusters
                return np.nan

            cluster_summary = self._within_summarize(within_distances)
            within_summary = summarize(cluster_summary, self._outer_summary)

            if dispersion_measure == "within-cluster":
                return within_summary
       
        across_distances = fitted_estimator._compute_cross_cluster_minkowski_distances(
            data,
            minkowski_p=self._dispersion_minkowski_p,
            clusters=labels)

        if not across_distances: # Only one cluster.
            if dispersion_measure == "across-cluster":
                return np.nan

            across_summary = 0
        else:
            cluster_summary = self._within_summarize(across_distances)
            across_summary = summarize(cluster_summary, self._outer_summary)
            
        if dispersion_measure == "across-cluster":
            return across_summary

        # Point-to-cluster cluster-to-anchor distance summary ratio
        # Anchor is centroid or medoid depending on selected option.
        if within_summary == 0 and across_summary == 0:
            return 1.0

        return within_summary / across_summary


    def _within_summarize(
            self,
            cluster_arrays: dict[int, np.ndarray[tuple[int], np.dtype[np.floating]]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        cluster_summaries = np.full(shape=len(cluster_arrays), fill_value=np.nan)
        
        for i, (_, array) in enumerate(cluster_arrays.items()):
            cluster_summary = (0 if array.size == 0
                else summarize(
                    array, summary_type=self._inner_summary))
            cluster_summaries[i] = cluster_summary

        return cluster_summaries

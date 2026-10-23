import numpy as np

from typing import Literal
from sklearn.metrics.cluster import silhouette_score

from pyspoc._tag import Tag, Type, Signature
from pyspoc.statistics.clustering.kclustering._mixin import KClusteringMixin
from pyspoc.statistics.clustering.kclustering._base import KClusteringEstimator


class KClusteringErrorsMixin(KClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.ERROR
    
    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KClusteringEstimator) -> float:
        
        model = fitted_estimator._get_model()
        return model.inertia_


class KClusteringSILScoresMixin(KClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.DESCRIPTIVE

    def __init__(
            self,
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
                "yule"] = "euclidean"):

        self._metric = metric
        super().__init__()
    
    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KClusteringEstimator) -> float:
        
        model = fitted_estimator._get_model()
        labels = model.predict(data)
        return float(silhouette_score(data, labels, metric=self._metric))


class KClusteringDispersionSummaryMixin(KClusteringMixin):

    _tags = (Tag.NONLINEAR, Tag.NONPARAMETRIC, Tag.CLUSTERING)
    _signatures = (Signature.BLOBBINESS,)
    _type = Type.DESCRIPTIVE

    # Distances are euclidean
    def __init__(
            self,
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
                "ptc-ctm-ratio" # Point-to-cluster cluster-to-medoid distance summary ratios
            ]):

        self._summary_type = summary_type
        self._distance_type = distance_type
    
    
    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KClusteringEstimator
    ) -> np.floating | float:
        
        model = fitted_estimator._get_model()
        
        if self._distance_type in ("ptc", "ptc-ctc-ratio", "ptc-ctm-ratio"):
            ptc_distances = model.transform(data)
            ptc_distance_summary = self._scalar_summarize(ptc_distances)

            if self._distance_type == "ptc":
                return ptc_distance_summary # Point-to-cluster distance summary

            anchor_type = ("centroid"
                if self._distance_type == "ptc-ctc-ratio"
                else "medoid")

            atc_distance_summary = self._get_anchor_to_cluster_distances(
                data,
                cluster_centers=model.cluster_centers_,
                anchor_type=anchor_type
            )

            # Point-to-cluster cluster-to-anchor distance summary ratio
            # Anchor is centroid or medoid depending on selected option.
            return ptc_distance_summary / atc_distance_summary

        anchor_type = ("centroid"
            if self._distance_type == "ctc"
            else "medoid")
        atc_distance_summary = self._get_anchor_to_cluster_distances(
            data,
            cluster_centers=model.cluster_centers_,
            anchor_type=anchor_type)
        return atc_distance_summary


    def _get_anchor_to_cluster_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            cluster_centers: np.ndarray[tuple[int], np.dtype[np.floating]],
            anchor_type: Literal["centroid", "medoid"]
    ) -> float | np.floating:
            anchor = self._get_anchor(data, anchor_type=anchor_type)
            atc_distances = np.linalg.norm(
                cluster_centers - anchor, ord=2, axis=1)
            atc_distance_summary = self._scalar_summarize(
                atc_distances)
            return atc_distance_summary
        
    
    def _get_anchor(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            anchor_type: Literal["centroid", "medoid"]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        centroid = data.mean(axis=0)

        if anchor_type == "centroid":
            return centroid
        
        dist = np.linalg.norm(data - centroid, ord=2, axis=1)
        med_idx = np.argmin(dist)
        return data[med_idx]
    
    def _scalar_summarize(
            self,
            value: (np.ndarray[tuple[int, ...], np.dtype[np.floating]] |
                    np.floating |
                    float)
    ) -> np.floating | float:
        
        while isinstance(value, np.ndarray):
            value = self._summarize(value)

        return value

    
    def _summarize(
            self,
            array: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            axis: Literal[0,1,None] = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]] | np.floating:

        match self._summary_type:

            case "mean":
                return array.mean(axis=axis)

            case "median":
                return np.median(array, axis=axis)

            case "min":
                return array.min(axis=axis)

            case "max":
                return array.max(axis=axis)

            case "stdev":
                return array.std(axis=axis)

            case "iqr":
                return (np.quantile(array, 0.75, axis=axis) -
                        np.quantile(array, 0.25, axis=axis))

            case "range":
                return (array.max(axis=axis) - array.min(axis=axis))

            case _:
                return array.mean(axis=axis)

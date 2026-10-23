from typing import Literal
from abc import ABC

from pyspoc.settings import settings
from pyspoc.statistics.clustering.kclustering.kmedoids._base import KMedoidsEstimator
from .._base import KClusteringSweepStatistic



class KMedoidsSweepStatistic(KClusteringSweepStatistic[KMedoidsEstimator], ABC):
    """Provide common initialization and estimator access for KMedoids Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.


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
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
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


    Notes
    -----
    This class participates in cooperative multiple inheritance. Its
    initializer therefore calls ``super().__init__()`` after storing its own
    state.
    """

    def __init__(
            self,
            min_k: int,
            max_k: int,
            max_iter: int = 300,
            random_seed: int | None = None,
            initializer: Literal[
                "k-medoids++", "random", "heuristic", "build"] = "heuristic",
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
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        estimator_kwargs = {
            "initializer": initializer,
            "algorithm": algorithm,
            "metric": metric}
    
        super().__init__(
            clustering_type=KMedoidsEstimator,
            min_k=min_k,
            max_k=max_k,
            max_iter=max_iter,
            random_seed=random_seed,
            dim=dim,
            nan_policy=nan_policy,
            estimator_kwargs=estimator_kwargs)

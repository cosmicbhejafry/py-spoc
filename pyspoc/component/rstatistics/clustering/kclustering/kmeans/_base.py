from typing import Literal
from abc import ABC

from pyspoc.settings import settings
from pyspoc.component.statistics.clustering.kclustering.kmeans._base import KMeansEstimator
from .._base import KClusteringSweepStatistic



class KMeansSweepStatistic(KClusteringSweepStatistic[KMeansEstimator], ABC):
    """Provide common initialization and estimator access for KMeans Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.


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
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
        initializer: Literal["k-means++", "random"] = "k-means++",
        tol: float = 0.0001,
        dim: Literal["n", "p"] = "p",
        nan_policy: Literal[
            "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

   
        super().__init__(
            clustering_type=KMeansEstimator,
            min_k=min_k,
            max_k=max_k,
            max_iter=max_iter,
            random_seed=random_seed,
            dim=dim,
            nan_policy=nan_policy,
            estimator_kwargs={
                "initializer": initializer,
                "tol": tol,
                "algorithm": algorithm
            })

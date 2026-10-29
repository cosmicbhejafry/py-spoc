"""Parameter handling and estimator resolution for K-Medoids Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Literal
from abc import ABC
from sklearn_extra.cluster import KMedoids

from .._base import KClusteringEstimator, KClusteringStatistic


_LOGGER = logging.getLogger(__name__)


AVAILABLE_DISTANCES = (
    "euclidean",
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
    "yule"
)



class KMedoidsEstimator(KClusteringEstimator):
    """Manage a scikit-learn backed KMedoids model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

    Parameters
    ----------
    k : int, requires:>=2
        Number of clusters to form.
    max_iter : int, default=300, requires:>=1
        Maximum number of iterations for a single K-Medoids run.
    random_seed : int or `None`, optional
        Per-estimator random-seed override. If ``None``, use the active
        library-wide seed.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : string, default="euclidean"
        What distance metric to use. See :obj:`AVAILABLE_DISTANCES` for a list of
        available distance metrics.

    Notes
    -----
    The underlying :class:`sklearn_extra.cluster.KMedoids` objects remains private so
    cache consumers cannot accidentally replace it through a public property.
    """

    def __init__(
            self,
            k: int,
            max_iter: int,
            random_seed: int | None,
            initializer: Literal["k-medoids++", "random", "heuristic", "build"],
            algorithm: Literal["alternate", "pam"],
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
                "yule"]):

        super().__init__(
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        self._initializer = initializer
        self._metric = metric
        self._algorithm = algorithm


    # Override the abstract fitting hook inherited from LazyFittedCachedEstimatorMixin.
    # Its public fit() method handles thread-safe locking, data equivalence,
    # and the fitted-state transition around this operation.
    def _fit_estimator(self, data: np.ndarray) -> KMedoids:
        """Construct and fit the private K-Medoids model.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional observations accepted by the estimator fitting
            lifecycle.

        Returns
        -------
        None
            The fitted scikit-learn model is stored on this estimator.

        Notes
        -----
        This method overrides the abstract
        :meth:`LazyFittedCachedEstimatorMixin._fit_estimator` hook. Call
        :meth:`fit` rather than invoking this method directly so cache and
        thread-safety invariants remain enforced.
        """
        
        # Translate the library's boolean verbosity setting to scikit-learn's
        # integer verbosity parameter. False becomes 0 and True becomes 1.
        model = KMedoids(
            n_clusters=self._k,
            metric=self._metric,
            method=self._algorithm,
            init=self._initializer,
            max_iter=self._max_iter,
            random_state=self.random_seed,
        )

        # scikit-learn returns the fitted estimator from fit(); retain that
        # object as the private model published by the lazy fitting lifecycle.
        return model.fit(data)



class KMedoidsStatistic(KClusteringStatistic[KMedoidsEstimator], ABC):
    """Provide common initialization and estimator access for KMedoids Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.


    Parameters
    ----------
    k : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations for a single K-Medoids run.
    random_seed : int or `None`, optional
        Per-estimator random-seed override. If ``None``, use the active
        library-wide seed.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    metric : string, default="euclidean"
        What distance metric to use. See :obj:`AVAILABLE_DISTANCES` for a list of
        available distance metrics.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    dim : {"n", "p"}
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}
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
            k: int,
            max_iter: int,
            random_seed: int | None,
            initializer: Literal["k-medoids++", "random", "heuristic", "build"],
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
                "yule"],
            algorithm: Literal["alternate", "pam"],
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"]):

        estimator_kwargs = {
            "initializer": initializer,
            "metric": metric,
            "algorithm" : algorithm}
        
        super().__init__(
            clustering_type=KMedoidsEstimator,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            dim=dim,
            nan_policy=nan_policy,
            estimator_kwargs=estimator_kwargs)

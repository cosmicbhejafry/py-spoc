"""Shared parameter handling and estimator resolution for KMeans Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, Literal
from abc import ABC
from sklearn.cluster import KMeans

from pyspoc.settings import settings
from .._base import KClusteringStatistic, KClusteringEstimator


_LOGGER = logging.getLogger(__name__)



class KMeansEstimator(KClusteringEstimator):
    """Manage a scikit-learn backed KMeans model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

    Parameters
    ----------
    k : int, requires: >1
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}, default="k-means++"
        Strategy used to initialize cluster centroids.
        
        "k-means++": selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        "random": choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Which algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan"
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).


    Notes
    -----
    The underlying :class:`sklearn.cluster.KMeans` objects remains private so
    cache consumers cannot accidentally replace it through a public property.
    """

    # Include the resolved random seed in cache identity. This prevents an
    # estimator created under one settings context from being reused after the
    # effective library-wide seed changes.
    _freeze_random_seed = True
    

    def __init__(
            self,
            k: int,
            max_iter: int,
            random_seed: int | None,
            tol: float,
            initializer: Literal["k-means++", "random"],
            algorithm: Literal["lloyd", "elkan"],
        ):
        """Initialize K-Means configuration and unfitted state.

        Notes
        -----
        Constructor arguments are recorded automatically by the inherited
        estimator machinery and subsequently participate in cache matching.

        """

        super().__init__(
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        self._initializer = initializer
        self._algorithm = algorithm
        self._tol = tol


    # Override the abstract fitting hook inherited from LazyFittedCachedEstimatorMixin.
    # Its public fit() method handles thread-safe locking, data equivalence,
    # and the fitted-state transition around this operation.

    def _fit_estimator(self, data: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> KMeans:
        """Construct and fit the private K-Means model.

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
        model = KMeans(
            n_clusters=self._k,
            init=self._initializer,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self.random_seed,
            algorithm=self._algorithm,
            copy_x=True,
            verbose=int(settings.current.verbose),
        )

        # scikit-learn returns the fitted estimator from fit(); retain that
        # object as the private model published by the lazy fitting lifecycle.
        return model.fit(data)



class KMeansStatistic(KClusteringStatistic[KMeansEstimator], ABC):
    """Provide common initialization and estimator access for KMeans Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.


    Parameters
    ----------
    k : int, requires: >1
        Number of clusters to form.
    max_iter : int
        Maximum number of K-Means iterations for a single run.
    random_seed : int or None
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-means++", "random"}
        Strategy used to initialize cluster centroids.
        
        "k-means++": selects initial cluster centroids using sampling based on an
        empirical probability distribution of the points' contribution to the overall inertia.
        This technique speeds up convergence. The algorithm implemented is "greedy k-means++".
        It differs from the vanilla k-means++ by making several trials at each sampling step and
        choosing the best centroid among them.

        "random": choose n_clusters observations (rows) at random from data for the initial
        centroids.
    algorithm : {"lloyd", "elkan"}
        Which algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan"
        variation can be more efficient on some datasets with well-defined clusters, by
        using the triangle inequality. However it's more memory intensive due to the allocation
        of an extra array of shape (n_samples, n_clusters).
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
            tol: float,
            initializer: Literal["k-means++", "random"],
            algorithm: Literal["lloyd", "elkan"],
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"]):

        estimator_kwargs = {
            "tol": tol,
            "initializer": initializer,
            "algorithm": algorithm
        }

        super().__init__(
            clustering_type=KMeansEstimator,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            estimator_kwargs=estimator_kwargs,
            dim=dim,
            nan_policy=nan_policy
        )

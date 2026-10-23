"""Shared parameter handling and estimator contracts for K-clustering Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, Literal
from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN

import pyspoc._argchecking as argchk

from pyspoc.settings import settings
from pyspoc._tag import Tag, Signature
from pyspoc._estimators import LazyFittedCachedEstimatorMixin
from pyspoc._random import RandomSeedMixin
from pyspoc.statistics.base import Statistic


_LOGGER = logging.getLogger(__name__)


class DBScanEstimator(
    LazyFittedCachedEstimatorMixin[DBSCAN]):
    """Manage a scikit-learn backed DBScan model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

    Parameters
    ----------
    eps : float, requires: >0
        The maximum distance between two samples for one to be considered as in the neighborhood of
        the other. This is not a maximum bound on the distances of points within a cluster. This is
        the most important DBSCAN parameter to choose appropriately for your data set and distance
        function. Smaller values generally lead to more clusters.
    min_samples : int, requires: >=2, <=`n` with `n` the number of observations in the data matrix \
        provided

        The number of samples (or total weight) in a neighborhood for a point to be considered as a
        core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN
        will find denser clusters, whereas if it is set to a lower value, the found clusters will be
        more sparse.
    
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.

    Notes
    -----
    The underlying :class:`sklearn.cluster.KMeans` or :class:`sklearn_extra.cluster.KMedoids`
    objects remains private so cache consumers cannot accidentally replace it through
    a public property.
    """

    # Include the resolved random seed in cache identity. This prevents an
    # estimator created under one settings context from being reused after the
    # effective library-wide seed changes.

    def __init__(
        self,
        eps: float,
        min_samples: int,
        metric: Literal[
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
            "yule"],
        algorithm: Literal[
            "auto", "ball_tree", "kd_tree", "brute"],
        leaf_size: int,
        minkowski_p: float | None,
        n_jobs: int | None
    ) -> None:

        self._eps = eps # Checked upstream.
        self._min_samples = argchk.check_integer(min_samples, minimum=2, arg_name="min_samples")
        self._metric = metric
        self._algorithm = algorithm
        self._leaf_size = argchk.check_integer(leaf_size, minimum=1, arg_name="leaf_size")
        self._minkowski_p = (argchk.check_float(
                minkowski_p, exclusive_minimum=0, arg_name="minkowski_p")
            if minkowski_p is not None
            else None)

        if (n_jobs is None
            or n_jobs < 0
            or n_jobs > settings.current.max_worker_threads):
            n_jobs = -1

        self._n_jobs = n_jobs
        super().__init__()
    

    def _fit_estimator(self, data: np.ndarray) -> DBSCAN:
        model = DBSCAN(
            eps=self._eps,
            min_samples=self._min_samples,
            metric=self._metric,
            algorithm=self._algorithm,
            leaf_size=self._leaf_size,
            p=self._minkowski_p,
            n_jobs=self._n_jobs)

        # scikit-learn returns the fitted estimator from fit(); retain that
        # object as the private model published by the lazy fitting lifecycle.
        return model.fit(data)



class DBScanStatisticBaseMixin(
    RandomSeedMixin,
    ABC,
):
    """Provide common initialization and estimator access for KClustering Statistics
    used by both standard clustering and sweep statistics.

    The base validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.


    Parameters
    ----------
    clustering_type : type[KClusteringEstimator]
        The specific KClusteringEstimator type to use.
    max_iter : int, default=300
        Maximum number of K-Clustering iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.


    Raises
    ------
    TypeError
        If a runtime-checked argument has an incompatible type.
    ValueError
        If a numeric argument or component is outside its accepted range,
        or if required components are empty.


    Notes
    -----
    This class participates in cooperative multiple inheritance. Its
    initializer therefore calls ``super().__init__()`` after storing its own
    state.
    """

    def __init__(
            self,
            min_samples: int,
            metric: Literal[
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
                "yule"],
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            minkowski_p: float | None,
            n_jobs: int | None,
            **kwargs: Any):

        self._min_samples = min_samples
        self._metric = metric
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._minkowski_p = minkowski_p
        self._n_jobs = n_jobs

        # Continue the cooperative initializer chain so random-seed and other
        # inherited mixins can initialize their state.
        super().__init__(**kwargs)
        

    def _get_fitted_estimator(
            self,
            eps: float,
            data: np.ndarray) -> DBScanEstimator:
        """Resolve, fit, and retain the family-specific clustering estimator.

        Parameters
        ----------
        k : int
            Number of clusters to form.
        data : numpy.ndarray
            Dataset used both as the estimator cache identity and training
            input.

        Returns
        -------
        KClusteringEstimator
            Fitted cached estimator. The returned object may be shared by
            multiple Statistics and should therefore be treated as read-only.
        """
        
        self._estimator_ = DBScanEstimator.get_or_create(
            data=data,
            eps=eps,
            min_samples=self._min_samples,
            metric=self._metric,
            algorithm=self._algorithm,
            leaf_size=self._leaf_size,
            minkowski_p=self._minkowski_p,
            n_jobs=self._n_jobs)

        self._estimator_.fit(data)
        return self._estimator_


class DBScanStatistic(
    DBScanStatisticBaseMixin,
    Statistic):

    """Provides shared KClustering statistic configuration and functionality.

    Parameters
    ----------
    clustering_type : type[KClusteringEstimator]
        The specific KClusteringEstimator type to use.
    k : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of K-Clustering iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    kwargs : Any, optional
        Additional key-word arguments required by the specific KClustering
        statistic implementation.

    Raises
    ------
    TypeError
        If a runtime-checked argument has an incompatible type.
    ValueError
        If a numeric argument or component is outside its accepted range,
        or if ``components`` is empty.
    """
    _tags = (Tag.CLUSTERING, Tag.NONPARAMETRIC, Tag.NONLINEAR)
    _signatures = (Signature.BLOBBINESS,)

    def __init__(
            self,
            eps: float,
            min_samples: int,
            metric: Literal[
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
                "yule"],
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            minkowski_p: float | None,
            n_jobs: int | None,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"]):

        # eps must be greater than 0.
        self._eps = argchk.check_float(eps, exclusive_minimum=0, arg_name="eps")

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism="parallel",
            thread_safety="unsafe",
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs)
        
    # Override Statistic.compute(), whose abstract contract defines the public
    # entry point used by Calculator.
    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Fit or reuse an estimator and derive the concrete Statistic result.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional point-cloud dataset.

        Returns
        -------
        numpy.ndarray[tuple[int, int], numpy.dtype[numpy.floating]]
            Two-dimensional floating numpy array Statistic result.
        """

        n = data.shape[0]
        
        if n < self._min_samples:
            raise ValueError(
                "The minimum samples (min_samples) must not exceed the number of data points.")

        # Resolve dimensions against this particular dataset without mutating
        # the originally requested component configuration.
        fitted_estimator = self._get_fitted_estimator(eps=self._eps, data=data)
        return self._get_result(data=data, fitted_estimator=fitted_estimator)


    @abstractmethod
    def _get_result(
            self,
            data: np.ndarray,
            fitted_estimator: DBScanEstimator
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Extract a concrete result from a shared fitted estimator.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional point-cloud dataset.
        fitted_estimator : DBScanEstimator
            Estimator fitted to the current dataset. The estimator and
            its internal model are shared and must be treated as read-only.

        Returns
        -------
        numpy.ndarray[tuple[int, int], numpy.dtype[numpy.floating]]
            Two-dimensional floating numpy array selected by the concrete
            implementation of :meth:`_get_result`.

        Notes
        -----
        This abstract hook must be implemented by every concrete K-clustering Statistic.
        Implementations may inspect the borrowed estimator but must not retrain
        or mutate its shared model.
        """
        pass

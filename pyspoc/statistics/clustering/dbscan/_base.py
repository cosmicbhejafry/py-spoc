"""Shared parameter handling and estimator contracts for K-clustering Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, Literal
from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.validation import check_array
from scipy.spatial.distance import pdist

import pyspoc._argchecking as argchk

from pyspoc.types import DistanceMetric
from pyspoc.settings import settings
from pyspoc._tag import Tag, Signature
from pyspoc._random import RandomSeedMixin
from pyspoc.statistics.base import Statistic
from .._base import ClusteringEstimator

_LOGGER = logging.getLogger(__name__)



class DBScanEstimator(
    ClusteringEstimator[DBSCAN]):
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
    minkowski_p : float or `None`, default=None, required if `metric = "minkowski"`
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
    -----
    The underlying :class:`sklearn.cluster.DBSCAN` objects remains private so cache consumers cannot
    accidentally replace it through a public property.
    """

    # Include the resolved random seed in cache identity. This prevents an
    # estimator created under one settings context from being reused after the
    # effective library-wide seed changes.

    def __init__(
            self,
            eps: float,
            min_samples: int,
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            metric: DistanceMetric,
            minkowski_p: float | None,
            n_jobs: int = settings.current.max_worker_threads):

        self._eps = eps # Checked upstream.
        self._min_samples = argchk.check_integer(min_samples, minimum=2, arg_name="min_samples")
        self._metric = metric
        self._algorithm = algorithm
        self._leaf_size = argchk.check_integer(leaf_size, minimum=1, arg_name="leaf_size")
        
        if metric == "minkowski":
            if minkowski_p is None:
                raise ValueError("minkowski_p must be supplied when metric is 'minkowski'.")

            self._minkowski_p = argchk.check_float(
                minkowski_p, exclusive_minimum=0, arg_name="minkowski_p")
        else:
            self._minkowski_p = None

        settings_n_jobs = settings.current.max_worker_threads

        if n_jobs < 1 or (n_jobs > settings_n_jobs and settings_n_jobs > 0):
            n_jobs = settings_n_jobs

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


    def _get_centres(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Return the arithmetic centroid of each non-noise fitted cluster.

        Returns
        -------
        numpy.ndarray of shape (n_clusters, n_features), dtype float64
            Centroids in ascending cluster-label order. Both core and border
            observations contribute; noise observations (label -1) do not.
            An all-noise fit returns an empty array of shape (0, n_features).

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        RuntimeError
            If the training dataset is unavailable.

        Notes
        -----
        DBSCAN does not estimate centres itself. These are post-fit geometric
        summaries, not the core observations stored in model.components_.
        """

        model = self._get_model()
        data = self._get_attached_dataset()

        if data is None:
            raise RuntimeError("The fitted DBSCAN estimator has no attached training dataset.")

        labels = model.labels_
        assigned = labels >= 0

        # Map labels to compact indices instead of assuming contiguous labels.
        # This also excludes noise before any sums or counts are calculated.
        cluster_labels, inverse, counts = np.unique(
            labels[assigned], return_inverse=True, return_counts=True)
        centres = np.zeros((len(cluster_labels), data.shape[1]), dtype=np.float64)
        np.add.at(centres, inverse, data[assigned])

        # Empty counts broadcast correctly for the all-noise (0, p) result.
        centres /= counts[:, None]
        return centres


    def _predict(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> np.ndarray:
        """Assign observations to the nearest fitted core point within eps.

        Parameters
        ----------
        data : numpy.ndarray of shape (n_query, n_features)
            Finite query observations with the same feature count as the
            training data. Empty query matrices are supported.

        Returns
        -------
        numpy.ndarray of shape (n_query,), dtype intp
            Cluster label of the nearest core point if its distance is at most
            the fitted eps; otherwise -1. If fitting found no core points,
            every query is labelled as noise.

        Raises
        ------
        ValueError
            If the estimator is unfitted, or query data are not a finite
            two-dimensional matrix with the fitted number of features.

        Notes
        -----
        This is a prediction extension, not a native DBSCAN operation: query
        observations do not change density estimates or merge fitted clusters.
        Only core points support assignments; border points do not propagate
        cluster membership. Exact distance ties select the core point with
        the smallest original training-row index. Ambiguous training border
        points may consequently receive different labels than model.labels_.
        Distance computation uses the fitted metric and Minkowski exponent,
        and is chunked to avoid a full n_query by n_core distance allocation.
        """
        model = self._get_model()
        queries = check_array(data, dtype=(np.float64, np.float32), ensure_min_samples=0)

        if queries.shape[1] != model.n_features_in_:
            raise ValueError("Query data must have the same number of features as the fitted data.")

        predictions = np.full(queries.shape[0], -1, dtype=np.intp)

        if queries.shape[0] == 0 or model.core_sample_indices_.size == 0:
            return predictions

        core_labels = model.labels_[model.core_sample_indices_]
        metric_kwargs = {}

        if self._metric == "minkowski":
            metric_kwargs.setdefault("p", self._minkowski_p)

        def assign_chunk(distances, start):
            # components_ follows core_sample_indices_ order. NumPy argmin
            # chooses the first minimum, giving deterministic training-order
            # tie breaking without depending on thread completion order.
            nearest = np.argmin(distances, axis=1)
            minimum = distances[np.arange(distances.shape[0]), nearest]
            assigned = np.where(minimum <= self._eps, core_labels[nearest], -1)
            predictions[start:start + len(assigned)] = assigned

        # Consuming this generator executes each chunk and immediately discards
        # its distance matrix; the reducer writes only one label per query.
        for _ in pairwise_distances_chunked(
                queries, model.components_, reduce_func=assign_chunk,
                metric=self._metric, n_jobs=self._n_jobs, **metric_kwargs):
            pass
        return predictions
    

    def _get_signal_data(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            clusters: np.ndarray[tuple[int]]
    ) -> tuple[
            np.ndarray[tuple[int, int], np.dtype[np.floating]],
            np.ndarray[tuple[int]]]:
        
        mask = clusters > -1
        return data[mask], clusters[mask]


    def _get_noise_data(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            clusters: np.ndarray[tuple[int]]
    ) -> tuple[
            np.ndarray[tuple[int, int], np.dtype[np.floating]],
            np.ndarray[tuple[int]]]:
                
        mask = clusters == -1
        return data[mask], clusters[mask]
    

    def _compute_within_cluster_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int, np.ndarray[tuple[int], np.dtype[np.floating]]]:

        if not self._are_clusters_valid(clusters, data.shape[0]):
            clusters = self._predict(data)

        # Filter out noise data points.
        signal_data, signal_clusters = self._get_signal_data(data, clusters)

        # Proceed as normal.
        return super()._compute_within_cluster_minkowski_distances(
            signal_data,
            minkowski_p=minkowski_p,
            clusters=signal_clusters)
    
    
    def _compute_cross_cluster_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int, np.ndarray[tuple[int], np.dtype[np.floating]]]:
        
        if not self._are_clusters_valid(clusters, data.shape[0]):
            clusters = self._predict(data)

        # Filter out noise data points.
        signal_data, signal_clusters = self._get_signal_data(data, clusters)

        # Proceed as normal.
        return super()._compute_cross_cluster_minkowski_distances(
            signal_data,
            minkowski_p=minkowski_p,
            clusters=signal_clusters)


    def _compute_noise_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        # Filter to noise data points.
        noise_data, noise_clusters = self._get_noise_data(data, clusters)

        if noise_data is None or noise_data.shape[0] == 0:
            return np.array([])

        noise_label = -1

        # Proceed as normal.
        results = super()._compute_within_cluster_minkowski_distances(
            noise_data,
            minkowski_p=minkowski_p,
            clusters=noise_clusters)

        return results[noise_label]
    

    def _compute_signal_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        signal_data, signal_clusters = self._get_signal_data(data, clusters)

        if signal_data is None or signal_data.shape[0] == 0:
            return np.array([])

        dummy_label = 0
        dummy_labels = np.full_like(signal_clusters, fill_value=dummy_label)

        # Proceed as normal.
        results = super()._compute_within_cluster_minkowski_distances(
            signal_data,
            minkowski_p=minkowski_p,
            clusters=dummy_labels)

        return results[dummy_label]


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
    min_samples : int, requires: >=2, <=`n` with `n` the number of observations in the data matrix \
        provided

        The number of samples (or total weight) in a neighborhood for a point to be considered as a
        core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN
        will find denser clusters, whereas if it is set to a lower value, the found clusters will be
        more sparse.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}
        a
    leaf_size : int, requires: >=1
        a
    metric : DistanceMetric
        A string identifier for a supported distance metric. See :class:`DistanceMetric` for the
        list of supported distance metrics and their formulae.
    minkowski_p : float, optional
        If `"minkowski"` distance metric selected, supplies `p` parameter used in its formula.
    n_jobs : int, optional
        The number of threads to recruit for the DBScan fit. If `n_jobs < 1` then all available
        threads are recruited. If `n_jobs = None` then the library-wide setting
        `settings.current.max_worker_threads` is used.
    kwargs : Any, optional
        Additional keyword-arguments specific to the underlying statistic implementation.

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
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            metric: DistanceMetric,
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
    min_samples : int, requires: >=2, <=`n` with `n` the number of observations in the data matrix \
        provided

        The number of samples (or total weight) in a neighborhood for a point to be considered as a
        core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN
        will find denser clusters, whereas if it is set to a lower value, the found clusters will be
        more sparse.
    metric : DistanceMetric
        A string identifier for a supported distance metric. See :class:`DistanceMetric` for the
        list of supported distance metrics and their formulae.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}
        a
    leaf_size : int, requires: >=1
        a
    metric : DistanceMetric
        A string identifier for a supported distance metric used in the DBSCAN algorithm's
        determination of core points. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
    minkowski_p : float, required if `metric = "minkowski"`
        If `"minkowski"` distance metric selected, supplies `p` parameter used in its formula.
    n_jobs : int, optional
        The number of threads to recruit for the DBScan fit. If `n_jobs < 1` then all available
        threads are recruited. If `n_jobs = None` then the library-wide setting
        `settings.current.max_worker_threads` is used.

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
            eps: float | None,
            desired_k : float | None,
            max_discover_eps_iter: float | None,
            min_samples: int,
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            metric: DistanceMetric,
            minkowski_p: float | None,
            n_jobs: int | None,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"]):

        self._eps = None
        self._desired_k = None
        self._max_discover_eps_iter = None

        if eps is not None:
            # eps must be greater than 0.
            self._eps = argchk.check_float(eps, exclusive_minimum=0, arg_name="eps")

        else:
            if desired_k is None:
                raise ValueError("Either eps or k must be supplied.")

            self._desired_k = argchk.check_natural_number(desired_k, arg_name="k")
            self._max_discover_eps_iter = argchk.check_natural_number(
                max_discover_eps_iter, arg_name="max_discovery_iter")


        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism="parallel",
            thread_safety="unsafe",
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

        if self._eps is not None:
            # Resolve dimensions against this particular dataset without mutating
            # the originally requested component configuration.
            fitted_estimator = self._get_fitted_estimator(eps=self._eps, data=data)
            return self._get_result(data=data, fitted_estimator=fitted_estimator)

        return self._compute_by_k(data)


    def _compute_by_k(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) ->  np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        eps = pdist(data, metric="euclidean").mean()
        fit_results = self._fit_by_k(eps, data=data, prev_eps=None, prev_k=None)
        prev_eps = eps
        result, k, eps = fit_results

        if k == self._desired_k:
            return result

        prev_k = k
        best_result = result
        closest_k = k

        for _ in range(self._max_discover_eps_iter):
            fit_results = self._fit_by_k(
                eps=eps, data=data, prev_eps=prev_eps, prev_k=prev_k)
            prev_eps = eps
            result, k, eps = fit_results

            if k == self._desired_k:
                return result

            if abs(k - self._desired_k) < abs(closest_k - self._desired_k):
                closest_k = k
                best_result = result

        return best_result


    def _fit_by_k(
            self,
            eps: float,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            prev_eps: float | None,
            prev_k: int | None):
        
        fitted_estimator = self._get_fitted_estimator(eps=eps, data=data)
        result = self._get_result(data=data, fitted_estimator=fitted_estimator)
        k = result.shape[1]

        if k == self._desired_k:
            return result, k, eps
        
        if k == 0:
            return result, k, eps*2

        if k > self._desired_k:
            if prev_k is None or k <= prev_k:
                return result, k, eps*2
            
            return result, k, abs(eps - prev_eps) / 2

        if prev_k is None or k >= prev_k:
            return result, k, eps/2

        return result, k, abs(eps - prev_eps) / 2


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

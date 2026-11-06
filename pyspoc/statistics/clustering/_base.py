"""Shared parameter handling and estimator contracts for K-clustering Statistics."""
from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import TypeVar, TypeGuard, Literal
from abc import abstractmethod
from numba import njit
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids

from pyspoc.settings import settings
from pyspoc._numba import numba_dispatch
from pyspoc._estimators import LazyFittedCachedEstimatorMixin


TClusteringModel = TypeVar("TClusteringModel", bound=KMeans | KMedoids | DBSCAN)



class ClusteringEstimator(
    #Generic[TClusteringModel],
    LazyFittedCachedEstimatorMixin[TClusteringModel]):
    """Manage a scikit-learn backed Clustering model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

    Parameters
    ----------
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.

    Notes
    -----
    The underlying estimator objects remain private so cache consumers cannot
    accidentally replace it through a public property.
    """

    # Include the resolved random seed in cache identity. This prevents an
    # estimator created under one settings context from being reused after the
    # effective library-wide seed changes.
    _freeze_random_seed = True

    def __init__(
        self,
        random_seed: int | None = None, # Handled automatically by RandomSeedMixin
    ) -> None:

        super().__init__()
    

    # The abstract fitting hook inherited from LazyFittedCachedEstimatorMixin
    # must be overridden by concrete class.
    # Its public fit() method handles thread-safe locking, data equivalence,
    # and the fitted-state transition around this operation.
    @abstractmethod
    def _fit_estimator(self, data: np.ndarray) -> KMeans | KMedoids | DBSCAN:
        pass


    @abstractmethod
    def _get_centres(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        pass


    @abstractmethod
    def _predict(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> np.ndarray:
        pass


    @staticmethod
    def _are_clusters_valid(
        clusters: np.ndarray[tuple[int]] | None,
        expected_size: int
    ) -> TypeGuard[np.ndarray[tuple[int]]]:
        return isinstance(clusters, np.ndarray) and clusters.shape[0] == expected_size


    def _compute_within_cluster_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int, np.ndarray[tuple[int], np.dtype[np.floating]]]:

        return self._compute_minkowski_distances(
            data,
            minkowski_p=minkowski_p,
            cluster_comparison="within",
            clusters=clusters)


    def _compute_cross_cluster_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int, np.ndarray[tuple[int], np.dtype[np.floating]]]:

        return self._compute_cross_minkowski_distances(
            data,
            minkowski_p=minkowski_p,
            clusters=clusters)
        

    def _compute_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            cluster_comparison: Literal["all","within"],
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int,
              np.ndarray[tuple[int], np.dtype[np.floating]]]:

        if not self._are_clusters_valid(clusters, data.shape[0]):
            clusters = self._predict(data)

        uq_clusters = np.unique(clusters)
        cluster_distances = {}

        for cluster in uq_clusters:
            cluster_mask = (clusters == cluster
                if cluster_comparison == "within"
                else clusters != cluster)
            cluster_points = data[cluster_mask]
            distances = _minkowski_condensed_distances_numpy_nb(
                cluster_points, p=minkowski_p)
            cluster_distances[cluster] = distances

        return cluster_distances
    

    def _compute_cross_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float,
            clusters: np.ndarray[tuple[int]] | None = None
    ) -> dict[int,
              np.ndarray[tuple[int], np.dtype[np.floating]]]:

        if not self._are_clusters_valid(clusters, data.shape[0]):
            clusters = self._predict(data)

        uq_clusters = np.unique(clusters)
        n_clusters = uq_clusters.shape[0]
        cluster_distances = defaultdict(
            list[np.ndarray[tuple[int], np.dtype[np.floating]]])

        for i in range(n_clusters):
            label = uq_clusters[i]
            cluster_points = data[clusters == label]

            for j in range(i+1, n_clusters):
                other_cluster = uq_clusters[j]
                other_cluster_points = data[clusters == other_cluster]
                distances = _minkowski_pairwise_distances_numpy_nb(
                    cluster_points, other_cluster_points, p=minkowski_p)
                distances = distances.flatten()
                cluster_distances[label].append(distances)
                cluster_distances[other_cluster].append(distances)

        return {
            cluster : np.concat(distances)
            for cluster, distances in cluster_distances.items()}


    def _compute_centre_to_cluster_minkowski_distances(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            minkowski_p: float
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        centres = self._get_centres()
        return _minkowski_pairwise_distances_numpy_nb(centres, data, p=minkowski_p)



def _minkowski_pairwise_distances_numpy(
        X: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        Y: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        p: float,
        block_size: int=128) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    
    dtype = np.result_type(X.dtype, Y.dtype, np.float64)
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=dtype)
    distances = np.empty((len(X), len(Y)), dtype=dtype)

    for start in range(0, len(X), block_size):
        stop = min(start + block_size, len(X))
        differences = np.abs(X[start:stop, None, :] - Y[None, :, :])
        np.pow(differences, p, out=differences)
        np.pow(differences.sum(axis=2), 1/p, out=distances[start:stop])

    return distances


@njit(
    cache=settings.current.numba_caching,
    boundscheck=settings.current.numba_boundschecking,
    error_model=settings.current.numba_error_model,
    fastmath=settings.current.numba_fastmath,
    nogil=settings.current.numba_nogil
)
def _minkowski_pairwise_distances_numba(
        X: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        Y: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        p: float,
        block_size: int = 128) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Compute cross-matrix Euclidean distances using compiled scalar loops.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, p)
        First matrix of observations. Float32 and float64 arrays are supported.
    Y : numpy.ndarray of shape (m, p)
        Second matrix of observations, with the same number of features as X.
    p : float
        `p` parameter for Minkowski/L_p distance.
    block_size : int, default=128
        Positive number of X rows processed per block. Retained for call
        compatibility with the NumPy implementation; no broadcast temporary
        is allocated by this implementation.

    Returns
    -------
    numpy.ndarray of shape (n, m), dtype float64
        Euclidean distance between each row of X and each row of Y.

    Raises
    ------
    ValueError
        If feature counts differ or block_size is not positive.

    Notes
    -----
    Coordinates are promoted to float64 before subtraction and accumulation.
    Direct coordinate differences avoid dot-product cancellation and give
    exact zero for identical finite rows. Extremely large differences can
    still overflow when squared. Working storage, excluding inputs and the
    returned matrix, is constant rather than proportional to n * m * p.
    Execution is serial. GIL release is controlled by
    ``settings.current.numba_nogil``, captured when the module is imported.
    When enabled, callers can run compiled calls concurrently using threads;
    this option does not create threads or parallelize the loops internally.
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    distances = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    for start in range(0, X.shape[0], block_size):
        stop = min(start + block_size, X.shape[0])
        for i in range(start, stop):
            for j in range(Y.shape[0]):
                p_distance = 0.0
                for feature in range(X.shape[1]):
                    # Cast before subtracting: float32 arithmetic could lose
                    # precision even if the accumulator itself is float64.
                    difference = np.abs(np.float64(X[i, feature]) - np.float64(Y[j, feature]))
                    p_distance += np.pow(difference, p)
                distances[i, j] = np.pow(p_distance, 1/p)
    return distances


_minkowski_pairwise_distances_numpy_nb = numba_dispatch(
    _minkowski_pairwise_distances_numba)(_minkowski_pairwise_distances_numpy)


def _minkowski_condensed_distances_numpy(
        X: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        p: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute each unordered within-matrix Minkowski distance once.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, n_features)
        Matrix whose rows are observations. Coordinates are promoted to
        float64 before subtraction.
    p : float
        Finite, strictly positive Minkowski exponent. For 0 < p < 1 the
        result is a quasi-distance, not a metric.

    Returns
    -------
    numpy.ndarray of shape (n * (n - 1) // 2,), dtype float64
        Distances in strict upper-triangle row order: (0, 1), (0, 2), ...,
        (1, 2), ... . Self-pairs are excluded; duplicate observations still
        contribute zero distances. Fewer than two observations give an empty
        array. Zero-feature observations have distance zero.

    Raises
    ------
    ValueError
        If X is not two-dimensional or p is not finite and positive.

    Notes
    -----
    Processing one row against its successors avoids a full (n, n) matrix
    and a global (n * (n - 1) // 2, n_features) temporary. Apart from the
    float64 input conversion and output, temporary storage is O(n * n_features).
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and strictly positive.")
    
    matrix = np.asarray(X, dtype=np.float64)
    n = matrix.shape[0]
    distances = np.empty(n * (n - 1) // 2, dtype=np.float64)
    offset = 0

    for i in range(n - 1):
        count = n - i - 1
        distances[offset:offset + count] = np.linalg.norm(
            matrix[i + 1:] - matrix[i], ord=p, axis=1)
        offset += count

    return distances


@njit(
    cache=settings.current.numba_caching,
    boundscheck=settings.current.numba_boundschecking,
    error_model=settings.current.numba_error_model,
    fastmath=settings.current.numba_fastmath,
    nogil=settings.current.numba_nogil,
)
def _minkowski_condensed_distances_numba(
        X: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        p: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute condensed Minkowski distances using nopython scalar loops.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, n_features)
        Float32 or float64 observations. Noncontiguous and read-only arrays
        are supported. Each coordinate is promoted to float64 before subtraction.
    p : float
        Finite, strictly positive Minkowski exponent. Values below one give
        a quasi-distance.

    Returns
    -------
    numpy.ndarray of shape (n * (n - 1) // 2,), dtype float64
        Distances in strict upper-triangle row order, matching the NumPy
        implementation. Empty and singleton inputs yield empty arrays.

    Raises
    ------
    ValueError
        If X is not two-dimensional or p is not finite and positive.

    Notes
    -----
    No pair-index arrays, converted input copy, or difference matrix are
    allocated. Working storage excluding input and output is constant.
    Compilation options are captured from settings when the module is imported.
    Execution is serial; numba_nogil controls GIL release, not internal threading.
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and strictly positive.")
    
    n = X.shape[0]
    distances = np.empty(n * (n - 1) // 2, dtype=np.float64)
    offset = 0

    for i in range(n - 1):
        # j starts after i: neither self-pairs nor mirrored pairs are visited.
        for j in range(i + 1, n):
            powered_distance = 0.0
            
            for feature in range(X.shape[1]):
                difference = abs(np.float64(X[i, feature]) - np.float64(X[j, feature]))
                powered_distance += difference ** p
            distances[offset] = powered_distance ** (1.0 / p)
            offset += 1

    return distances


_minkowski_condensed_distances_numpy_nb = numba_dispatch(
    _minkowski_condensed_distances_numba)(_minkowski_condensed_distances_numpy)

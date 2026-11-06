"""Parameter handling and estimator resolution for K-Medoids Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, Literal
from abc import ABC
from sklearn_extra.cluster import KMedoids

import pyspoc._argchecking as argchk

from pyspoc.types import DistanceMetric
from .._base import KClusteringEstimator, KClusteringStatistic
from ..._base import (
    _minkowski_pairwise_distances_numpy_nb,
    _minkowski_condensed_distances_numpy_nb)

_LOGGER = logging.getLogger(__name__)



class KMedoidsEstimator(KClusteringEstimator[KMedoids]):
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
    random_seed : int or `None`
        Per-estimator random-seed override. If ``None``, use the active
        library-wide seed.
    initializer : {"k-medoids++", "random", "heuristic", "build"}
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : DistanceMetric
        A string identifier for a supported distance metric. See :class:`DistanceMetric` for the
        list of supported distance metrics and their formulae.

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
            metric: DistanceMetric,
            minkowski_p: float | None):

        super().__init__(
            k=k,
            max_iter=max_iter,
            random_seed=random_seed)
        
        self._fit_precomputed = False
        
        if metric == "minkowski":
            if minkowski_p is None:
                raise ValueError("minkowski_p must be provided when metric is 'minkowski'")
            
            self._fit_precomputed = True
            self._minkowski_p = argchk.check_float(
                minkowski_p, exclusive_minimum=0, arg_name="minkowski_p")
        else:
            self._minkowski_p = None

        self._initializer = initializer
        self._algorithm = algorithm
        self._metric = metric
        

    def _get_centres(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        if not self._fit_precomputed:
            return super()._get_centres()

        model = self._get_model()
        training_data = self._get_attached_dataset()

        if training_data is None:
            raise RuntimeError("Training observations are unavailable.")

        medoids = training_data[model.medoid_indices_]
        return medoids


    def _predict(
            self,
            data: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:

        if not self._fit_precomputed:
            return super()._predict(data)

        model = self._get_model()
        training_data = self._get_attached_dataset()

        if training_data is None:
            raise RuntimeError("Training observations are unavailable.")

        medoids = training_data[model.medoid_indices_]
        D = _minkowski_pairwise_distances_numpy_nb(data, medoids, p=self._minkowski_p)
        labels = D.argmin(axis=1)
        return labels
    

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

        if self._metric == "minkowski":
            dist = _minkowski_condensed_distances_numpy_nb(data, p=self._minkowski_p)
            n = data.shape[0]
            upper_tri_idxs = np.triu_indices(n, k=1)
            D = np.zeros(shape=(n,n))
            D[upper_tri_idxs] = dist
            D += D.T
            model = KMedoids(
                n_clusters=self._k,
                metric="precomputed",
                method=self._algorithm,
                init=self._initializer,
                max_iter=self._max_iter,
                random_state=self.random_seed).fit(D)
            return model
        
        # Translate the library's boolean verbosity setting to scikit-learn's
        # integer verbosity parameter. False becomes 0 and True becomes 1.
        model = KMedoids(
            n_clusters=self._k,
            metric=self._metric,
            method=self._algorithm,
            init=self._initializer,
            max_iter=self._max_iter,
            random_state=self.random_seed)

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
    random_seed : int or `None`
        Per-estimator random-seed override. If ``None``, use the active
        library-wide seed.
    initializer : {"k-medoids++", "random", "heuristic", "build"}
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}
        Which algorithm to use. "alternate" is faster while "pam" is more accurate.
        Corresponds to the "method" argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    metric : DistanceMetric
        A string identifier for a supported distance metric used in the KMedoids algorithm's
        determination of selected medoids. See :class:`DistanceMetric` for the list of supported
        distance metrics and their formulae.
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
            algorithm: Literal["alternate", "pam"],
            metric: DistanceMetric,
            minkowski_p: float | None,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"]):

        estimator_kwargs = {
            "initializer": initializer,
            "algorithm" : algorithm,
            "metric": metric,
            "minkowski_p": minkowski_p
        }
        
        super().__init__(
            clustering_type=KMedoidsEstimator,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            dim=dim,
            nan_policy=nan_policy,
            estimator_kwargs=estimator_kwargs)

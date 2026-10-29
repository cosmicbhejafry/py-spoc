"""Shared parameter handling and estimator contracts for K-clustering Statistics."""
from __future__ import annotations

import logging
import numpy as np

from typing import Any, TypeVar, Generic, Literal
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import pyspoc._argchecking as argchk

from pyspoc._tag import Tag, Signature
from pyspoc._estimators import LazyFittedCachedEstimatorMixin
from pyspoc._random import RandomSeedMixin
from pyspoc.component.statistics.base import Statistic


_LOGGER = logging.getLogger(__name__)



class KClusteringEstimator(
    LazyFittedCachedEstimatorMixin[KMeans | KMedoids]):
    """Manage a scikit-learn backed KClustering model.

    The estimator combines runtime constructor checking, library-wide random
    seed resolution, estimator caching, and synchronized lazy fitting. A fitted
    estimator may be shared by multiple Statistics requesting equivalent
    clustering configuration for equivalent data.

    Parameters
    ----------
    k : int, requires: >=2
        Number of clusters to form.
    max_iter : int, default=300, requires: >=1
        Maximum number of K-Means iterations for a single run.
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
    _freeze_random_seed = True

    def __init__(
        self,
        k: int,
        max_iter: int = 300,
        random_seed: int | None = None, # Handled automatically by RandomSeedMixin
    ) -> None:

        self._k = k # Already checked by KClusteringStatisticMixin
        self._max_iter = max_iter # Already checked by KClusteringStatisticBaseMixin
        super().__init__()
    

    # The abstract fitting hook inherited from LazyFittedCachedEstimatorMixin
    # must be overridden by concrete class.
    # Its public fit() method handles thread-safe locking, data equivalence,
    # and the fitted-state transition around this operation.
    @abstractmethod
    def _fit_estimator(self, data: np.ndarray) -> KMeans | KMedoids:
        pass



TKClusteringEstimator = TypeVar("TKClusteringEstimator", bound=KClusteringEstimator)


class KClusteringStatisticBaseMixin(
    Generic[TKClusteringEstimator],
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
            clustering_type: type[TKClusteringEstimator],
            max_iter: int,
            random_seed: int | None, # Handled automatically by RandomSeedMixin
            estimator_kwargs: dict[str, Any],
            **kwargs: Any
    ):

        self._max_iter = argchk.check_natural_number(max_iter, "max_iter")
        self._clustering_type = clustering_type
        self._estimator_kwargs = estimator_kwargs

        # Continue the cooperative initializer chain so random-seed and other
        # inherited mixins can initialize their state.
        super().__init__(**kwargs)
       

    def _get_fitted_estimator(self, k: int, data: np.ndarray) -> TKClusteringEstimator:
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
        
        self._estimator_ = self._clustering_type.get_or_create(
            data=data,
            max_iter=self._max_iter,
            k=k,
            random_seed=self.random_seed,
            **self._estimator_kwargs
        )

        self._estimator_.fit(data)
        return self._estimator_



class KClusteringSweepStatisticMixin(
    KClusteringStatisticBaseMixin[TKClusteringEstimator],
    ABC):

    """Provides shared KClustering sweep statistic configuration and functionality.

    Parameters
    ----------
    clustering_type : type[KClusteringEstimator]
        The specific KClusteringEstimator type to use.
    min_k : int
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int
        Maximum number of number of clusters to form as part of k-sweep.
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
    """

    def __init__(
            self,
            clustering_type : type[TKClusteringEstimator],
            min_k: int,
            max_k: int,
            max_iter: int,
            random_seed: int | None,
            estimator_kwargs: dict[str, Any],
            **kwargs: Any):

        # A pairwise cluster matrix is only meaningful for at least two
        # clusters.
        argchk.check_integer(min_k, minimum=2, arg_name="min_k")
        self._min_k = min_k

        # Maximum clusters must be greater than minimum clusters
        argchk.check_integer(max_k, minimum=min_k, arg_name="max_k")
        self._max_k = max_k

        # The base performs the remaining K-Clustering checks.
        super().__init__(
            clustering_type=clustering_type,
            max_iter=max_iter,
            random_seed=random_seed,
            estimator_kwargs=estimator_kwargs,
            **kwargs)



class KClusteringStatisticMixin(
    KClusteringStatisticBaseMixin[TKClusteringEstimator],
    ABC):

    """Provides shared KClustering statistic configuration and functionality.

    Parameters
    ----------
    clustering_type : type[KClusteringEstimator]
        The specific KClusteringEstimator type to use.
    k : int, requires: >=2
        Number of clusters to form.
    max_iter : int, default=300, requires: >=1
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

    def __init__(
            self,
            clustering_type : type[TKClusteringEstimator],
            k: int,
            max_iter: int,
            random_seed: int | None,
            estimator_kwargs: dict[str, Any],
            **kwargs: Any):

        # A pairwise cluster matrix is only meaningful for at least two
        # clusters.
        self._k = argchk.check_integer(k, 2, arg_name="k")

        # The base performs the remaining K-Clustering checks.
        super().__init__(
            clustering_type=clustering_type,
            max_iter=max_iter,
            random_seed=random_seed,
            estimator_kwargs=estimator_kwargs,
            **kwargs)



class KClusteringStatistic(
    KClusteringStatisticMixin[TKClusteringEstimator],
    Statistic,
    ABC):

    """
    Underlying KClustering Statistic class with mixin configuration and
    functionality included, ready for inheritence by concrere Statistic
    implementations.

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
    dim : {"n", "p"}
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}
        Treatment of missing elements in the data matrix.
        After orienting the requested variables,
        `"raise"` throws a :class:`ValueError` on encountering a NaN;
        `"row-omit"` removes complete rows containing a NaN;
        `"col-omit"` removes complete columns containg a NaN.
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
            clustering_type : type[TKClusteringEstimator],
            k: int,
            max_iter: int,
            random_seed: int | None,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise","row-omit","col-omit"],
            estimator_kwargs: dict[str, Any],
            **kwargs: Any):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism="parallel",
            thread_safety="unsafe",
            clustering_type=clustering_type,
            k=k,
            max_iter=max_iter,
            random_seed=random_seed,
            estimator_kwargs=estimator_kwargs,
            **kwargs)
        
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
        
        if n < self._k:
            raise ValueError(
                "The number of clusters (k) must not exceed the number of data points.")

        # Resolve dimensions against this particular dataset without mutating
        # the originally requested component configuration.
        fitted_estimator = self._get_fitted_estimator(k=self._k, data=data)
        return self._get_result(data=data, fitted_estimator=fitted_estimator)


    @abstractmethod
    def _get_result(
            self,
            data: np.ndarray,
            fitted_estimator: TKClusteringEstimator
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Extract a concrete result from a shared fitted estimator.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional point-cloud dataset.
        fitted_estimator : KClusteringEstimator
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

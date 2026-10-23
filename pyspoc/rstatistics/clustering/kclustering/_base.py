import numpy as np

from typing import Literal, Any, TypeVar
from abc import ABC, abstractmethod

from pyspoc._tag import Tag, Signature
from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.statistics.clustering.kclustering._base import (
    KClusteringSweepStatisticMixin,
    KClusteringEstimator)


TKClusteringEstimator = TypeVar("TKClusteringEstimator", bound=KClusteringEstimator)



class KClusteringSweepStatistic(
    KClusteringSweepStatisticMixin[TKClusteringEstimator],
    ReducedStatistic,
    ABC):

    """
        Underlying KClustering sweeping Statistic class with mixin configuration and
        functionality included, ready for inheritence by concrete Statistic
        implementations.
    
        Parameters
        ----------
        clustering_type : type[KClusteringEstimator]
            The specific KClusteringEstimator type to use.
        min_k : int, requires: >=2
            Minimum number of clusters to form in the sweep.
        max_k : int, requires: >=`min_k`
            Maximum number of clusters to form in the sweep.
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
            min_k: int,
            max_k: int,
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
            min_k=min_k,
            max_k=max_k,
            max_iter=max_iter,
            random_seed=random_seed,
            estimator_kwargs=estimator_kwargs,
            **kwargs)


    # Override Statistic.compute(), whose abstract contract defines the public
    # entry point used by Calculator.

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Fit or reuse an estimator and derive the concrete Statistic result.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional point-cloud dataset.

        Returns
        -------
        numpy.ndarray[tuple[int], np.dtype[np.floating]]
            One-dimensional floating numpy array ReducedStatistic result.
        """
        n = data.shape[0]
        
        if n < self._max_k:
             raise ValueError(
                  "The maximum number of clusters (max_k) " \
                  "must not exceed the number of data points.")

        results = []

        for k in range(self._min_k, self._max_k + 1):

            # Resolve dimensions against this particular dataset without mutating
            # the originally requested component configuration.
            fitted_estimator = self._get_fitted_estimator(k=k, data=data)
            result = self._get_result(data=data, fitted_estimator=fitted_estimator)

            # Wrap in numpy array if required.
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float64)
            
            # Collect the result
            results.append(result.squeeze())

        # Convert to a numpy array.
        return np.hstack(results)
    

    @abstractmethod
    def _get_result(
            self,
            data: np.ndarray,
            fitted_estimator: TKClusteringEstimator
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating]] |
          np.ndarray[tuple[()], np.dtype[np.floating]] |
          np.floating |
          float):
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
        numpy.ndarray[tuple[int], numpy.dtype[numpy.floating]] |
        numpy.ndarray[tuple[()], numpy.dtype[numpy.floating]] |
        numpy.floating |
        float)
            One-dimensional floating numpy array or float scalar result selected
            by the concrete implementation of :meth:`_get_result`.

        Notes
        -----
        This abstract hook must be implemented by every concrete :class:`KClusteringStatistic`.
        Implementations may inspect the borrowed estimator but must not retrain
        or mutate its shared model.
        """
        pass

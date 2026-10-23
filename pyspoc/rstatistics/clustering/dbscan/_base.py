import numpy as np

from typing import Literal
from abc import abstractmethod

import pyspoc._argchecking as argchk
from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.statistics.clustering.dbscan._base import (
    DBScanStatisticBaseMixin,
    DBScanEstimator)


class DBScanSweepStatistic(
    DBScanStatisticBaseMixin,
    ReducedStatistic):

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
            min_eps: float,
            max_eps: float,
            n_results: int,
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
        self._min_eps = argchk.check_float(min_eps, exclusive_minimum=0, arg_name="min_eps")

        # Maximum eps must be greater than minimum eps.
        self._max_eps = argchk.check_float(max_eps, minimum=min_eps, arg_name="max_eps")

        # Must return at least one result.
        self._n_results = argchk.check_integer(n_results, minimum=1, arg_name="n_results")

        # The base performs the remaining DBScan checks.
        super().__init__(
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs,
            dim=dim,
            nan_policy=nan_policy)


    # Override Statistic.compute(), whose abstract contract defines the public
    # entry point used by Calculator.
    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
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
        eps_set = np.linspace(
            self._min_eps,
            self._max_eps,
            self._n_results)

        results = []
        
        for eps in eps_set:
            fitted_estimator = self._get_fitted_estimator(eps=eps, data=data)
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
            fitted_estimator: DBScanEstimator
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating]] |
          np.ndarray[tuple[()], np.dtype[np.floating]] |
          np.floating |
          float):
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

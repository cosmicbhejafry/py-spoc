import numpy as np

from typing import Literal
from pyspoc.types import DistanceMetric
from abc import abstractmethod

import pyspoc._argchecking as argchk
from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.statistics.clustering.dbscan._base import (
    DBScanStatisticBaseMixin,
    DBScanEstimator)



class DBScanSweepStatistic(
    DBScanStatisticBaseMixin,
    ReducedStatistic):

    """Provides shared DBScan sweep statistic configuration and functionality.

    Parameters
    ----------
    min_eps : float, requires: >0
        Minimum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
    max_eps : float, requires: >=`min_eps`
        Maximum number of `eps` parameter as part of DBScan sweep. See **Notes** for
        information on the `eps` parameter.
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


    The maximum distance between two samples for one to be considered as in the neighborhood of
    the other. This is not a maximum bound on the distances of points within a cluster. This is
    the most important DBSCAN parameter to choose appropriately for your data set and distance
    function. Smaller values generally lead to more clusters.

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
            algorithm: Literal[
                "auto", "ball_tree", "kd_tree", "brute"],
            leaf_size: int,
            metric: DistanceMetric,
            minkowski_p: float | None,
            n_jobs: int,
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
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            minkowski_p=minkowski_p,
            n_jobs=n_jobs,
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism="parallel")


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

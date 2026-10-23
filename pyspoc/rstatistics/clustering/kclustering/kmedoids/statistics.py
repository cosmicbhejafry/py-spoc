import numpy as np

from pyspoc.statistics.clustering.kclustering.kmedoids._base import KMedoidsEstimator
from ._base import KMedoidsSweepStatistic
from .._mixin import KClusteringErrorsMixin, KClusteringSILScoresMixin


class KMedoidsErrors(
    KMedoidsSweepStatistic,
    KClusteringErrorsMixin):
    """
    Returns a series of errors from KMedoids fits with a sweep of `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300, requires >1
        Maximum number of K-Medoids iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. `"alternate"` is faster while `"pam"` is more accurate.
        Corresponds to the `method` argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=`settings.current.data_nan_policy`
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

    """

    _long_name = "KMedoids - Error Distribution"
    _identifier = "kmedoids-error"

    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMedoidsEstimator) -> float:
                
        return KClusteringErrorsMixin._get_result(
            self, data, fitted_estimator)



class KMedoidsSILScores(
    KMedoidsSweepStatistic,
    KClusteringSILScoresMixin):
    """
    Returns a series of errors from KMedoids fits with a sweep of `k` parameters.


    Parameters
    ----------
    min_k : int, requires: >1
        Minimum number of number of clusters to form as part of k-sweep.
    max_k : int, requires: >min_k
        Maximum number of number of clusters to form as part of k-sweep.
    max_iter : int, default=300, requires >1
        Maximum number of K-Medoids iterations for a single run.
    random_seed : int or None, optional
        Per-estimator random-seed override. If ``None``, the active
        library-wide seed is used.
    initializer : {"k-medoids++", "random", "heuristic", "build"}, default="heuristic"
        Medoid initialization strategy passed to scikit-learn.
    algorithm : {"alternate", "pam"}, default="alternate"
        Which algorithm to use. `"alternate"` is faster while `"pam"` is more accurate.
        Corresponds to the `method` argument in the KMedoids constructor:
        :meth:`sklearn_extra.cluster.KMedoids.__init__`.
    dim : {"n", "p"}, default="p"
        Axis whose elements become variables in the returned matrix.
        ``"p"`` compares data columns; ``"n"`` compares data rows.
    nan_policy : {"raise", "row-omit", "col-omit"}, default=`settings.current.data_nan_policy`
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

    """

    _long_name = "KMedoids - Silhouette Score Distribution"
    _identifier = "kmedoids-sil"

    def _get_result(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            fitted_estimator: KMedoidsEstimator) -> float:
                
        return KClusteringSILScoresMixin._get_result(
            self, data, fitted_estimator)

"""Abstract Statistic adapter for the KMeans estimator family."""

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pyspoc.statistics.base import Statistic
from ._mixin import KClusteringMixin

if TYPE_CHECKING:
    from ._estimator import KClusteringEstimator


class KClusteringStatistic(KClusteringMixin, Statistic, ABC):
    """
    Bridge shared KMeans machinery into the Statistic hierarchy.

    The private method :meth:`_get_result` must be implemented by concrete
    subclasses.
    """

    # Override Statistic.compute(), whose abstract contract defines the public
    # entry point used by Calculator.
    def compute(self, data: np.ndarray) -> np.ndarray | float:
        """Fit or reuse an estimator and derive the concrete Statistic result.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional dataset supplied by :class:`Calculator`.

        Returns
        -------
        numpy.ndarray or float
            Result selected by the concrete implementation of
            :meth:`_get_result`.
        """
        # Resolve dimensions against this particular dataset without mutating
        # the originally requested component configuration.
        fitted_estimator = self._compute_estimator_output(data)
        return self._get_result(data, fitted_estimator)

    @abstractmethod
    def _get_result(self, data: np.ndarray, fitted_estimator: KClusteringEstimator) -> np.ndarray:
        """Extract a concrete result from a shared fitted estimator.

        Parameters
        ----------
        fitted_estimator : KMeansEstimator
            Estimator fitted to the current dataset. The estimator and
            its internal model are shared and must be treated as read-only.

        Returns
        -------
        numpy.ndarray
            Concrete Statistic result.

        Notes
        -----
        This abstract hook must be implemented by every concrete KMeans Statistic.
        Implementations may inspect the borrowed estimator but must not retrain
        or mutate its shared model.
        """

        pass

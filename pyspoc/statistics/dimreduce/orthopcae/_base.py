"""Abstract Statistic adapter for the OrthogonalPCAE estimator family."""

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

from pyspoc.statistic import Statistic
from ._mixin import OrthogonalPCAEMixin

if TYPE_CHECKING:
    from ._estimator import OrthogonalPCAEEstimator


class OrthogonalPCAEStatistic(OrthogonalPCAEMixin, Statistic, ABC):
    """
    Bridge shared OrthogonalPCAE machinery into the Statistic hierarchy.

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
        resolved_parameters = self._resolve_parameters(data)
        fitted_estimator = self._compute_estimator_output(data, resolved_parameters)
        return self._get_result(fitted_estimator, resolved_parameters.components)

    @abstractmethod
    def _get_result(
        self, fitted_estimator: OrthogonalPCAEEstimator, components: tuple[int, ...]
    ) -> Union[np.ndarray, float]:
        """Extract a concrete result from a shared fitted estimator.

        Parameters
        ----------
        fitted_estimator : OrthogonalPCAEEstimator
            Estimator fitted to the current dataset. The estimator and
            its internal model are shared and must be treated as read-only.
        components : tuple[int, ...]
            One-based components available for this dataset used to filter
            results.

        Returns
        -------
        numpy.ndarray or float
            Concrete Statistic result.

        Notes
        -----
        This abstract hook must be implemented by every concrete OrthogonalPCAE
        Statistic. Implementations may inspect the borrowed estimator but must
        not retrain or mutate its shared model.
        """

        pass

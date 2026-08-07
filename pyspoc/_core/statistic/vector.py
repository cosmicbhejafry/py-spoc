from abc import ABC, abstractmethod
from typeguard import check_type

from pyspoc._core.types import NumpyDataMatrix, NumpyRealVector
from .base import Statistic

class VectorStatistic(Statistic, ABC):

    @abstractmethod
    def compute(self, data: NumpyDataMatrix) -> NumpyRealVector:
        pass

    def _get_validated_result(
            self,
            result: NumpyRealVector) -> NumpyRealVector:

        """Validate and return a value produced by :meth:`compute`.

        Parameters
        ----------
        result : :class:`NumpyRealVector`
            Candidate statistic result. It must be a Numpy array with real-valued
            dtype of a one dimension.

        Returns
        -------
        :class:`NumpyRealVector`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a one-dimensional, real-valued Numpy array.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`compute` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """
        # Typeguard delegates the array-specific part of this annotation to
        # jaxtyping. Both symbolic axes occur in this single check, so no
        # shared argument/return dimension context is required here.
        check_type(result, NumpyRealVector)

        # Validation must not copy or otherwise transform statistic results.
        return result

import numpy as np

from abc import ABC, abstractmethod
from typeguard import check_type
from typing import TypeVar, Generic

from pyspoc.core.reducer._base import Reducer
from pyspoc.core.types import NumpyFloatSquareMatrix


TMatrix = TypeVar("TMatrix", bound=NumpyFloatSquareMatrix)

class MatrixToVectorReducer(
    Reducer[TMatrix, np.ndarray[tuple[int], np.dtype[np.floating]]],
    Generic[TMatrix],
    ABC):

    @abstractmethod
    def _reduce(
            self,
            data: NumpyFloatSquareMatrix) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        pass


    def _get_validated_result(
            self,
            result: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        
        """Validate and return a value produced by :meth:`_reduce`.

        Parameters
        ----------
        result : `np.ndarray[tuple[int], np.dtype[np.floating]]`
            Candidate vector reducer result. It must be a
            one-dimensional Numpy array of :class:`np.floating` dtype.

        Returns
        -------
        `np.ndarray[tuple[int], np.dtype[np.floating]]`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a one-dimensional Numpy array
            of :class:`np.floating` dtype.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`_reduce` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """

        check_type(result, np.ndarray[tuple[int], np.dtype[np.floating]])

        return result

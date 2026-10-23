import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToScalarReducer
from pyspoc.core.types import NumpyRealSquareMatrix


class SquareMatrixToScalarReducer(
    MatrixToScalarReducer[NumpyRealSquareMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSquareMatrix) -> float | np.floating:
        pass

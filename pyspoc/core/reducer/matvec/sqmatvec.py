import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToVectorReducer
from pyspoc.core.types import NumpyFloatSquareMatrix


class SquareMatrixToScalarReducer(
    MatrixToVectorReducer[NumpyFloatSquareMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyFloatSquareMatrix) -> float | np.floating:
        pass

import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToScalarReducer
from pyspoc.core.types import NumpyFloatSquareMatrix


class SquareMatrixToScalarReducer(
    MatrixToScalarReducer[NumpyFloatSquareMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyFloatSquareMatrix) -> float | np.floating:
        pass

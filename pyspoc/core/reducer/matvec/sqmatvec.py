import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToVectorReducer
from pyspoc.core.types import NumpyRealSquareMatrix


class SquareMatrixToVectorReducer(
    MatrixToVectorReducer[NumpyRealSquareMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSquareMatrix) -> float | np.floating:
        pass

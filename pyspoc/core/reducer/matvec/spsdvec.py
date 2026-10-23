import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToVectorReducer
from pyspoc.core.types import NumpyRealSPSDMatrix


class SPSDMatrixToVectorReducer(
    MatrixToVectorReducer[NumpyRealSPSDMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSPSDMatrix) -> float | np.floating:
        pass

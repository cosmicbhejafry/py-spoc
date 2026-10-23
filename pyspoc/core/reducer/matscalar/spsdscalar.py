import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToScalarReducer
from pyspoc.core.types import NumpyRealSPSDMatrix


class SPSDMatrixToScalarReducer(
    MatrixToScalarReducer[NumpyRealSPSDMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSPSDMatrix) -> float | np.floating:
        pass

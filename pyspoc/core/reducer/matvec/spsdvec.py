import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToVectorReducer
from pyspoc.core.types import NumpyFloatSPSDMatrix


class SPSDMatrixToScalarReducer(
    MatrixToVectorReducer[NumpyFloatSPSDMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyFloatSPSDMatrix) -> float | np.floating:
        pass

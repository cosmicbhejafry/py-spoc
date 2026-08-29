import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToScalarReducer
from pyspoc.core.types import NumpyFloatSPSDMatrix


class SPSDMatrixToScalarReducer(
    MatrixToScalarReducer[NumpyFloatSPSDMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyFloatSPSDMatrix) -> float | np.floating:
        pass

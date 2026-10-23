import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToScalarReducer
from pyspoc.core.types import NumpyRealSymmetricMatrix


class SymmetricMatrixToScalarReducer(
    MatrixToScalarReducer[NumpyRealSymmetricMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSymmetricMatrix) -> float | np.floating:
        pass

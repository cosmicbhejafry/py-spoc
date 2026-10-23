import numpy as np

from abc import ABC, abstractmethod

from ._base import MatrixToVectorReducer
from pyspoc.core.types import NumpyRealSymmetricMatrix


class SymmetricMatrixToVectorReducer(
    MatrixToVectorReducer[NumpyRealSymmetricMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: NumpyRealSymmetricMatrix) -> float | np.floating:
        pass

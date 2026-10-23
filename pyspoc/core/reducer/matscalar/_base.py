import numpy as np

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from pyspoc.core.reducer._base import Reducer
from pyspoc.core.types import NumpyRealSquareMatrix


TMatrix = TypeVar("TMatrix", bound=NumpyRealSquareMatrix)

class MatrixToScalarReducer(
    Reducer[TMatrix, float | np.floating],
    Generic[TMatrix],
    ABC):


    @abstractmethod
    def _reduce(
            self,
            data: TMatrix) -> float | np.floating:
        pass

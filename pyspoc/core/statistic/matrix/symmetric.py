import numpy as np

from abc import ABC, abstractmethod
from typing import final, cast

from .square import SquareMatrixStatistic
from pyspoc.core.types import NumpyRealSymmetricMatrix


class SymmetricMatrixStatistic(SquareMatrixStatistic, ABC):

    @final
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> NumpyRealSymmetricMatrix:

        result = super().summarize(data)
        return cast(NumpyRealSymmetricMatrix, result)
    

    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyRealSymmetricMatrix:
        pass


    def validate_output(self, result: NumpyRealSymmetricMatrix) -> NumpyRealSymmetricMatrix:
        super().validate_output(result)

        if not np.allclose(result, result.conj().T):
            raise ValueError(f"{self.name} must return a Hermitian matrix.")

        return result

import numpy as np

from abc import ABC, abstractmethod

from .square import SquareMatrixStatistic
from pyspoc._core.types import NumpyDataMatrix, NumpyRealSPSDMatrix


class SPSDMatrixStatistic(SquareMatrixStatistic, ABC):

    def _validate_result(self, result: NumpyRealSPSDMatrix):
        if not np.allclose(result, result.conj().T):
            raise ValueError(f"{self.__name__} must return a Hermitian matrix.")

        eigenvalues = np.linalg.eigvalsh(result)

        if eigenvalues[0] < -1e-10:
            raise ValueError(
                f"{self.__name__} must return a positive-semidefinite matrix."
            )

        return result


    @abstractmethod
    def compute(self, data: NumpyDataMatrix) -> NumpyRealSPSDMatrix:
        pass

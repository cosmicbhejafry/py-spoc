import numpy as np

from abc import ABC, abstractmethod
from typing import final, cast

from .symmetric import SymmetricMatrixStatistic
from pyspoc.core.types import NumpyRealSPSDMatrix


class SPSDMatrixStatistic(SymmetricMatrixStatistic, ABC):

    @final
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> NumpyRealSPSDMatrix:

        result = super().summarize(data)
        return cast(NumpyRealSPSDMatrix, result)
    

    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyRealSPSDMatrix:
        pass


    def validate_output(self, result: NumpyRealSPSDMatrix) -> NumpyRealSPSDMatrix:
        super().validate_output(result)

        eigenvalues = np.linalg.eigvalsh(result)

        if eigenvalues[0] < -1e-10:
            raise ValueError(
                f"{self.name} must return a positive-semidefinite matrix."
            )

        return result

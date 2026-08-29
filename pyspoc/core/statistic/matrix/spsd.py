import numpy as np

from abc import ABC, abstractmethod
from typing import final, cast

from .square import SquareMatrixStatistic
from pyspoc.core.types import NumpyFloatSPSDMatrix


class SPSDMatrixStatistic(SquareMatrixStatistic, ABC):

    @final
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> NumpyFloatSPSDMatrix:

        result = super().summarize(data)
        return cast(NumpyFloatSPSDMatrix, result)
    

    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyFloatSPSDMatrix:
        pass


    def _get_validated_result(self, result: NumpyFloatSPSDMatrix) -> NumpyFloatSPSDMatrix:
        super()._get_validated_result(result)

        if not np.allclose(result, result.conj().T):
            raise ValueError(f"{self.name} must return a Hermitian matrix.")

        eigenvalues = np.linalg.eigvalsh(result)

        if eigenvalues[0] < -1e-10:
            raise ValueError(
                f"{self.name} must return a positive-semidefinite matrix."
            )

        return result

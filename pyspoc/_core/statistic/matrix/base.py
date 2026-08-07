from __future__ import annotations

from typing import final
from abc import ABC, abstractmethod

from pyspoc._core.statistic import Statistic
from pyspoc._core.np_types import NumpyDataMatrix, NumpyRealMatrix

class MatrixStatistic(Statistic, ABC):

    """
    Contract
    ----------
        Input: (n x p) -> Output: (p x p)
    """

    @abstractmethod
    def compute(self, data: NumpyDataMatrix) -> NumpyRealMatrix:
        pass

    @final
    @classmethod
    def _get_component_type(cls) -> type:
        return MatrixStatistic

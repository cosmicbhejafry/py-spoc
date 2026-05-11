import numpy as np

from typing import Literal

from pyspoc.statistic import ReducedStatistic
from pyspoc.reducer import Reducer


class Norm(Reducer, ReducedStatistic):

    _identifier = "norm"
    _labels = ["scalar"]

    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def __init__(self, order: float | Literal["fro", "nuc"] | None):
        self._order : float | Literal["fro", "nuc"] | None = order
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        return np.array(np.linalg.norm(x=data,
                                       ord=self._order))


class EntryWiseMatrixNorm(Reducer, ReducedStatistic):

    _name = "Entry Wise Norm (L_p,q)"
    _identifier = "ew-norm"
    _labels = ["scalar"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def __init__(self, p: int, q: int):
        self._p = p
        self._q = q
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        component_wise_power = abs(data)**self._p
        inner_sums = component_wise_power.sum(axis=1)**(self._p / self._q)
        outer_sum = inner_sums.sum()**(1 / self._q)
        return outer_sum


class SchattenNorm(Reducer, ReducedStatistic):

    _name = "Schatten Norm"
    _identifier = "sch-norm"
    _labels = ["scalar"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def __init__(self, p: float):
        self._p = p
        super().__init__()

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        svs = np.linalg.svd(data, compute_uv=False)
        svs_power_sum = (svs**self._p).sum()
        return svs_power_sum**(1 / self._p)

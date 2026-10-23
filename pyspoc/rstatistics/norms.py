import numpy as np

from typing import Literal

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.reducers.base import Reducer
from pyspoc.settings import settings


class Norm(Reducer, ReducedStatistic):

    _identifier = "norm"
    _labels = ["scalar"]

    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def __init__(
            self,
            order: float | Literal["fro", "nuc"] | None,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
                    
        self._order : float | Literal["fro", "nuc"] | None = order
        super().__init__(dim=dim, nan_policy=nan_policy)

    def reduce(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        return np.array(
            np.linalg.norm(
                x=data.astype(np.float64),
                ord=self._order))


    def _compute(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        return self.reduce(data)


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
    def tags(self) -> list[str]:
        return self._labels

    def __init__(
            self,
            p: int,
            q: int,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        self._p = p
        self._q = q
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)

    def reduce(
            self,
            data: np.ndarray) -> np.ndarray | np.floating | float:
        component_wise_power = abs(data.astype(np.float64))**self._p
        inner_sums = component_wise_power.sum(axis=1)**(self._p / self._q)
        outer_sum = inner_sums.sum()**(1 / self._q)
        return outer_sum


    def _compute(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        return self.reduce(data)


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
    def tags(self) -> list[str]:
        return self._labels

    def __init__(
            self,
            p: int,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        self._p = p
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)

    def reduce(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        svs = np.linalg.svd(data.astype(np.float64), compute_uv=False)
        svs_power_sum = (svs**self._p).sum()
        return svs_power_sum**(1 / self._p)

    def _compute(self, data: np.ndarray) -> np.ndarray | np.floating | float:
        return self.reduce(data)

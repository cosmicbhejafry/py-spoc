import numpy as np
import scipy.stats as sp

from typing import Literal

from pyspoc.component.reducers.base import Reducer
from pyspoc.component.rstatistics.base import ReducedStatistic
from pyspoc.settings import settings


class Moment(Reducer):

    _name = "Moment"
    _identifier = "moment"
    _labels = ["vector"]
    
    def __init__(self, moments: list[int] = [2,4]):
        self._moments = moments
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        mom = sp.moment(data, order = self._moments)
        return mom

class SingularValues(Reducer, ReducedStatistic):

    _name = "SVD"
    _identifier = "svd"
    _labels = ["vector"]

    def __init__(
            self,
            num_values: int = 2,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        self._num_values = num_values

        super().__init__(
            dim=dim,
            nan_policy=nan_policy)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        svs = np.linalg.svd(data, compute_uv=False)
        return svs[:self._num_values]

    def _compute(self, data: np.ndarray) -> np.ndarray | float:
        return self.reduce(data)


class EigenValues(Reducer):

    _name = "Eigen"
    _identifier = "eig"
    _labels = ["square", "vector"]

    def __init__(self, num_values: int = 2):
        self._num_values = num_values
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        eigs = np.linalg.eigvals(data)
        return eigs[:self._num_values]


class Diag(Reducer):

    _name = "Diag"
    _identifier = "diag"
    _labels = ["vector"]

    def __init__(self, num_values: int = 2):
        self._num_values = num_values
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels
    
    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        diag = np.diag(data)
        return diag[:self._num_values]


class Trace(Reducer):

    _name = "Matrix trace"
    _identifier = "tr"
    _labels = ["scalar"]

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        return np.trace(data.astype(np.float64))


class Determinant(Reducer):

    _name = "Matrix determinant"
    _identifier = "det"
    _labels = ["square", "scalar"]

    def __init__(self, scaled: bool = True):
        self._scaled = scaled

        if scaled:
            self._identifier += "-scaled"

        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def tags(self) -> list[str]:
        return self._labels

    def reduce(self, data: np.ndarray) -> np.ndarray | float:
        sign, logabsdet = np.linalg.slogdet(data)
        absdet = np.exp(logabsdet)

        if absdet == np.inf:
            return sign * absdet

        if self._scaled:
            return sign * (absdet ** (1 / data.ndim))

        return sign * absdet

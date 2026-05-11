import numpy as np
import scipy.stats as sp

from pyspoc.reducer import Reducer
from pyspoc.statistic import ReducedStatistic

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
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        mom = sp.moment(data, order = self._moments)
        return mom

class SingularValues(Reducer, ReducedStatistic):

    _name = "SVD"
    _identifier = "svd"
    _labels = ["vector"]

    def __init__(self, num_values: int = 2):
        self.__num_values = num_values
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        svs = np.linalg.svd(data, compute_uv=False)
        return svs[:self.__num_values]


class EigenValues(Reducer):

    _name = "Eigen"
    _identifier = "eig"
    _labels = ["square", "vector"]

    def __init__(self, num_values: int = 2):
        self.__num_values = num_values
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        eigs = np.linalg.eigvals(data)
        return eigs[:self.__num_values]


class Diag(Reducer):

    _name = "Diag"
    _identifier = "diag"
    _labels = ["vector"]

    def __init__(self, num_values: int = 2):
        self.__num_values = num_values
        super().__init__()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    def compute(self, data: np.ndarray) -> np.ndarray | float:
        diag = np.diag(data)
        return diag[:self.__num_values]


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
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        return np.trace(data)


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
    def labels(self) -> list[str]:
        return self._labels

    def compute(self, data: np.ndarray) -> np.ndarray | float:
        det = np.linalg.det(data)

        if self._scaled:
            return det ** -data.ndim

        return det

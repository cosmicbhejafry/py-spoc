import numpy as np

from jaxtyping import Float
from numpy import ndarray
from typing import TypeAlias

NumpyFloatVector = Float[ndarray, "m"]
NumpyFloatSquareMatrix = Float[ndarray, "m m"]
NumpyFloatSPSDMatrix = Float[ndarray, "m m"]
NumpyFloatCubicTensorUpTo1D: TypeAlias = (
    float |
    np.floating |
    NumpyFloatVector
)
NumpyFloatCubicTensor1DTo2D: TypeAlias = (
    NumpyFloatVector |
    NumpyFloatSquareMatrix
)
NumpyFloatCubicTensorUpTo2D: TypeAlias = (
    float |
    np.floating |
    NumpyFloatVector |
    NumpyFloatSquareMatrix)

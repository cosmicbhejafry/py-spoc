import numpy as np

from jaxtyping import Real
from numpy import ndarray
from typing import TypeAlias, NewType

NumpyRealScalar: TypeAlias = (
    Real[ndarray, ""] |
    np.floating |
    np.floating |
    float |
    int)

NumpyRealVector = Real[ndarray, "m"]
NumpyRealMatrix = Real[ndarray, "m q"]
NumpyRealSquareMatrix = Real[ndarray, "m m"]
NumpyRealSymmetricMatrix = Real[ndarray, "m m"]
NumpyRealSPSDMatrix = Real[ndarray, "m m"]

NumpyRealTensorUpTo1D: TypeAlias = (
    NumpyRealScalar |
    NumpyRealVector
)
NumpyRealTensor1DTo2D: TypeAlias = (
    NumpyRealVector |
    NumpyRealMatrix
)
NumpyRealTensorUpTo2D: TypeAlias = (
    NumpyRealTensorUpTo1D |
    NumpyRealMatrix)

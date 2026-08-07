from jaxtyping import Real
from numpy import ndarray
from numbers import Real as RealNumber
from typing import TypeAlias

NumpyDataMatrix = Real[ndarray, "n p"]
NumpyRealMatrix = Real[ndarray, "m q"]
NumpyRealSquareMatrix = Real[ndarray, "m m"]
NumpyRealSPSDMatrix = Real[ndarray, "m m"]
NumpyRealVector = Real[ndarray, "m"]
NumpyRealTensorAtMost2D: TypeAlias = RealNumber | NumpyRealVector | NumpyRealMatrix

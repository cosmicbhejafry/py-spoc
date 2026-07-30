import numpy as np

from pyspoc._numba import numba_dispatch
from .numba import array_equal_numba


@numba_dispatch(array_equal_numba)
def array_equal(x: np.ndarray, y: np.ndarray, equal_nan: bool = False):
    """
    Return whether two one-dimensional arrays are elementwise equal.

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional arrays with the same length.

    equal_nan : bool, default=False
        Treat nan entries as equal.

    Returns
    -------
    bool
        ``True`` when every corresponding element is equal, otherwise ``False``.

    Notes
    -----
    This manual loop avoids temporary boolean arrays inside Numba kernels. The
    function assumes ``x`` and ``y`` have matching lengths.
    """
    return np.array_equal(x, y, equal_nan=equal_nan)

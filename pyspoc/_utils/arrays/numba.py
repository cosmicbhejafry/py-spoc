import numpy as np

from numba import njit


@njit
def array_equal_numba(x: np.ndarray, y: np.ndarray, equal_nan: bool = False):
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
    for j in range(x.shape[0]):
        if x[j] != y[j]:
            if equal_nan and np.isnan(x[j]) and np.isnan(y[j]):
                continue

            return False

    return True

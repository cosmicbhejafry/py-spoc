import numpy as np

def _get_unordered_array_key(x: np.ndarray) -> tuple[str, int, bytes]:
    """
    Convert a NumPy array into a hashable, order-invariant cache key.

    Arrays with the same elements in different orders produce the same key.
    """

    x = np.asarray(x)
    values = np.sort(x.ravel())
    values = np.ascontiguousarray(values)

    return (
        values.dtype.str,
        values.size,
        values.tobytes()
    )


def _get_array_from_key(x_key: tuple[str, int, bytes]):
    """
    Reconstruct the canonical sorted 1D array from the cache key.

    This does NOT reconstruct the original order. It reconstructs the sorted
    canonical version used for caching
    """
    dtype_str, size, data = x_key

    x = np.frombuffer(
        data,
        dtype=np.dtype(dtype_str),
        count=size,
    )

    # Copy so the returned array is writable and independent of the bytes object.
    return x.copy()


    

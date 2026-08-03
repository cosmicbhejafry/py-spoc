import numpy as np
import scipy.linalg as la


def spsd_matrix_power(
    A: np.ndarray,
    pow: float,
    relative_tolerance: float | None = None,
    absolute_tolerance: float = 0.0,
):

    if A.ndim != 2:
        raise ValueError(f"Input matrix A must be 2-dimensional, but got shape {A.shape}.")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix A must be square, but got shape {A.shape}.")

    # Return identity matrix if power = 0
    if pow == 0:
        return np.eye(N=A.shape[0])

    # 1. Compute eigenvalues (w) and eigenvectors (v)
    w, v = la.eigh(A)

    if relative_tolerance is None:
        relative_tolerance = A.shape[0] * float(np.finfo(w.dtype).eps)

    tolerance = max(absolute_tolerance, relative_tolerance * np.max(np.abs(w)))

    # Eigenvalues materially below zero indicate that the input is not PSD.
    if np.any(w < -tolerance):
        raise ValueError("Matrix must be symmetric positive semidefinite.")

    # 2. Apply the exponent to the eigenvalues that mean the requirements.
    w_pow = np.zeros_like(w)
    w_retained = w > tolerance
    w_pow[w_retained] = np.power(w[w_retained], pow)

    # 3. Reconstruct the matrix
    A_pow = (v * w_pow) @ v.T

    return (A_pow + A_pow.T) / 2

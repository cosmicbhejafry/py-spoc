import numpy as np

from pyspoc.core.types import NumpyRealSPSDMatrix
from .symmat import SymmetricMatrixValidator
from ._base import Validator


class SPSDMatrixValidator(
    Validator[NumpyRealSPSDMatrix],
    validates=NumpyRealSPSDMatrix):


    def validate(self, data: NumpyRealSPSDMatrix):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatSPSDMatrix`
                A floating point semi-positive definite matrix,
                hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions, equal sized axes,
                  symmetric constraint ``data[i,j] == data[j,i]`` fulfilled and non-negative
                  eigenvalues (implemented as greater than or equal to `eps = -1e-10` to handle
                  floating point errors).

            Raises
            ------
            TypeCheckError
                If ``data`` does not meet the typing or semantic requirements.

            ValueError
                If ``data`` does not meet semantic requirements.

            Notes
            -----
            Subclasses should override this method with narrower jaxtyping annotations
            and/or additional semantic checks, calling ``super().validate_io(data)``
            if any base contract requirements must also be met.
        """

        # Validate the array is a symmetric matrix.
        SymmetricMatrixValidator().validate(data)
        
        # Then validate it is semi-positive definite.
        eigenvalues = np.linalg.eigvalsh(data)

        if eigenvalues[0] < -1e-10:
            raise ValueError("Array must be a positive-semidefinite matrix.")

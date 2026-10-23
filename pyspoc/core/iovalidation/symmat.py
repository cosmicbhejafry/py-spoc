import numpy as np

from pyspoc.core.types import NumpyRealSymmetricMatrix
from .sqmat import SquareMatrixValidator
from ._base import Validator


class SymmetricMatrixValidator(
    Validator[NumpyRealSymmetricMatrix],
    validates=NumpyRealSymmetricMatrix):


    def validate(
            self,
            data: NumpyRealSymmetricMatrix
    ):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatSymmetricMatrix`
                A floating point symmetric matrix, hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions, equal sized axes,
                    and symmetric constraint ``data[i,j] == data[j,i]`` fulfilled.

            Raises
            ------
            TypeCheckError
                If ``data`` does not meet typing requirements.

            ValueError
                If ``data`` does not meet semantic requirements.

            Notes
            -----
            Subclasses should override this method with narrower jaxtyping annotations
            and/or additional semantic checks, calling ``super().validate_io(data)``
            if any base contract requirements must also be met.
        """

        # Validate the data are a square matrix.
        SquareMatrixValidator().validate(data)

        # Validate that the matrix is Hermitian.
        if not np.allclose(data, data.conj().T):
            raise ValueError("Array must be an Hermitian matrix.")

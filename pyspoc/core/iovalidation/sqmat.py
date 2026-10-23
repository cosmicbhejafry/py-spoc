from pyspoc.core.types import NumpyRealSquareMatrix
from ._base import Validator
from .mat import MatrixValidator



class SquareMatrixValidator(
    Validator[NumpyRealSquareMatrix],
    validates=NumpyRealSquareMatrix):

    def validate(
            self,
            data: NumpyRealSquareMatrix
    ):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatSquareMatrix`
                A floating point square matrix, hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions and equal sized axes.

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
        
        # Check the matrix is square using typeguard and jaxtyping.
        super().validate(data)
        
        # Check the array is not just an embedded scalar (1x1 matrix).
        MatrixValidator().validate(data)
        

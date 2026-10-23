from pyspoc.core.types import NumpyRealMatrix
from ._base import Validator



class MatrixValidator(
    Validator[NumpyRealMatrix],
    validates=NumpyRealMatrix):

    
    def validate(
            self,
            data: NumpyRealMatrix
    ):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatMatrix`
                A floating point matrix, hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 2 dimensions.

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

        # Validate the typing using typeguard and jaxtyping.
        super().validate(data)

        # Verify is a matrix rather than an embedded scalar.
        if data.shape[0] <= 1:
            raise ValueError(
                f"Array must contain at least 2 rows and columns, but got shape {data.shape}."
            )

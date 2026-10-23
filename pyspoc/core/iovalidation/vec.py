from pyspoc.core.types import NumpyRealVector
from ._base import Validator



class VectorValidator(
    Validator[NumpyRealVector],
    validates=NumpyRealVector):

    
    def validate(
            self,
            data: NumpyRealVector
    ):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatVector`
                A floating point vector, hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 1 dimension.

            Raises
            ------
            TypeCheckError
                If ``data`` does not meet typing requirements.

            ValueError
                If ``data`` does not meet semantic requirements.

            Notes
            -----
            Subclasses can override this method with narrower jaxtyping annotations
            and/or additional semantic checks, calling ``super().validate(data)``
            if any base contract requirements must also be met.
        """
        super().validate(data)

        # Verify is a vector rather than an embedded scalar.
        if data.shape[0] <= 1:
            raise ValueError(
                f"Array must contain at least 2 rows, but got shape {data.shape}."
            )

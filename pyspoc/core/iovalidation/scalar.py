from pyspoc.core.types import NumpyRealScalar
from ._base import Validator



class ScalarValidator(
    Validator[NumpyRealScalar],
    validates=NumpyRealScalar):

    
    def validate(
            self,
            data: NumpyRealScalar
    ):
        """Validate a candidate :class:`Component` input or output data object
        
            Parameters
            ----------
            data : :class:`NumpyFloatScalar`
                A floating point scalar, hence the following result is accepted:
                - an :class:`np.ndarray` of float dtype with 0 dimensions.
                - an :np.floating` type.
                - a `float` type.

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

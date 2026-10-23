import numpy as np

from abc import ABC
from typeguard import check_type
from jaxtyping import Float

from ._base import Reducer


class MatrixOrVectorToScalarReducer(
    Reducer[
        Float[np.ndarray, "n p"] | Float[np.ndarray, "n"],
        float | np.floating
    ],
    ABC):


    
    def _reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]] |
                  np.ndarray[tuple[int], np.dtype[np.floating]])  -> float | np.floating:
        print("lol")
        return 0


    def _get_validated_result(
            self,
            result: float | np.floating
    ) -> float | np.floating:

        """Validate and return a value produced by :meth:`_reduce`.

        Parameters
        ----------
        result : `float` or :class:`np.floating`
            Candidate scalar reducer result. It must be a scalar result
            of floating point type, hence the following results are accepted:
            - a `float` scalar.
            - an :class:`np.floating` scalar.

        Returns
        -------
        `float` or :class:`np.floating`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a scalar floating point type.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`_reduce` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """

        check_type(result, float | np.floating)

        return result


test = MatrixOrVectorToScalarReducer(short_name="test", labels=[])
test._reduce()
from abc import ABC, abstractmethod
from typeguard import check_type

from .base import MatrixStatistic
from pyspoc._core.types import NumpyDataMatrix, NumpyRealSquareMatrix


class SquareMatrixStatistic(MatrixStatistic, ABC):

    def _get_validated_result(
        self,
        result: NumpyRealSquareMatrix) -> NumpyRealSquareMatrix:
        
        """Validate and return a square matrix statistic result.

        Parameters
        ----------
        result : :class:`NumpyRealSquareMatrix`
            Candidate statistic result. It must be a two-dimensional,
            real-valued NumPy array whose row and column counts are equal.

        Returns
        -------
        :class:`NumpyRealSquareMatrix`
            The original square matrix, unchanged, after successful
            validation.

        Raises
        ------
        TypeCheckError
            If ``result`` does not satisfy either the general matrix contract
            inherited from :class:`MatrixStatistic` or this class's square
            matrix contract.

        Notes
        -----
        This method overrides the general result-validation hook inherited
        from :class:`MatrixStatistic`. Subclasses may extend the validation
        chain with semantic requirements such as symmetry or positive
        semidefiniteness.
        """

        # Retain the general matrix validation supplied by the parent class
        # before imposing the more restrictive square-matrix contract.
        super()._get_validated_result(result)

        # Reusing the symbolic name on both axes tells jaxtyping that their
        # lengths must be equal; Typeguard executes that check at runtime.
        check_type(result, NumpyRealSquareMatrix)

        # The validation boundary does not copy or transform the result.
        return result
    

    @abstractmethod
    def compute(self, data: NumpyDataMatrix) -> NumpyRealSquareMatrix:
        pass

import numpy as np

from typing import cast
from abc import ABC, abstractmethod
from typeguard import check_type

from pyspoc.core.statistic._base import Statistic
from pyspoc.core.types import NumpyFloatSquareMatrix


class SquareMatrixStatistic(Statistic, ABC):
    
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> NumpyFloatSquareMatrix:

        result = super().summarize(data)
        return cast(NumpyFloatSquareMatrix, result)
    

    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyFloatSquareMatrix:
        pass


    def _get_validated_result(
        self,
        result: NumpyFloatSquareMatrix) -> NumpyFloatSquareMatrix:
        
        """Validate and return a square matrix statistic result.

        Parameters
        ----------
        result : :class:`NumpyFloatSquareMatrix`
            Candidate statistic result. It must be a two-dimensional,
            NumPy array of float dtype with equal row and column counts.

        Returns
        -------
        :class:`NumpyFloatSquareMatrix`
            The original square matrix, unchanged, after successful
            validation.

        Raises
        ------
        TypeCheckError
            If ``result`` does not satisfy this class's square matrix
            contract.

        Notes
        -----
        This method overrides the general result-validation hook inherited
        from :class:`Statistic`. Subclasses may extend the validation
        chain with semantic requirements such as symmetry or positive
        semidefiniteness.
        """

        # Retain the general matrix validation supplied by the parent class
        # before imposing the more restrictive square-matrix contract.
        super()._get_validated_result(result)

        # Reusing the symbolic name on both axes tells jaxtyping that their
        # lengths must be equal; Typeguard executes that check at runtime.
        check_type(result, NumpyFloatSquareMatrix)

        # Verify is a matrix rather than an embedded scalar.
        if result.shape[0] <= 1:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}' must return array with "
                f"at least 2 rows and columns, but got shape {result.shape}."
            )

        # The validation boundary does not copy or transform the result.
        return result


    @classmethod
    def _get_component_type(cls) -> type:
        return SquareMatrixStatistic

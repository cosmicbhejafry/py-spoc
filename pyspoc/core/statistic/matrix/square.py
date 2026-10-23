import numpy as np

from abc import ABC, abstractmethod

from pyspoc.core.statistic._base import Statistic
from pyspoc.core.types import NumpyRealSquareMatrix
from pyspoc.core.iovalidation import SquareMatrixValidator



class SquareMatrixStatistic(
    Statistic[NumpyRealSquareMatrix],
    SquareMatrixValidator,
    ABC):
    
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> NumpyRealSquareMatrix:

        return super().summarize(data)
    

    @abstractmethod
    def _summarize(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> NumpyRealSquareMatrix:
        pass


    def _validate_output(
        self,
        result: NumpyRealSquareMatrix):
        
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
        self.validate(result)


    @classmethod
    def _get_component_type(cls) -> type:
        return SquareMatrixStatistic

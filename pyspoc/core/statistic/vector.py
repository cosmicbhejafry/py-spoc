import numpy as np

from typing import final
from abc import ABC, abstractmethod

from ._base import Statistic
from pyspoc.core.iovalidation import VectorValidator



class VectorStatistic(
    Statistic[np.ndarray[tuple[int], np.dtype[np.floating]]],
    VectorValidator,
    ABC):
    
    @final
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        return super().summarize(data)
    

    @abstractmethod
    def _summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        pass


    def validate_output(
            self,
            result: np.ndarray[tuple[int], np.dtype[np.floating]]
    ):
        """Validate and return a value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : :class:`NumpyFloatVector`
            Candidate statistic result. It must be a Numpy array with float
            dtype of one dimension.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a one-dimensional float Numpy array.

        Notes
        -----
        Subclasses can override with narrower jaxtyping annotations and
        additional semantic checks, calling ``super()._validate_output(result)``
        to retain the base contract.
        """
        # Typeguard delegates the array-specific part of this annotation to
        # jaxtyping. Both symbolic axes occur in this single check, so no
        # shared argument/return dimension context is required here.
        self.validate(result)
    

    @classmethod
    def _get_component_type(cls) -> type:
        return VectorStatistic

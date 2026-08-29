import numpy as np

from typing import cast, final
from abc import ABC, abstractmethod
from typeguard import check_type

from ._base import Statistic


class VectorStatistic(Statistic, ABC):
    
    @final
    def summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        result = super().summarize(data)
        return cast(np.ndarray[tuple[int], np.dtype[np.floating]], result)
    

    @abstractmethod
    def _summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        pass


    def _get_validated_result(
            self,
            result: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:

        """Validate and return a value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : :class:`NumpyFloatVector`
            Candidate statistic result. It must be a Numpy array with float
            dtype of one dimension.

        Returns
        -------
        :class:`NumpyFloatVector`
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a one-dimensional, real-valued Numpy array.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`_summarize` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """
        # Typeguard delegates the array-specific part of this annotation to
        # jaxtyping. Both symbolic axes occur in this single check, so no
        # shared argument/return dimension context is required here.
        check_type(result, np.ndarray[tuple[int], np.dtype[np.floating]])

        # Typeguard validates the ndarray and dtype but does not enforce the
        # native ndarray shape tuple. Check rank before indexing ``shape`` so
        # zero-dimensional results produce a meaningful contract error.
        if result.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}' must return "
                f"one-dimensional, but got shape {result.shape}."
            )

        if result.shape[0] == 0:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}' must return "
                "at least one element.")

        # Validation must not copy or otherwise transform statistic results.
        return result
    

    @classmethod
    def _get_component_type(cls) -> type:
        return VectorStatistic

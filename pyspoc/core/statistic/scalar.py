import numpy as np

from abc import ABC, abstractmethod
from typeguard import check_type

from ._base import Statistic


class ScalarStatistic(Statistic, ABC):

    @abstractmethod
    def _summarize(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]) -> float:
        pass


    def _get_validated_result(
            self,
            result: float | np.floating) -> float | np.floating:

        """Validate and return a value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : float or :class:`np.floating`
            Candidate statistic result. It must be float typed value.

        Returns
        -------
        float
            The original result, unchanged, after successful validation.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a real-valued numeric value.

        Notes
        -----
        This method is deliberately an identity operation. Its purpose is to
        provide a stable runtime-validation boundary after a concrete
        :meth:`compute` implementation returns. Subclasses can override it
        with narrower jaxtyping annotations and additional semantic checks,
        calling ``super()._get_validated_result(result)`` to retain the base
        contract.
        """
        # Typeguard delegates the array-specific part of this annotation to
        # jaxtyping. Both symbolic axes occur in this single check, so no
        # shared argument/return dimension context is required here.
        check_type(result, float)

        # Validation must not copy or otherwise transform statistic results.
        return result


    @classmethod
    def _get_component_type(cls) -> type:
        return ScalarStatistic

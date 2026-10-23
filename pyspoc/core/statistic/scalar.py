import numpy as np

from abc import ABC

from ._base import Statistic
from pyspoc.core.types import NumpyRealScalar
from pyspoc.core.iovalidation import ScalarValidator


class ScalarStatistic(
    Statistic[np.ndarray[tuple[()], np.dtype[np.floating]] |
              np.floating |
              float],
    ScalarValidator,
    ABC):


    def _validate_output(
            self,
            result: NumpyRealScalar):

        """Validate and return a value produced by :meth:`_summarize`.

        Parameters
        ----------
        result : :class:`NumpyFloatScalar`
            Candidate statistic result. It must be a float scalar or
            embedded value, hence the following results are accepted:
            - a `float` scalar.
            - an :class:`np.floating` scalar.
            - a `float` embedded scalar: a zero-dimensional Numpy array of
                :class:`np.floating` dtype.

        Raises
        ------
        TypeCheckError
            If ``result`` is not a floating point value.

        Notes
        -----
        Subclasses can override with narrower jaxtyping annotations and
        additional semantic checks, calling ``super()._validate_output(result)``
        to retain the base contract.
        """
        self.validate(result)


    @classmethod
    def _get_component_type(cls) -> type:
        return ScalarStatistic

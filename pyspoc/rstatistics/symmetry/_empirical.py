import numpy as np

from typing import Literal, Callable, Concatenate, Iterator, Sequence, TypeAlias

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.settings import settings
from pyspoc._tag import Tag, Type, Signature
from pyspoc._execution.generic import execute_async


IsotropyOperation: TypeAlias = Callable[
    Concatenate[np.ndarray, ...],
    np.ndarray | float,
]

class EmpiricalIsotropy(ReducedStatistic):

    _long_name = "Empirical Isotropy"
    _identifier = "emp-iso"
    _tags = (Tag.DISTRIBUTION, Tag.EMPIRICAL)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.SYMMETRY,)
    
    def __init__(
            self,
            n_operations: int = 1000,
             
             # Handled automatically by RandomSeedMixin
            random_seed: int = settings.current.random_seed,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe"):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

        self._n_operations = n_operations
        self._data_var = None
        self._tot_sq_var = None

    def _compute_isotropies(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            arguments_collection: (
                Iterator[dict[str, np.floating | tuple[int, int]]] |
                Sequence[dict[str, np.floating | tuple[int, int]]] |
                None),
            isotropy_operation: IsotropyOperation
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]] | None:
        
        # Return blank result if zero variation found in data.
        if np.all(data == data[0, :]):
            return None

        # Scale the data to handle extremely small or large scaling,
        # but without affecting isotropy score.
        scaled_data = data / np.max(np.abs(data))
        scaled_data = scaled_data - scaled_data.mean(axis=0)
        scaled_data = scaled_data / np.max(np.abs(scaled_data))

        # Get the variance of the new data.
        data_var, tot_sq_var = self._get_data_variance(scaled_data)
        data_var.flags.writeable = False
        self._data_var = data_var
        self._tot_sq_var = tot_sq_var
        
        result, exceptions = execute_async(
            data=scaled_data,
            operation=isotropy_operation,
            inputs=arguments_collection,
            n_operations=self._n_operations)

        if exceptions:
            raise ExceptionGroup(
                "A number of errors were encountered during asynchronous execution "
                f"of {isotropy_operation.__name__}.",
                exceptions)

        return result


    def _get_isotropy_score(
            self,
            transformed_data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> float:
        data_var = self._data_var
        transformed_var = np.var(transformed_data, axis=0)
        transformed_sse = np.square(data_var - transformed_var).sum()
        
        # Return coefficient of determination normalised result.
        return 1 - transformed_sse / self._tot_sq_var


    @staticmethod
    def _get_data_variance(
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], float]:
        data_var = np.var(data, axis=0)
        tot_sq_var = np.square(data_var).sum()
        return data_var, tot_sq_var

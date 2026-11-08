import numpy as np

from typing import Literal

from pyspoc.settings import settings
from ._empirical import EmpiricalIsotropy



class HouseholderEmpiricalIsotropy(EmpiricalIsotropy):

    def __init__(
            self,
            n_reflections: int = 1000,

            # Handled automatically by RandomSeedMixin
            random_seed: int = settings.current.random_seed,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            n_operations=n_reflections,
            random_seed=random_seed,
            internal_parallelism="parallel",
            thread_safety="safe")

    
    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.floating | float:
        
        seed_sequence = np.random.SeedSequence(self.random_seed)
        seeds = [
            {"seed": child_seed}
            for child_seed in seed_sequence.spawn(self._n_operations)
        ]
        reflection_func = self._householder_random_reflection_var_mse
        reflection_isotropies = self._compute_isotropies(
            data,
            arguments_collection=seeds,
            isotropy_operation=reflection_func)

        if reflection_isotropies is None: # Means we have zero variance data.
            return np.nan
        
        return reflection_isotropies.mean()

    
    def _householder_random_reflection_var_mse(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            seed: np.random.SeedSequence
    ) -> float:
        rng = np.random.default_rng(seed)
        p = data.shape[1]
        v = rng.standard_normal(size=p)
        w = v / np.sqrt(np.square(v).sum())
        H = np.eye(p) - 2*np.outer(w, w)
        data_reflect = np.matmul(data, H)
        return self._get_isotropy_score(data_reflect)

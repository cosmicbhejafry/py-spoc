import numpy as np

from typing import Iterator, Literal

from pyspoc.rstatistics.base import ReducedStatistic
from pyspoc.reducers import EigenValues
from pyspoc.settings import settings
from pyspoc._tag import Tag, Type, Signature
from ._empirical import EmpiricalIsotropy



class EigenIsotropy(ReducedStatistic):

    _long_name = "Eigen Isotropy"
    _identifier = "eig-iso"
    _tags = (Tag.DISTRIBUTION,)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.SYMMETRY,)

    def __init__(
            self,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            dim=dim,
            nan_policy=nan_policy)

    def _compute(self, data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.floating:
        p = data.shape[1]
        cov = np.cov(data.T)
        eigenvals = EigenValues(num_values=p).reduce(cov)
        geo_avg = np.prod(eigenvals) ** (1 / p)
        arith_avg = np.sum(eigenvals) / p
        return geo_avg / arith_avg



class GivensEmpiricalIsotropy(EmpiricalIsotropy):

    _long_name = "Givens Empirical Isotropy"
    _identifier = "emp-iso-givens"
    _tags = (Tag.DISTRIBUTION, Tag.EMPIRICAL)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.SYMMETRY,)
    
    def __init__(
            self,
            n_rotations: int = 1000,
            random_seed: int = settings.current.random_seed,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            n_operations=n_rotations,
            random_seed=random_seed,
            internal_parallelism="parallel",
            thread_safety="safe")


    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.floating | float:

        rotation_schemes = self._get_givens_rotation_schemes(
            p=data.shape[1])
        rotation_func = self._givens_rotation_var_mse
        rotation_isotropies = self._compute_isotropies(
            data,
            arguments_collection=rotation_schemes,
            isotropy_operation=rotation_func)
        
        if rotation_isotropies is None: # Means we have zero variance data.
            return np.nan
        
        return rotation_isotropies.mean()

        
    @staticmethod
    def _generate_givens_thetas(
        size: int,
        rng: np.random.Generator
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return rng.uniform(size=size) * 2 * np.pi


    @staticmethod
    def _generate_givens_planes(
            p: int,
            size: int,
            rng: np.random.Generator
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        x = rng.integers(p, size=size)
        y = rng.integers(p - 1, size=size)

        y += y >= x # Increment the forbidden value at each position.
        return np.column_stack((x,y))
    

    def _givens_rotation_var_mse(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            theta: float | np.floating,
            plane_axes: tuple[int, int]
    ) -> float:
        p = data.shape[1]
        G = np.eye(p)
        i,j = plane_axes

        if i == j:
            raise ValueError(f"Plane axes must be distinct, but got {i} and {j}")

        c = np.cos(theta)
        s = np.sin(theta)
        G[i,i] = G[j,j] = c
        G[i,j] = s
        G[j,i] = -s
        data_rot = np.matmul(data, G)
        return self._get_isotropy_score(data_rot)

    
    def _get_givens_rotation_schemes(
            self,
            p: int
        ) -> Iterator[
                dict[str, np.floating | tuple[int, int]]]:
        
        n_rotations = self._n_operations
        rng = self.make_numpy_generator(self.random_seed)
        thetas = self._generate_givens_thetas(size=n_rotations, rng=rng)
        planes = self._generate_givens_planes(p=p, size=n_rotations, rng=rng)
        schemes = zip(thetas, planes)

        for theta, plane in schemes:
            plane_x = plane[0]
            plane_y = plane[1]
            yield {
                "theta": theta,
                "plane_axes": (plane_x, plane_y)
            }


class HarrEmpiricalIsotropy(EmpiricalIsotropy):

    _long_name = "Harr Empirical Isotropy"
    _identifier = "emp-iso-harr"
    _tags = (Tag.DISTRIBUTION, Tag.EMPIRICAL)
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.SYMMETRY,)
    
    def __init__(
            self,
            n_rotations: int = 1000,
            random_seed: int = settings.current.random_seed,
            dim: Literal["n", "p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            n_operations=n_rotations,
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
        rotation_func = self._harr_random_rotation_var_mse
        rotation_isotropies = self._compute_isotropies(
            data,
            arguments_collection=seeds,
            isotropy_operation=rotation_func)

        if rotation_isotropies is None: # Means we have zero variance data.
            return np.nan
        
        return rotation_isotropies.mean()
        

    def _harr_random_rotation_var_mse(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            seed: np.random.SeedSequence
    )  -> float:
        """
        Generates a uniformly random p x p rotation matrix (Haar distributed).
        """
        rng = np.random.default_rng(seed)
        p = data.shape[1]
        # 1. Fill a matrix with standard normal random variables
        H = rng.standard_normal((p, p))
        
        # 2. Compute QR decomposition
        Q, R = np.linalg.qr(H)
        
        # 3. Correct for sign bias to ensure a proper rotation (determinant = +1)
        # instead of a reflection (determinant = -1)
        d = np.diag(R)
        ph = d / np.abs(d)
        Q = Q * ph
        
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
                    
        data_rot = np.matmul(data, Q)
        return self._get_isotropy_score(data_rot)

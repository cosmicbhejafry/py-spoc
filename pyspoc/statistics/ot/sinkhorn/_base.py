import numpy as np

from typing import Any, Generic, Callable, Literal, TypeAlias, TypeVar, cast, get_args, get_origin
from types import get_original_bases
from abc import abstractmethod
from functools import partial

import pyspoc._argchecking as argchk

from pyspoc.settings import settings
from pyspoc.statistics.base import PairwiseStatistic
from pyspoc._utils.lib import OPTIONAL_MODULES
from pyspoc._exceptions import OptionalDependencyMissingError, OptionalDependencyImportError
from pyspoc._estimators import LazyFittedPairwiseCachedEstimatorMixin


try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    from pyspoc._utils.lib import TORCH_URL

    if e.name not in OPTIONAL_MODULES:
        raise e
    
    if isinstance(e, ModuleNotFoundError):
        details = None
        install_hint = None
        
        if e.name == "torch":
            details = (f"We recommend installing torch directly rather "
                "than through pySPoC, which will allow for hardware-specific installation. "
                f"Installation instructions can be found here: {TORCH_URL}.")
        else: # Geomloss
            install_hint = "extended"

        raise OptionalDependencyMissingError(
            dependency=e.name,
            feature="ot.loss",
            install_hint=install_hint,
            details=details) from e

    raise OptionalDependencyImportError(
        dependency=e.name,
        feature="ot.loss") from e



SinkhornLoss: TypeAlias = Literal[
    "euclidean",
    "sqeuclidean",
    "energy",
    "gaussian",
    "laplacian_l1",
    "laplacian_l2"]


T_OTResult = TypeVar("T_OTResult")

###########################
# Base Abstract Estimator #
###########################

class PairwiseSinkhornEstimator(
    LazyFittedPairwiseCachedEstimatorMixin[T_OTResult]):

    CostFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
            self,
            loss: SinkhornLoss,
            epsilon: float = 0.001,
            sigma: float | None = 0.05, # Only used for rbf/gaussian and laplacian.
            max_iter: int = 1000,
            normalisation: Literal["data", "reg", None] = "data"):
                                
        self._loss_options = get_args(SinkhornLoss)
        self._loss = loss
        self._epsilon = argchk.check_float(epsilon, exclusive_minimum=0, arg_name="epsilon")
        self._sigma = (argchk.check_float(sigma, exclusive_minimum=0, arg_name="sigma")
            if loss in ("gaussian", "laplacian_l1", "laplacian_l2")
            else None)
        self._max_iter = argchk.check_natural_number(max_iter, arg_name="max_iter")
        self._normalisation = normalisation


    @abstractmethod
    def _fit_estimator(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> T_OTResult:
        pass


    def _get_cost_func(self) -> CostFunction:
        match self._loss:

            case "euclidean":
                return torch.cdist

            case "sqeuclidean":
                return self._sqeuclidean_cost

            case "energy":
                return self._energy_cost

            case "gaussian":
                return partial(self._gaussian_cost, sigma=self._sigma)

            case "laplacian_l1":
                return partial(self._laplacian_cost, p=1, sigma=self._sigma)

            case "laplacian_l2":
                return partial(self._laplacian_cost, p=2, sigma=self._sigma)

            case _:
                raise ValueError(
                    "Invalid loss function specified. Valid options are: {options}".format(
                        options="\t\n".join(self._loss_options)))


    @staticmethod
    def _sqeuclidean_cost(
            x_2d: torch.Tensor,
            y_2d: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * torch.cdist(x_2d,y_2d,p=2).square()
    

    @staticmethod
    def _energy_cost(
            x_2d: torch.Tensor,
            y_2d: torch.Tensor
    ) -> torch.Tensor:
        K_xy = -torch.cdist(x_2d,y_2d,p=2)
        K_xx = -torch.diagonal(torch.cdist(x_2d, x_2d))[:, None]
        K_yy = -torch.diagonal(torch.cdist(y_2d, y_2d))[None, :]
        return K_xx + K_yy - 2 * K_xy
    

    @staticmethod
    def _gaussian_cost(
            x_2d: torch.Tensor,
            y_2d: torch.Tensor,
            *,
            sigma: float
    ) -> torch.Tensor:
        return 2 * (1 - torch.exp(-torch.cdist(x_2d,y_2d,p=2).square() / (2 * sigma ** 2)))
    

    @staticmethod
    def _laplacian_cost(
            x_2d: torch.Tensor,
            y_2d: torch.Tensor,
            *,
            p: int,
            sigma: float
    ) -> torch.Tensor:
        return 2 * (1 - torch.exp(-torch.cdist(x_2d,y_2d,p=p) / sigma))
        
    
    def _normalise_reg(
            self,
            epsilon: float,
            C_xy: torch.Tensor,
            C_xx: torch.Tensor | None = None,
            C_yy: torch.Tensor | None = None
    ) -> float:

        cost_matrices = [C_xy]

        if C_xx is not None:
            cost_matrices.append(C_xx)

        if C_yy is not None:
            cost_matrices.append(C_yy)
        
        tot_max = max((C.max() for C in cost_matrices))
        tot_min = min((C.min() for C in cost_matrices))
        tot_span = (tot_max - tot_min).item()

        if tot_span == 0:
            return epsilon
        
        return epsilon * tot_span


TEstimator = TypeVar("TEstimator", bound=PairwiseSinkhornEstimator)

###########################
# Base Abstract Statistic #
###########################

class PairwiseSinkhornStatistic(
    Generic[TEstimator],
    PairwiseStatistic):

    def __init__(
            self,
            loss: SinkhornLoss,
            epsilon: float = 0.001,
            sigma: float | None = 0.05, # Only used for rbf/gaussian and laplacian.
            max_iter: int = 1000,
            normalisation: Literal["data", "reg", None] = "data",
            extra_kwargs: dict[str, Any] | None = None,
            dim: Literal["n","p"] = "p",
            nan_policy: Literal[
                "raise","row-omit","col-omit"] = settings.current.data_nan_policy):
                                
        self._loss = loss
        self._epsilon = epsilon
        self._sigma = sigma
        self._max_iter = max_iter
        self._normalisation = normalisation
        self._extra_kwargs = extra_kwargs or {}
        self._estimator_type = self._get_estimator_type()
        self._estimator_: TEstimator | None = None
        
        super().__init__(
            dim=dim,
            nan_policy=nan_policy)


    @abstractmethod
    def _pairwise_compute_indexed(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]],
            index: tuple[int, int]
    ) -> np.floating:
        pass


    @classmethod
    def _get_estimator_type(cls) -> type[TEstimator]:
        """Resolve the estimator specialization through intermediate bases.

        Returns
        -------
        type[TEstimator]
            Concrete Sinkhorn estimator supplied to the generic statistic base.

        Raises
        ------
        TypeError
            If no concrete Sinkhorn estimator specialization can be resolved.
        """
        def resolve(
            base: Any, substitutions: dict[TypeVar, Any],
        ) -> type[TEstimator] | None:
            origin = get_origin(base)
            current = base if origin is None else origin

            # Ordinary bases have no typing origin. Check that the resolved
            # object is a class before calling issubclass, and ignore unrelated
            # mixins rather than mistaking their generic arguments for ours.
            if not isinstance(current, type) or not issubclass(
                current, PairwiseSinkhornStatistic
            ):
                return None

            # Carry concrete bindings through generic intermediate classes,
            # e.g. Intermediate[E] -> PairwiseSinkhornStatistic[E].
            bindings = substitutions.copy()
            arguments = tuple(substitutions.get(arg, arg) for arg in get_args(base))
            bindings.update(zip(getattr(current, "__parameters__", ()), arguments))

            if current is PairwiseSinkhornStatistic:
                estimator = bindings.get(TEstimator)
                if isinstance(estimator, type) and issubclass(
                    estimator, PairwiseSinkhornEstimator
                ):
                    return cast(type[TEstimator], estimator)
                return None

            # __orig_bases__ can be inherited. Only use original-base metadata
            # owned by this class; otherwise traverse its actual direct parents
            # so plain subclasses do not skip intermediate inheritance layers.
            parents = (
                get_original_bases(current)
                if "__orig_bases__" in vars(current)
                else current.__bases__
            )
            for parent in parents:
                estimator = resolve(parent, bindings)
                if estimator is not None:
                    return estimator
            return None

        estimator = resolve(cls, {})
        if estimator is None:
            raise TypeError(
                f"No concrete Sinkhorn estimator supplied for {cls.__qualname__} "
                "through PairwiseSinkhornStatistic."
            )
        return estimator

    
    def _get_fitted_estimator(self, data: np.ndarray) -> TEstimator:
        """Resolve, fit, and retain the family-specific clustering estimator.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset used both as the estimator cache identity and training
            input.

        Returns
        -------
        TEstimator
            Fitted cached estimator. The returned object may be shared by
            multiple Statistics and should therefore be treated as read-only.
        """
        
        estimator = self._estimator_type.get_or_create(
            data=data,
            loss=self._loss,
            epsilon=self._epsilon,
            sigma=self._sigma,
            max_iter=self._max_iter,
            normalisation=self._normalisation,
            **self._extra_kwargs)

        estimator.fit(data)
        return estimator


    def _compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        
        if self._normalisation == "data":
            data = self._normalise_data(data)

        self._estimator_ = self._get_fitted_estimator(data)

        return super()._compute(data)


    def _normalise_data(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        range = (data.max() - data.min())

        if range == 0:
            return data - data.min()
        
        return (data - data.min()) / range

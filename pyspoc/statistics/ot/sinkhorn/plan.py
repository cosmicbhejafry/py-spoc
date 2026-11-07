import numpy as np

from scipy.special import xlogy
from typing import Literal
from dataclasses import dataclass

from pyspoc.settings import settings
from pyspoc.types import SummaryType
from pyspoc._utils.lib import OPTIONAL_MODULES
from pyspoc._utils.numerical import summarize
from pyspoc._exceptions import OptionalDependencyMissingError, OptionalDependencyImportError
from ._base import PairwiseSinkhornStatistic, PairwiseSinkhornEstimator, SinkhornLoss


try:
    import torch
    from geomloss.ot import solve, OTResultMatrix
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

####################
# Estimator Result #
####################

@dataclass(frozen=True, kw_only=True)
class OTResultPlan:
    xy: OTResultMatrix

    def get_plan(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        # NOTE: plan attribute DOES exist, error caused by invalid Geomloss stub.
        return self.xy.plan.detach().cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue]

#############
# Estimator #
#############

class PairwiseSinkhornPlanEstimator(
    PairwiseSinkhornEstimator[OTResultPlan]):


    def _fit_estimator(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> OTResultPlan:
        """Fit the concrete estimator."""

        x_torch = torch.tensor(x).reshape(-1,1)
        y_torch = torch.tensor(y).reshape(-1,1)
        cost_func = self._get_cost_func()
        C_xy = cost_func(x_torch, y_torch)
        reg = (self._epsilon
            if self._normalisation != "reg"
            else self._normalise_reg(
                self._epsilon,
                C_xy=C_xy))

        xy = solve(C_xy, reg=reg, max_iter=self._max_iter)
        return OTResultPlan(xy=xy)
    
    
    def _get_result(
            self,
            index: tuple[int, int]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        model = self._get_model(index)
        return model.get_plan() # pyright: ignore[reportAttributeAccessIssue]

###########################
# Abstract Plan Statistic #
###########################

class PairwiseSinkhornPlanStatistic(
    PairwiseSinkhornStatistic[PairwiseSinkhornPlanEstimator]):

    def __init__(
            self,
            loss: SinkhornLoss,
            epsilon: float | None = 0.001,
            sigma: float | None = 0.05, # Only used for rbf/gaussian and laplacian.
            max_iter: int = 1000,
            normalisation: Literal["data", "reg", None] = "data",
            dim: Literal["n","p"] = "p",
            nan_policy: Literal[
                "raise","row-omit","col-omit"] = settings.current.data_nan_policy):

        super().__init__(
            loss=loss,
            epsilon=epsilon,
            sigma=sigma,
            max_iter=max_iter,
            normalisation=normalisation,
            dim=dim,
            nan_policy=nan_policy)


    def _get_plan(
            self,
            index: tuple[int, int]):
        
        if self._estimator_ is None:
            raise RuntimeError("Estimator not fitted. Call _compute() first.")

        ot_plan = self._estimator_._get_model(index)
        return ot_plan.get_plan()

#######################
# Concrete Statistics #
#######################

class PairwiseSinkhornPlanEntropy(
    PairwiseSinkhornPlanStatistic):


    def __init__(
            self,
            loss: SinkhornLoss,
            entropy_type: Literal[
                "allocation",
                "mass-weighted"],
            epsilon: float = 0.001,
            sigma: float | None = 0.05,
            max_iter: int = 1000,
            normalisation: Literal["data", "reg", None] = "data",
            dim: Literal["n","p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit","col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            loss=loss,
            epsilon=epsilon,
            sigma=sigma,
            max_iter=max_iter,
            normalisation=normalisation,
            dim=dim,
            nan_policy=nan_policy)

        self._entropy_type = entropy_type
        

    def _pairwise_compute_indexed(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]],
            index: tuple[int, int]
    ) -> np.floating:
        
        Pi = self._get_plan(index)

        if self._entropy_type == "allocation":
            return -np.sum(xlogy(Pi, Pi))

        a = Pi.sum(axis=1)
        Q = np.divide(
            Pi, a[:, None],
            out=np.zeros_like(Pi),
            where=a[:, None] > 0)
        return -np.sum(xlogy(Pi, Q))



class PairwiseSinkhornPlanEffectiveDestinationsSummary(
    PairwiseSinkhornPlanStatistic):

    def __init__(
            self,
            loss: SinkhornLoss,
            summary_type: SummaryType,
            epsilon: float | None = 0.001,
            sigma: float | None = 0.05,
            max_iter: int = 1000,
            normalisation: Literal["data", "reg", None] = "data",
            dim: Literal["n","p"] = "p",
            nan_policy: Literal[
                "raise", "row-omit","col-omit"] = settings.current.data_nan_policy):
        
        super().__init__(
            loss=loss,
            epsilon=epsilon,
            sigma=sigma,
            max_iter=max_iter,
            normalisation=normalisation,
            dim=dim,
            nan_policy=nan_policy)

        self._summary_type = summary_type

    def _pairwise_compute_indexed(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]],
            index: tuple[int, int]
    ) -> np.floating | float:
                
        Pi = self._get_plan(index)
        a = Pi.sum(axis=1)
        Q = np.divide(
            Pi, a[:, None],
            out=np.zeros_like(Pi),
            where=a[:, None] > 0)
        effective_destinations = np.exp(-np.sum(xlogy(Q, Q), axis=1))
        return summarize(effective_destinations, self._summary_type)

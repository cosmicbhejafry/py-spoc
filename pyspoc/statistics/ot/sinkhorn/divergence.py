import numpy as np

from typing import Literal
from dataclasses import dataclass

from pyspoc.settings import settings
from pyspoc._utils.lib import OPTIONAL_MODULES
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
class OTResultDivergence:
    xx: OTResultMatrix
    xy: OTResultMatrix
    yy: OTResultMatrix


    def get_divergence(self) -> float:
        # NOTE: value attribute DOES exist, error caused by invalid Geomloss stub.
        return (
            self.xy.value.item() - 0.5 * (self.xx.value.item() + self.yy.value.item())) # pyright: ignore[reportAttributeAccessIssue]

#############
# Estimator #
#############

class PairwiseSinkhornDivergenceEstimator(
    PairwiseSinkhornEstimator[OTResultDivergence]):


    def _fit_estimator(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> OTResultDivergence:
        """Fit the concrete estimator."""

        x_torch = torch.tensor(x).reshape(-1,1)
        y_torch = torch.tensor(y).reshape(-1,1)
        cost_func = self._get_cost_func()
        C_xy = cost_func(x_torch, y_torch)
        C_xx = cost_func(x_torch, x_torch)
        C_yy = cost_func(y_torch, y_torch)

        reg = (self._epsilon
            if self._normalisation != "reg"
            else self._normalise_reg(
                self._epsilon,
                C_xy=C_xy,
                C_xx=C_xx,
                C_yy=C_yy))
    
        xy = solve(C_xy, reg=reg, max_iter=self._max_iter)
        xx = solve(C_xx, reg=reg, max_iter=self._max_iter)
        yy = solve(C_yy, reg=reg, max_iter=self._max_iter)
        return OTResultDivergence(
            xx=xx,
            xy=xy,
            yy=yy)


    def _get_result(
            self,
            index: tuple[int, int]
    ) -> float:
        model = self._get_model(index)
        return model.get_divergence() # pyright: ignore[reportAttributeAccessIssue]

#######################
# Concrete Statistics #
#######################

class PairwiseSinkhornDivergence(
    PairwiseSinkhornStatistic[PairwiseSinkhornDivergenceEstimator]):

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
        

    def _get_model(
            self,
            index: tuple[int, int]):
        
        if self._estimator_ is None:
            raise RuntimeError("Estimator not fitted. Call _compute() first.")

        return self._estimator_._get_model(index)

    
    def _pairwise_compute_indexed(
            self,
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            y: np.ndarray[tuple[int], np.dtype[np.floating]],
            index: tuple[int, int]
    ) -> float:
        ot_loss = self._get_model(index)
        return ot_loss.get_divergence()

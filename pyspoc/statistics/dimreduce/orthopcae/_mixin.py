import logging
import numpy as np

from typing import Union, Iterable, Optional, Literal, Any
from abc import ABC
from dataclasses import dataclass

from pyspoc import _argchecking
from pyspoc.settings import settings
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from ._estimator import OrthogonalPCAEEstimator


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedOrthogonalPCAEParameters:
    """Dataset-specific OrthogonalPCAE parameters."""

    components: tuple[int, ...]
    max_bottleneck_dim: int


class OrthogonalPCAEMixin(RuntimeTypeCheckedMixin, ABC):

    def __init__(
        self,
        batch_size: int,
        components: Union[int, Iterable[int]],
        train_steps: int = 10000,
        burn_in_steps_prop: float = 0.1,
        alpha: float = 0.1,
        compute_model_type: Literal["current", "optimal"] = "optimal",
        max_bottleneck_dim: Optional[int] = None,
        shuffle: bool = True):

        self._batch_size = _argchecking.check_natural_number(
            batch_size,
            "batch_size",
        )

        if isinstance(components, int):
            component_count = _argchecking.check_natural_number(
                components,
                "components",
            )
            normalized_components = tuple(range(1, component_count + 1))
        else:
            supplied_components = tuple(components)

            if not supplied_components:
                raise ValueError(
                    "components must contain at least one component."
                )

            normalized_components = tuple(
                sorted({
                    _argchecking.check_natural_number(
                        component,
                        f"components[{index}]",
                    )
                    for index, component in enumerate(supplied_components)
                })
            )

        self._components = normalized_components

        self._train_steps = _argchecking.check_natural_number(
            train_steps,
            "train_steps",
        )
        self._burn_in_steps_prop = _argchecking.clip_float(
            burn_in_steps_prop,
            lower_bound=0,
            upper_bound=0.5,
            arg_name="burn_in_steps_prop",
        )
        self._alpha = _argchecking.check_float(
            alpha,
            arg_name="alpha",
        )
        self._compute_model_type = compute_model_type
        self._max_bottleneck_dim = max_bottleneck_dim
        self._shuffle = shuffle
        self._estimator_ = None
        super().__init__()


    def _compute_estimator_output(
            self,
            data: np.ndarray,
            resolved_parameters: ResolvedOrthogonalPCAEParameters) -> dict[str, Any]:
        
        self._estimator_ = OrthogonalPCAEEstimator.get_or_create(
            data=data,
            batch_size=self._batch_size,
            max_bottleneck_dim=resolved_parameters.max_bottleneck_dim,
            train_steps=self._train_steps,
            burn_in_steps_prop=self._burn_in_steps_prop,
            alpha=self._alpha,
            compute_model_type=self._compute_model_type,
            shuffle=self._shuffle)
        
        results = self._estimator_.compute(data)
        return results
    

    def _resolve_parameters(self, data: np.ndarray) -> ResolvedOrthogonalPCAEParameters:
        n, p = data.shape

        if self._max_bottleneck_dim is None:
            effective_max_bottleneck_dim = min(n, p)
        else:
            effective_max_bottleneck_dim = _argchecking.clip_integer(
                self._max_bottleneck_dim,
                lower_bound=1,
                arg_name="max_bottleneck_dim",
            )
            effective_max_bottleneck_dim = min(
                n,
                p,
                effective_max_bottleneck_dim,
            )

        valid_components = tuple(
            component
            for component in self._components
            if component <= effective_max_bottleneck_dim
        )

        removed_components = tuple(
            component
            for component in self._components
            if component > effective_max_bottleneck_dim
        )

        if not valid_components:
            raise ValueError(
                "No requested components are available within the resolved "
                f"bottleneck dimension of {effective_max_bottleneck_dim}."
            )

        effective_components = valid_components

        if removed_components and settings.current.verbose:
            _LOGGER.info(
                "The following components were removed as they exceeded "
                "the maximum bottleneck dimension of %r: %r.",
                effective_max_bottleneck_dim,
                removed_components
            )

        return ResolvedOrthogonalPCAEParameters(
            components=effective_components,
            max_bottleneck_dim=effective_max_bottleneck_dim
        )


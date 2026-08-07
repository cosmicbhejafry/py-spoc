"""Shared parameter handling and estimator resolution for OrthogonalPCAE Statistics."""

from __future__ import annotations

import logging
import numpy as np

from typing import Union, Iterable, Optional
from abc import ABC
from dataclasses import dataclass
from math import log2

from pyspoc import _argchecking
from pyspoc.settings import settings
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._random import RandomSeedMixin
from ._estimator import OrthogonalPCAEEstimator


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedOrthogonalPCAEParameters:
    """Store OrthogonalPCAE parameters resolved for one dataset.

    Parameters
    ----------
    components : tuple of int
        Validated, one-based component indices that fit within the dataset.
    max_bottleneck_dim : int
        Upper bound on the model bottleneck dimension.
    """

    components: tuple[int, ...]
    max_bottleneck_dim: int


class OrthogonalPCAEMixin(RuntimeTypeCheckedMixin, RandomSeedMixin, ABC):
    """Provide common initialization and estimator access for OrthogonalPCAE Statistics.

    The mixin validates dataset-independent configuration at construction time.
    Dataset-dependent limits are deferred to :meth:`_resolve_parameters`.

    Notes
    -----
    This class participates in cooperative multiple inheritance. Its
    initializer therefore calls ``super().__init__()`` after storing its own
    state.
    """

    _MIN_DEFAULT_BATCH_SIZE = 64

    def __init__(
        self,
        components: Union[int, Iterable[int]],
        batch_size: Optional[int] = None,
        train_steps: int = 10000,
        burn_in_steps_prop: float = 0.1,
        alpha: float = 0.1,
        max_bottleneck_dim: Optional[int] = None,
        shuffle: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize shared OrthogonalPCAE configuration.

        Parameters
        ----------
        batch_size : int
            Number of observations supplied per optimization step.
        components : int or iterable of int
            If an integer is supplied, request every component from one
            through that value. An iterable requests the specified one-based
            components; duplicates are removed and values are sorted.
        train_steps : int, default=10000
            Target number of optimizer steps.
            Epoch count is a derived attribute. Subject to rounding conditional
            on ``shuffle``, the number of epochs is approximately ``train_steps``
            multiplied by ``batch_size`` divided by the number of data observations.
        burn_in_steps_prop : float, default=0.1
            Proportion of initial training epochs during which stochastic
            bottleneck masking is disabled. Values are clipped to ``[0, 0.5]``.
        alpha : float, default=0.1
            Target ratio between orthogonality and reconstruction gradient
            magnitudes in loss function.
        max_bottleneck_dim : int or None, optional
            Upper bound on the model bottleneck dimension.. If ``None``, the smaller
            dataset dimension is used when computation begins.
        shuffle : bool, default=True
            Whether to shuffle observations in the training data loader.
        random_seed : int or None, optional
            Per-object random-seed override. If ``None``, use the active
            library setting.

        Raises
        ------
        TypeError
            If a runtime-checked argument has an incompatible type.
        ValueError
            If a numeric argument or component is outside its accepted range,
            or if ``components`` is empty.
        """

        # Validate configuration that does not depend on the eventual dataset.
        if batch_size is not None:
            self._batch_size = _argchecking.check_natural_number(
                batch_size,
                "batch_size",
            )
        else:
            self._batch_size = None

        if isinstance(components, int):
            component_count = _argchecking.check_natural_number(
                components,
                "components",
            )
            normalized_components = tuple(range(1, component_count + 1))
        else:
            supplied_components = tuple(components)

            if not supplied_components:
                raise ValueError("components must contain at least one component.")

            normalized_components = tuple(
                sorted(
                    {
                        _argchecking.check_natural_number(
                            component,
                            f"components[{index}]",
                        )
                        for index, component in enumerate(supplied_components)
                    }
                )
            )

        self._components = normalized_components

        self._train_steps = _argchecking.check_natural_number(
            train_steps,
            "train_steps",
        )
        self._burn_in_steps_prop = _argchecking.clip_float(
            burn_in_steps_prop,
            minimum=0,
            maximum=0.5,
            arg_name="burn_in_steps_prop",
        )
        self._alpha = _argchecking.check_float(
            alpha,
            arg_name="alpha",
        )
        self._max_bottleneck_dim = max_bottleneck_dim
        self._shuffle = shuffle
        self._estimator_ = None
        # Continue the cooperative initializer chain so random-seed and other
        # inherited mixins can initialize their state.
        super().__init__()

    def _compute_estimator_output(
        self, data: np.ndarray, resolved_parameters: ResolvedOrthogonalPCAEParameters
    ) -> OrthogonalPCAEEstimator:
        """Resolve, fit, and retain the estimator for a Statistic computation.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset used both as the estimator cache identity and training
            input.
        resolved_parameters : ResolvedOrthogonalPCAEParameters
            Dataset-specific component and bottleneck configuration.

        Returns
        -------
        OrthogonalPCAEEstimator
            Fitted cached estimator. The returned object may be shared by
            multiple Statistics and should therefore be treated as read-only.
        """
        # Constructor arguments and data identify a reusable cached estimator.
        self._estimator_ = OrthogonalPCAEEstimator.get_or_create(
            data=data,
            batch_size=self._batch_size,
            max_bottleneck_dim=resolved_parameters.max_bottleneck_dim,
            train_steps=self._train_steps,
            burn_in_steps_prop=self._burn_in_steps_prop,
            alpha=self._alpha,
            shuffle=self._shuffle,
            random_seed=self.random_seed,
        )

        # fit() is lazy and synchronized: an already-fitted cached estimator
        # returns without repeating the training operation.
        self._estimator_.fit(data)
        return self._estimator_

    def _resolve_parameters(self, data: np.ndarray) -> ResolvedOrthogonalPCAEParameters:
        """Resolve component limits that depend on the supplied dataset.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional dataset whose observation and feature counts
            constrain the bottleneck.

        Returns
        -------
        ResolvedOrthogonalPCAEParameters
            Valid requested components and the effective bottleneck dimension.

        Raises
        ------
        ValueError
            If every requested component exceeds the available bottleneck
            dimension.
        """
        n, p = data.shape

        # An unconstrained bottleneck cannot exceed either matrix dimension.
        if self._max_bottleneck_dim is None:
            effective_max_bottleneck_dim = min(n, p)
        else:
            effective_max_bottleneck_dim = _argchecking.clip_integer(
                self._max_bottleneck_dim,
                minimum=1,
                arg_name="max_bottleneck_dim",
            )
            effective_max_bottleneck_dim = min(
                n,
                p,
                effective_max_bottleneck_dim,
            )

        if self._batch_size is None:
            relative_batch_size = 2 ** round(log2(n / 10))
            self._batch_size = max(self._MIN_DEFAULT_BATCH_SIZE, relative_batch_size)

        valid_components = tuple(
            component for component in self._components if component <= effective_max_bottleneck_dim
        )

        removed_components = tuple(
            component for component in self._components if component > effective_max_bottleneck_dim
        )

        if not valid_components:
            raise ValueError(
                "No requested components are available within the resolved "
                f"bottleneck dimension of {effective_max_bottleneck_dim}."
            )

        effective_components = valid_components

        # Component removal is non-fatal. Report it only when the library's
        # contextual verbosity setting asks for informational diagnostics.
        if removed_components and settings.current.verbose:
            _LOGGER.info(
                "The following components were removed as they exceeded "
                "the maximum bottleneck dimension of %r: %r.",
                effective_max_bottleneck_dim,
                removed_components,
            )

        return ResolvedOrthogonalPCAEParameters(
            components=effective_components, max_bottleneck_dim=effective_max_bottleneck_dim
        )

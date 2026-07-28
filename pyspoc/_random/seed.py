"""Random-seed resolution shared by statistics and estimators."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from pyspoc.exceptions import OptionalDependencyMissingError
from pyspoc._initialization import AutoInitializedMixin
from pyspoc.settings import settings

if TYPE_CHECKING:
    import torch


class RandomSeedMixin(AutoInitializedMixin):
    """Provide configurable random-seed resolution to library components.

    The mixin stores either an explicit override or a resolved seed. Dynamic
    storage is appropriate for Statistics, which should observe the active
    settings context when computation begins. Frozen storage is appropriate
    for cached estimators, whose effective seed forms part of their identity.
    """

    _freeze_random_seed = False

    @classmethod
    def _resolve_cache_init_args(
            cls,
            init_args: dict[str, object]) -> dict[str, object]:
        """Resolve an omitted random seed before cached construction.

        Parameters
        ----------
        init_args : dict[str, object]
            Normalized constructor arguments supplied by cache machinery.

        Returns
        -------
        dict[str, object]
            Cooperatively resolved constructor arguments. A ``random_seed``
            entry is replaced by its effective integer value when present.
        """
        parent_resolver = getattr(
            super(),
            "_resolve_cache_init_args",
            None,
        )
        resolved_args = (
            parent_resolver(init_args)
            if parent_resolver is not None
            else init_args.copy()
        )

        if (
            "random_seed" in resolved_args
            and resolved_args["random_seed"] is None
        ):
            resolved_args["random_seed"] = resolve_random_seed(None)

        return resolved_args

    def _before_component_init(
            self,
            init_args: dict[str, object]) -> None:
        """Initialize random-seed state before concrete construction.

        Parameters
        ----------
        init_args : dict[str, object]
            Normalized constructor arguments. If ``random_seed`` is not
            declared, the package-wide seed remains the dynamic default.

        Returns
        -------
        None
            Random-seed state is stored on the component.
        """
        super()._before_component_init(init_args)
        random_seed = init_args.get("random_seed")

        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer or None.")

        self._random_seed_override = (
            resolve_random_seed(random_seed)
            if self._freeze_random_seed
            else random_seed
        )

    @property
    def random_seed(self) -> int:
        """Return the component's effective random seed.

        Returns
        -------
        int
            Explicit or frozen seed when present; otherwise the active
            package-wide seed.
        """
        return resolve_random_seed(self._random_seed_override)

    @staticmethod
    def make_numpy_generator(random_seed: int) -> np.random.Generator:
        """Create an independent NumPy generator.

        Parameters
        ----------
        random_seed : int
            Seed used to initialize the generator.

        Returns
        -------
        numpy.random.Generator
            Independent seeded generator.
        """
        return np.random.default_rng(random_seed)


    @staticmethod
    def make_torch_generator(
            random_seed: int,
            *,
            device: torch.device | str = "cpu") -> torch.Generator:
        """Create an independent PyTorch generator.

        Parameters
        ----------
        random_seed : int
            Seed used to initialize the generator.
        device : torch.device or str, default="cpu"
            Device on which generator state is maintained.

        Returns
        -------
        torch.Generator
            Independent seeded generator for ``device``.
        """
        try:
            import torch
        except ImportError as error:
            raise OptionalDependencyMissingError(
                "torch",
                feature="PyTorch random-number generation",
                install_hint="Install pySPoC with the 'extended' extra.",
            ) from error

        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)
        return generator




def resolve_random_seed(random_seed: int | None) -> int:
    """Return an explicit seed or the active package default.

    Parameters
    ----------
    random_seed : int or None
        Explicit seed override. ``None`` selects
        ``settings.current.random_seed``.

    Returns
    -------
    int
        Effective random seed.
    """
    if random_seed is not None:
        return random_seed

    return settings.current.random_seed

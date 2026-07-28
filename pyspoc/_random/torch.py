"""PyTorch random-generator construction."""

from __future__ import annotations

import torch


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
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    return generator

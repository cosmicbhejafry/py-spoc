"""NumPy random-generator construction."""

from __future__ import annotations

import numpy as np


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

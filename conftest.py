import pytest
import numpy as np

from functools import cache

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@cache
def _make(n=50, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))

@pytest.fixture
def data_factory():
    return _make

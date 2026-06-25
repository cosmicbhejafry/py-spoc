import pytest
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def data_factory():
    def make(n=50, dim=2, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))
    return make

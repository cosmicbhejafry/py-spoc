import numpy as np
import pytest

from itertools import product

from pyspoc.rstatistics.fractal.renyi import (
    _get_renyi_entropy,
    _get_renyi_entropy_py,
    _get_renyi_entropy_numba_fast,
    _get_renyi_entropy_numba_bc,
    _get_renyi_entropy_numba_alt
)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

q_opt = [0.0, 1.0, 2.0]
ns = [1000, 10000]
dims = [2, 10, 100]
neg_log_scales = np.linspace(-10, 10, 31)
scales = np.exp(neg_log_scales)

IMPLEMENTATIONS = [
    pytest.param(_get_renyi_entropy, id="public-dispatch"),
    pytest.param(_get_renyi_entropy_py, id="python-kernel"),
    pytest.param(_get_renyi_entropy_numba_fast, id="numba-kernel"),
]

# --------------------------------------------------
# Benchmark
# --------------------------------------------------

@pytest.mark.benchmark(group="Renyi Entropy")
@pytest.mark.parametrize("kernel", IMPLEMENTATIONS)
@pytest.mark.parametrize("q", q_opt)
@pytest.mark.parametrize("n, dim", product(ns, dims))
def test_renyi_entropy(
    benchmark,
    data_factory,
    kernel,
    q,
    n,
    dim,
):
    data = data_factory(n=n, dim=dim, seed=0)

    # Warm up Numba compilation.
    if kernel is _get_renyi_entropy_numba_fast:
        kernel(q, data, scales)

    benchmark(
        kernel,
        q,
        data,
        scales,
    )

IMPLEMENTATIONS = [
    pytest.param(_get_renyi_entropy_numba_alt, id="numba-hash"),
    pytest.param(_get_renyi_entropy_numba_bc, id="numba-orig"),
]


@pytest.mark.benchmark(group="Renyi Entropy Numba Variants")
@pytest.mark.parametrize("kernel", IMPLEMENTATIONS)
@pytest.mark.parametrize("q", q_opt)
@pytest.mark.parametrize("n, dim", product(ns, dims))
def test_renyi_entropy_numba_variants(
        benchmark,
        data_factory,
        kernel,
        q,
        n,
        dim):
    
    data = data_factory(n=n, dim=dim, seed=0)

    # Warm up Numba compilation.
    kernel(q, data, scales)

    benchmark(
        kernel,
        q,
        data,
        scales,
    )

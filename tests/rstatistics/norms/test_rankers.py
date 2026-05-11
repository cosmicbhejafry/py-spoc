import pytest
import numpy as np

from itertools import product

from pyspoc.rstatistics import EffectiveRank, StableRank

ns = [10,50,100,1000]
ps = [1,2,5,10,100]
max_ranks = [(p,q) for p,q in list(product(ps, ps)) if p >= q]
rankers = [EffectiveRank, StableRank]

# Rank should always be equal to p for identity matrices.
@pytest.mark.parametrize(
    ("ranker", "p"),
    list(product(rankers, ps))
)
def test_identities(ranker, p):
    statistic = ranker()
    data = np.eye(p)
    result = float(statistic.compute(data))
    assert round(result, 4) == p, \
        f"Result was expected to be {p}, but was actually {result}."
    
# Rank should always be between 1 and p for identity matrices with randomly scaled columns.
@pytest.mark.parametrize(
    ("ranker", "p"),
    list(product(rankers, ps))
)
def test_randomly_column_scaled_matrices(ranker, p):
    statistic = ranker()
    data = np.eye(p) * np.random.random(size=(1,p))
    result = float(statistic.compute(data))
    assert round(result, 4) <= p and round(result, 4) >= 1, \
        f"Result was expected to be between 1 and {p}, but was actually {result}."

test_params = [(r,p,m) for r,(p,m) in product(rankers, max_ranks)]

# Rank should be >=1 and <= max rank for matrices built from identities with possible degeneracy.
@pytest.mark.parametrize(
    ("ranker", "p", "max_rank"),
    test_params
)
def test_partial_identities(ranker, p, max_rank):
    statistic = ranker()
    I = np.eye(p)  # noqa: E741
    vec = I[:,:max_rank]
    data = build_degenerate_mat(vec, p, max_rank)
    result = float(statistic.compute(data))
    assert round(result, 4) <= max_rank and \
        round(result, 4) >= 1, \
        f"Result was expected to be between 1 and {max_rank}," + \
        f" but was actually {result}."

test_params = [(r,n,p,m) for r,n,(p,m) in list(product(rankers, ns, max_ranks))]

# Rank should be 1 and max_rank
@pytest.mark.parametrize(
    ("ranker","n", "p", "max_rank"),
    test_params
)
def test_random_mats(data_factory, ranker, n, p, max_rank):
    statistic = EffectiveRank()
    vec = data_factory(n=n, dim=max_rank)
    data = build_degenerate_mat(vec, p, max_rank)
    result = float(statistic.compute(data))
    assert round(result, 4) <= max_rank and \
        round(result, 4) >= 1, \
        f"Result was expected to be between 1 and {max_rank}," + \
        f" but was actually {result}."

def build_degenerate_mat(vec: np.ndarray, p: int, rank: int) -> np.ndarray:
    if rank == p:
        return vec
    else:
        l_dep = np.expand_dims(vec[:,0],1) * np.linspace(2,p-rank+1,p-rank)
        return np.hstack((vec, l_dep))

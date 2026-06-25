import numpy as np

from pyspoc import Reducer, ReducedStatistic

class EffectiveRank(Reducer, ReducedStatistic):

    r"""
    Effective matrix rank based on Shannon entropy of matrix singular value distribution
    proposed by [1]_.

    Definition
    ----------
    Let ``A`` be a complex-valued, nonzero matrix of shape ``(M, N)`` with
    singular value decomposition

    .. math::

        A = U D V,

    where ``U`` and ``V`` are unitary matrices, and ``D`` is an ``M x N``
    diagonal matrix containing the nonnegative singular values

    .. math::

        \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_Q \geq 0,

    with

    .. math::

        Q = \min \{M, N\}.

    Define the singular value distribution by

    .. math::

        p_k = \frac{\sigma_k}{\|\sigma\|_1}, \qquad k = 1, 2, \ldots, Q,

    where

    .. math::

        \|\sigma\|_1 = \sum_{k=1}^{Q} |\sigma_k|.

    Formula
    ----------

    The effective rank of ``A`` is defined as

    .. math::

        \operatorname{erank}(A) = \exp\{ H(p_1, p_2, \ldots, p_Q) \},

    where ``H`` is the Shannon entropy

    .. math::

        H(p_1, p_2, \ldots, p_Q) = - \sum_{k=1}^{Q} p_k \log p_k.

    All logarithms are natural logarithms, and the convention
    ``0 log 0 = 0`` is used.

    References
    ----------

    .. [1] Roy, O. and Vetterli, M. (2007). The effective rank: A measure of
       effective dimensionality. In *15th European Signal Processing Conference*,
       606-610.

    """

    _name = "Effective Rank"
    _identifier = "erank"
    _labels = ["scalar"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    def compute(self, data: np.ndarray) -> np.ndarray | float:
        s = np.linalg.svd(data, compute_uv=False)
        p = s / np.linalg.norm(s, ord=1) # pyright: ignore[reportCallIssue]
        log_p = np.zeros_like(p)
        np.log(p, out=log_p, where=p > 0)
        H = - np.sum(p * log_p)
        return np.exp(H)

class StableRank(Reducer, ReducedStatistic):
    
    _name = "Stable Rank"
    _identifier = "stable-rank"
    _labels = ["scalar"]

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def labels(self) -> list[str]:
        return self._labels
    
    def compute(self, data: np.ndarray) -> np.ndarray | float:
        frob = np.array(np.linalg.norm(x=data, ord="fro"))
        l2 = np.array(np.linalg.norm(data, ord=2)) # pyright: ignore[reportCallIssue]
        return (frob / l2) ** 2

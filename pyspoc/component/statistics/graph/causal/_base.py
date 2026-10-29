from __future__ import annotations

import numpy as np

from abc import abstractmethod
from typing import Literal, Any, TypeVar, overload

import pyspoc._argchecking as argchk
from .._base import GraphStatistic
from pyspoc.component.statistics.base import Statistic, PairwiseStatistic
from pyspoc._tag import Tag, Type, Signature


TStatistic = TypeVar("TStatistic", bound=Statistic)


class CausalGraphStatistic(
    GraphStatistic):
    """
    The main addition of this mixin class is the :meth:`_standardise_graph` method,
    which must align all causal graph statistics to the following DAG-compatible conventions:
    
    For an element `X[i,j]` in the graph adjacency matrix, the following encodings
    should be applied:
    
    - No edge (X_i  X_j) => `X[i,j] = X[j,i] = 0`
    - Unidirectional (X_i -> X_j) => `X[i,j] = 1` and `X[j,i] = 0` (DAG convention)
    - Undirected (X_i -- X_j) => `X[i,j] = X[j,i] = -1`
    - Bidirectional (X_i <-> X_j) => `X[i,j] = X[j,i] = 1`
    - Latent Confounding (X_i <-> X_j) => `X[i,j] = X[j,i] = 2`
    - Conflicted Direction (X_i <-> X_j) => `X[i,j] = X[j,i] = -2`
    """

    _long_name = "Causal Discovery Graph"
    _identifier = "graph-cd"
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.CAUSAL,)
    _tags = (Tag.GRAPH, Tag.SIGNED)
    _standard_graph_convention = (
        """
        For an element `X[i,j]` in the graph adjacency matrix, the following encodings
        should be applied:

        - No edge (X_i  X_j) => `X[i,j] = X[j,i] = 0`
        - Unidirectional (X_i -> X_j) => `X[i,j] = 1` and `X[j,i] = 0` (DAG convention)
        - Undirected (X_i -- X_j) => `X[i,j] = X[j,i] = -1`
        - Bidirectional (X_i <-> X_j) => `X[i,j] = X[j,i] = 1`
        - Latent Confounding (X_i <-> X_j) => `X[i,j] = X[j,i] = 2`
        - Conflicted Direction (X_i <-> X_j) => `X[i,j] = X[j,i] = -2`
        """)

    def __init__(
            self,
            standardise: bool,
            dim: Literal["n", "p"],
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"],
            internal_parallelism: Literal[
                "serial", "parallel"] = "serial",
            thread_safety: Literal[
                "safe", "unsafe"] = "unsafe",
            **pairwise_kwargs: Any):

        self._standardise = standardise
        super().__init__(
            enforce_adjacency_type=None,
            adjacency_scaling=None,
            zero_threshold=None,
            self_value=0,
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety,
            **pairwise_kwargs)
   
    
    @classmethod
    @abstractmethod
    def _standardise_adjacency(
            cls,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.number]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        pass


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """
        Compute the Causal Graph Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.


        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Adjacency of the estimated causal graph.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            dataset_nan_policy: Literal["raise", "row-omit", "col-omit"]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """
        Compute the Causal Graph Statistic for the given point cloud data matrix.

        
        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.

            
        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Adjacency of the estimated causal graph.
        """
        ...


    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            *,
            dataset_nan_policy: Literal["raise", "row-omit", "col-omit", None] = None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        graph = super().compute(
            data, dataset_nan_policy=dataset_nan_policy)
        
        if not self._standardise:
            return graph.astype(np.float64)
        
        graph = graph.astype(np.int8)
        graph = self._standardise_adjacency(graph)

        # Check the graph meets the required conventions.
        argchk.check_causal_graph_adjacency_standardisation(
            graph, self._standard_graph_convention)

        return graph.astype(np.float16)


class CausalGraphPairwiseStatistic(
    CausalGraphStatistic,
    PairwiseStatistic):


    def __init__(
            self,
            standardise: bool,
            dim: Literal["n", "p"],
            nan_policy: Literal[
                "raise", "row-omit", "col-omit"],
            is_ordered: bool,
            symmetry_type: Literal[
                "exact", "negative", "reciprocal", "triangular", None] = None,
            internal_parallelism: Literal[
                "serial", "parallel"] = "serial",
            thread_safety: Literal[
                "safe", "unsafe"] = "unsafe"):

        super().__init__(
            standardise=standardise,
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=is_ordered,
            symmetry_type=symmetry_type,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

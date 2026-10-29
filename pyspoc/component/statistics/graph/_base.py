"""Shared contracts and validation for feature-graph Statistics."""

from __future__ import annotations

import numpy as np

from typing import Literal, Any, TypeVar, overload

import pyspoc._argchecking as argchk
from pyspoc.component.statistics.base import Statistic, PairwiseStatistic
from pyspoc._tag import Tag, Type, Signature


TStatistic = TypeVar("TStatistic", bound=Statistic)



class GraphStatistic(Statistic):
    """Base class for Statistics that represent dataset features as graph nodes.

    A data matrix is interpreted as ``(n_observations, n_features)``. Concrete
    implementations return a square adjacency matrix whose row and column
    ``j`` both refer to feature ``j``.
    """

    _long_name = "Data-Constructed Graph"
    _identifier = "graph"
    _types = (Type.DESCRIPTIVE,)
    _signatures = (Signature.TOPOLOGY, Signature.DIMENSIONALITY)
    _tags = (Tag.GRAPH_WEIGHTED,)


    def __init__(
            self,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"],
            
            # Scales the adjacency matrix to these ranges.
            adjacency_scaling: Literal[
                "binary",
                "bipolar",
                None
            ] = None,
            
            # Enforces the adjacency matrix to these constraints.
            enforce_adjacency_type: Literal[
                "binary",
                "bipolar",
                "absolute",
                None] = None,
            
            # Any element of magnitude below this value is set to zero.
            zero_threshold: float | None = None,

            # Sets the diagonal elements of the matrix to this value.
            self_value: float | None = 0,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe",
            **pairwise_kwargs: Any):
        
        self._zero_threshold = (
            argchk.check_float(zero_threshold, minimum=0, arg_name="threshold")
            if zero_threshold is not None
            else None)

        self._adjacency_type = enforce_adjacency_type
        self._adjacency_scaling = adjacency_scaling
        self._self_value = self_value

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety,
            **pairwise_kwargs)

    
    def _finalize_adjacency(
            self,
            graph: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:

        """Apply common weighting, threshold and diagonal conventions."""
        # Handle adjacency scaling.
        if self._adjacency_scaling is not None:
        
            if self._adjacency_scaling == "binary":
                graph = np.abs(graph)
                scale_min = 0
            else:
                scale_min = -1

            max_weight = graph.max()
            min_weight = graph.min()

            if max_weight > 1 or min_weight < scale_min:
                graph = (graph - min_weight + scale_min) / (max_weight - min_weight)

        # Handle adjacency type and zero thresholding.
        if self._adjacency_type in ("binary", "bipolar"):
            zero_threshold = 0.5 if self._zero_threshold is None else self._zero_threshold
            graph[np.abs(graph) <= zero_threshold] = 0.0

            if self._adjacency_type == "binary":
                graph[np.abs(graph) > zero_threshold] = 1.0
            else:
                graph = np.sign(graph)

        else:
            if self._adjacency_type == "absolute":
                graph = np.abs(graph)
            
            if self._zero_threshold is not None:
                graph[graph <= self._zero_threshold] = 0.0

        # Handle self value.
        if self._self_value is None:
            return graph

        if isinstance(self._self_value, float) and np.issubdtype(graph.dtype, np.integer):
            graph = graph.astype(np.float64)

        np.fill_diagonal(graph, self._self_value)
        return graph


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """
        Compute the Graph Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.


        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Statistic estimate.
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
        Compute the Graph Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.


        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Statistic estimate.
        """
        ...


    @overload
    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """
        Compute the Graph Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Internal Parameters

        dataset_nan_policy : {"raise", "row-omit", "col-omit"}, optional
            The :class:`Dataset` level NaN policy. Only relevant when :meth:`compute` is called
            by :meth:`calculate`.

        ## Optional Parameters

        kwargs: Any
            Additional Statistic-specific keyword arguments used during computation.

        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Statistic estimate.
        """
        ...


    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        graph = super().compute(data, kwargs=kwargs)
        return self._finalize_adjacency(graph)
      


class GraphPairwiseStatistic(GraphStatistic, PairwiseStatistic):
    """Base class for PairwiseStatistics that represent dataset features as graph nodes.

    A data matrix is interpreted as ``(n_observations, n_features)``. Concrete
    implementations return a square adjacency matrix whose row and column
    ``j`` both refer to feature ``j``.
    """

    def __init__(
            self,
            dim: Literal["n", "p"],
            nan_policy: Literal["raise", "row-omit", "col-omit"],
            is_ordered: bool,
            symmetry_type: Literal["exact", "negative", "reciprocal", "triangular", None],
            ensure_adjacency_type: Literal[
                "binary",
                "binary-weighted",
                "bipolar",
                "bipolar-weighted",
                "absolute",
                None] = None,
            adjacency_scaling: Literal[
                "minmax",
                "clip",
                None
            ] = None,
            zero_threshold: float | None = None,
            self_value: float | None = 0,
            internal_parallelism: Literal["serial", "parallel"] = "serial",
            thread_safety: Literal["safe", "unsafe"] = "unsafe"):

        super().__init__(
            enforce_adjacency_type=ensure_adjacency_type,
            adjacency_scaling=adjacency_scaling,
            zero_threshold=zero_threshold,
            self_value=self_value,
            dim=dim,
            nan_policy=nan_policy,
            is_ordered=is_ordered,
            symmetry_type=symmetry_type,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety)

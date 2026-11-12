"""Shared contracts and validation for feature-graph Statistics."""

from __future__ import annotations

import numpy as np

from typing import Literal, Any, TypeVar

import pyspoc._argchecking as argchk
from pyspoc.dataset import Dataset
from pyspoc.statistics.base import Statistic, PairwiseStatistic
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
                "unit-minmax",
                "unit-absolute",
                "unit-signed",
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

        self._enforce_adjacency_type = enforce_adjacency_type
        self._adjacency_scaling = adjacency_scaling
        self._self_value = self_value

        remove_tags = []
        
        if enforce_adjacency_type in ("binary", "bipolar"):
            remove_tags.append(Tag.GRAPH_WEIGHTED)
            remove_tags.append(Tag.SIGNED)

        if enforce_adjacency_type == "absolute":
            remove_tags.append(Tag.SIGNED)

        if adjacency_scaling in ("unit-absolute", "unit-minmax"):
            remove_tags.append(Tag.SIGNED)

        self._modify_tags(
            remove_tags=remove_tags
        )

        super().__init__(
            dim=dim,
            nan_policy=nan_policy,
            internal_parallelism=internal_parallelism,
            thread_safety=thread_safety,
            **pairwise_kwargs)

    
    def _finalize_adjacency(
            self,
            graph: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:

        """Apply common weighting, threshold and diagonal conventions."""
        # Handle adjacency scaling.
        if self._adjacency_scaling in ("unit-absolute", "unit-minmax"):

            if self._adjacency_scaling == "unit-absolute":
                graph = np.abs(graph)
                
            max_weight = graph.max()
            min_weight = graph.min()

            if max_weight > 1 or min_weight < 0:
                graph = (graph - min_weight) / (max_weight - min_weight)

        elif self._adjacency_scaling == "unit-signed":
            negative_nodes = graph < 0
            max_neg_magnitude = abs(graph[negative_nodes].min())
            max_pos_magnitude = graph[~negative_nodes].max()

            if max_neg_magnitude > 1 or max_pos_magnitude > 1:
                graph[negative_nodes] /= max_neg_magnitude
                graph[~negative_nodes] /= max_pos_magnitude

        # Handle adjacency type and zero thresholding.
        if self._enforce_adjacency_type in ("binary", "bipolar"):
            zero_threshold = 0.5 if self._zero_threshold is None else self._zero_threshold
            graph[np.abs(graph) <= zero_threshold] = 0.0

            if self._enforce_adjacency_type == "binary":
                graph[np.abs(graph) > zero_threshold] = 1.0
            else:
                graph = np.sign(graph)

        else:
            if self._enforce_adjacency_type == "absolute":
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
    

    def _compute_dataset(
            self,
            dataset: Dataset,
            **kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        graph = super()._compute_dataset(dataset, kwargs=kwargs)
        return self._finalize_adjacency(graph)


    def compute(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating]],
            **kwargs: Any
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        """
        Compute the Graph Statistic for the given point cloud data matrix.


        Parameters
        -------
        data : np.ndarray[int, int]
            2-dimensional floating point cloud data matrix with observations along the rows
            and features along the columns.

        ## Optional Parameters

        kwargs: Any
            Additional Statistic-specific keyword arguments used during computation.

        Returns
        -------
        :class:`np.ndarray[tuple[int, int], np.dtype[np.floating]]`
            Statistic estimate.
        """
        
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

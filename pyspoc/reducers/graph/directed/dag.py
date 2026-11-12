import numpy as np
import networkx as nx

from abc import abstractmethod

from pyspoc._tag import Tag
from ._base import DiGraphReducer
from .._base import FilterCondition, AnyOf


class DAGReducer(DiGraphReducer):


    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[FilterCondition[Tag], ...]:
        """Graph tags that the DAGReducer is restricted to.
        """
        return (
            AnyOf(Tag.GRAPH_TREE, Tag.GRAPH_FOREST, Tag.GRAPH_ACYCLIC),
            Tag.GRAPH_DIRECTED)
    

    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ):
        G = self._get_digraph(formatted_adjacency)
        is_acyclic = nx.is_directed_acyclic_graph(G)

        # TODO: Add library setting to allow for warning about invalid inputs.
        # and raise here.
        if not is_acyclic:
            return np.nan
                
        return self._reduce_dag(formatted_adjacency=formatted_adjacency, G=G)


    @abstractmethod
    def _reduce_dag(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph):
        pass


class DAGLongestPathLength(DAGReducer):

    _long_name = "Directed Acyclic Graph Longest Path Length"
    _identifier = "dag-longest-path"

    def __init__(
            self,
            weighted_depth: bool = False
    ):
        super().__init__()
        self._weighted_depth = weighted_depth

    def _reduce_dag(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph
    ) -> int | float:

        weight = "weight" if self._weighted_depth else None
        return nx.dag_longest_path_length(G, weight=weight)


class DAGTopologicalDepth(DAGReducer):

    _long_name = "Directed Acyclic Graph Topological Depth"
    _identifier = "dag-depth"

    def _reduce_dag(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph
    ) -> int:

        gens = tuple(nx.topological_generations(G))
        return len(gens)

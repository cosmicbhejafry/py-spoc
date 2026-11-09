import numpy as np
import networkx as nx

from abc import abstractmethod

from pyspoc._tag import Tag
from ._base import DirectedGraphReducer


class DAGReducer(DirectedGraphReducer):


    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Graph tags that the DAGReducer is restricted to.
        """
        return (Tag.GRAPH_TREE,
                Tag.GRAPH_ACYCLIC)
    

    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ):
        
        G = self._get_digraph(adjacency)
        is_acyclic = nx.is_directed_acyclic_graph(G)

        # TODO: Add library setting to allow for warning about invalid inputs.
        # and raise here.
        if not is_acyclic:
            return np.nan
                
        return self._reduce_dag(adjacency=adjacency, G=G)


    @abstractmethod
    def _reduce_dag(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph):
        pass


class DAGLongestPathLength(DAGReducer):

    def _reduce_dag(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph
    ) -> int:
        
        return nx.dag_longest_path_length(G)


class DAGTopologicalDepth(DAGReducer):

    def _reduce_dag(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            G: nx.DiGraph
    ) -> int:

        gens = tuple(nx.topological_generations(G))
        return len(gens)

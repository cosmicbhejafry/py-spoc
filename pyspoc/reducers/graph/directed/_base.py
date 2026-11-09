import numpy as np
import networkx as nx

from pyspoc._tag import Tag
from .._base import GraphReducer


class DirectedGraphReducer(GraphReducer):


    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Graph tags that the GraphReducer is restricted to.
        """
        return (Tag.GRAPH_DIRECTED,
                Tag.GRAPH_TREE,
                Tag.GRAPH_MIXED,
                Tag.GRAPH_FOREST,
                Tag.GRAPH_ACYCLIC)


    def _reformat_unweighted_causal(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
        ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
                
        # Remove latent confounding and conflicted direction.
        return np.where(np.isin(adjacency, (-2,-1,2)), 0, adjacency)
    

    def _get_digraph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> nx.DiGraph:
        return nx.DiGraph(adjacency)

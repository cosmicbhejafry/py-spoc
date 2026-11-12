import numpy as np
import networkx as nx

from pyspoc._tag import Tag
from .._base import (
    GraphReducer,
    GraphType,
    GraphDirectedType,
    FilterCondition,
    AnyOf)


class DiGraphReducer(GraphReducer):

    def __init__(self):
        super().__init__()

    
    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[FilterCondition[Tag], ...]:
        """Graph tags that the GraphReducer is restricted to.
        """
        return (AnyOf(Tag.GRAPH_DIRECTED, Tag.GRAPH_MIXED),)
    

    def get_binary_adjacency(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
        return formatted_adjacency != 0


    def _get_directed_adjacency(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]:
        
        # Return as is if directed.
        if self._stat_directed_type == GraphDirectedType.DIRECTED:
            return formatted_adjacency

        # If mixed and weighted, we don't current support it.
        if self._stat_graph_type not in (GraphType.UNWEIGHTED, GraphType.CAUSAL_UNWEIGHTED):
            raise ValueError("Weighted mixed directed graphs are not currently supported.")
        
        # Otherwise remove undirected edges and return.
        return np.where(formatted_adjacency == -1, 0, formatted_adjacency)


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


    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating |
          np.integer |
          float |
          int):
        
        formatted_adjacency = self._get_directed_adjacency(formatted_adjacency=data)
        return super().reduce(formatted_adjacency)

import numpy as np
import networkx as nx

from typing import Literal

from pyspoc.types import SummaryType
from pyspoc._utils.numerical import summarize
from .._base import GraphSummaryReducer
from ._base import DiGraphReducer



class DiGraphDirectedDegreeSummary(GraphSummaryReducer, DiGraphReducer):

    _long_name = "Directed Graph Degree Summary"
    _identifier = "digraph-degree"
    
    def __init__(
            self,
            direction: Literal["in", "out"],
            summary_type: SummaryType = "mean"
    ):
        super().__init__(summary_type=summary_type)
        self._axis = 0 if direction == "in" else 1

    
    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:
                
        binary = self.get_binary_adjacency(
            formatted_adjacency=formatted_adjacency)
        degree = binary.sum(axis=self._axis)
        return float(summarize(degree, self._summary_type))


class DiGraphStrengthSummary(DiGraphDirectedDegreeSummary):

    _long_name = "Directed Graph Strength Summary"
    _identifier = "digraph-strength"
    
    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:

        binary = self.get_binary_adjacency(
            formatted_adjacency=formatted_adjacency)
        parents = np.where(binary, formatted_adjacency, 0)
        in_strength = parents.sum(axis=self._axis)
        return float(summarize(in_strength, self._summary_type))


class DiGraphReciprocity(DiGraphReducer):

    _long_name = "Directed Graph Reciprocity"
    _identifier = "digraph-reciprocity"
    
    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:
        
        binary = self.get_binary_adjacency(
            formatted_adjacency=formatted_adjacency)
        
        if binary.sum() == 0:
            return np.nan

        reciprocated = binary & binary.T
        return reciprocated.sum() / binary.sum()
        

class DiGraphFlowHierarchy(DiGraphReducer):

    _long_name = "Directed Graph Reciprocity"
    _identifier = "digraph-reciprocity"
    
    def _reduce_graph(
            self,
            formatted_adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:
        
        binary = self.get_binary_adjacency(
            formatted_adjacency=formatted_adjacency)
        
        if binary.sum() == 0:
            return np.nan

        G = self._get_digraph(binary)
        return nx.flow_hierarchy(G)

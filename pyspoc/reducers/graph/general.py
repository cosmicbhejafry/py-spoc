import numpy as np

from pyspoc.types import SummaryType
from pyspoc._utils.numerical import summarize
from ._base import GraphType, GraphReducer

class GraphEdgeCount(GraphReducer):

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)
    
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.integer:
        
        return (~np.isclose(adjacency, 0)).sum()


class GraphDegreeSummary(GraphReducer):
    
    def __init__(
            self,
            summary_type: SummaryType = "mean"):

        self._summary_type = summary_type
        super().__init__()

    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)
    

    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:

        is_neighbour = np.abs(adjacency) > 0
        row_neighbours = is_neighbour.sum(axis=0)
        col_neighbours = is_neighbour.sum(axis=1)
        degree = np.column_stack((row_neighbours, col_neighbours)).max(axis=1)
        return float(summarize(degree, self._summary_type))

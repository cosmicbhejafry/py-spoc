import numpy as np

from pyspoc.types import SummaryType
from pyspoc._utils.numerical import summarize
from ._base import GraphType, GraphReducer, GraphSummaryReducer

class GraphEdgeCount(GraphReducer):

    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)
    
    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.integer:
        
        return (~np.isclose(data, 0)).sum()


class GraphDegreeSummary(GraphSummaryReducer):
    
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

    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.floating | float:

        is_neighbour = np.abs(data) > 0
        row_neighbours = is_neighbour.sum(axis=0)
        col_neighbours = is_neighbour.sum(axis=1)
        degree = np.column_stack((row_neighbours, col_neighbours)).max(axis=1)
        return summarize(degree, self._summary_type)




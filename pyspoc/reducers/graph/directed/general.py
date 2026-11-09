import numpy as np

from typing import Literal

from pyspoc.types import SummaryType
from pyspoc._utils.numerical import summarize
from .._base import GraphSummaryReducer
from ._base import DirectedGraphReducer



class GraphDegreeSummary(DirectedGraphReducer, GraphSummaryReducer):

    def __init__(
            self,
            direction: Literal["in", "out"],
            summary_type: SummaryType = "mean"
    ):
        super().__init__(summary_type=summary_type)
        self._axis = 0 if direction == "in" else 1

    
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:

        is_parent = adjacency > 0
        degree = is_parent.sum(axis=self._axis)
        return float(summarize(degree, self._summary_type))


class GraphStrengthSummary(GraphDegreeSummary):
    
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> float:

        is_parent = adjacency > 0
        in_strength = adjacency[is_parent].sum(axis=self._axis)
        return float(summarize(in_strength, self._summary_type))


class GraphReciprocity(DirectedGraphReducer):
    
    def _reduce_graph(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.integer:

        is_parent = adjacency > 0
        row_parents = is_parent[np.triu_indices_from(is_parent, 1)]
        col_parents = is_parent[np.tril_indices_from(is_parent, -1)]
        return (row_parents & col_parents).sum()
        


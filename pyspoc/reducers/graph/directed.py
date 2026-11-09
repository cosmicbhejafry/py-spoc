import numpy as np
import networkx

from pyspoc._utils.numerical import summarize
from pyspoc._tag import Tag
from ._base import GraphReducer, GraphSummaryReducer

class DirectedGraphReducer(GraphReducer):

    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Graph tags that the GraphReducer is restricted to.
        """
        return (Tag.GRAPH_DIRECTED,
                Tag.GRAPH_TREE,
                Tag.GRAPH_MIXED,
                Tag.GRAPH_FOREST)

    
    def _encode_digraph(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ):
        return networkx.DiGraph(data)
        


class GraphInDegreeSummary(DirectedGraphReducer, GraphSummaryReducer):

    
    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.floating | float:

        is_parent = data > 0
        in_degree = is_parent.sum(axis=0)
        return summarize(in_degree, self._summary_type)


class GraphOutDegreeSummary(DirectedGraphReducer, GraphSummaryReducer):
    
    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.floating | float:

        is_parent = data > 0
        out_degree = is_parent.sum(axis=1)
        return summarize(out_degree, self._summary_type)


class GraphReciprocity(DirectedGraphReducer, GraphSummaryReducer):
    
    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.integer:

        is_parent = data > 0
        row_parents = is_parent[np.triu_indices_from(is_parent, 1)]
        col_parents = is_parent[np.tril_indices_from(is_parent, -1)]
        return (row_parents & col_parents).sum()
        

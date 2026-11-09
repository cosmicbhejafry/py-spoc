import numpy as np

from enum import Enum

from pyspoc.reducers.base import Reducer
from pyspoc.statistics.base import Statistic
from pyspoc.types import SummaryType
from pyspoc._tag import Tag, Signature
from pyspoc._utils.numerical import summarize


class GraphType(Enum):
    UNWEIGHTED = "unweighted"
    WEIGHTED = "weighted"
    CAUSAL_UNWEIGHTED = "causal_unweighted"
    CAUSAL_WEIGHTED = "causal_weighted"


class GraphReducer(Reducer):

    @classmethod
    def get_statistic_tags_filter(cls) -> tuple[Tag, ...]:
        """Graph tags that the GraphReducer is restricted to.
        """
        return tuple(Tag(tag) for name, tag
            in Tag._member_map_.items()
            if "graph" in name.lower())

    
    @classmethod
    def _supported_graph_types(cls):
        return (
            GraphType.UNWEIGHTED,
            GraphType.WEIGHTED,
            GraphType.CAUSAL_UNWEIGHTED,
            GraphType.CAUSAL_WEIGHTED)


    def _has_causal_signature(
            self,
            stat: Statistic) -> bool:
        return Signature.CAUSAL in stat.get_signatures()


    def _get_adjacency_type(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            stat: Statistic
    ) -> tuple[GraphType | None,
               np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]] | None]:

        if not np.isfinite(adjacency).all():
            return None, adjacency

        is_causal = self._has_causal_signature(stat)

        if np.issubdtype(adjacency.dtype, np.integer):
            return self._get_integer_adjacency_type(
                adjacency=adjacency,
                is_causal=is_causal
            ), adjacency

        if not np.issubdtype(adjacency.dtype, np.floating):
            return None, adjacency

        int_adjacency = np.rint(adjacency)

        if not np.isclose(adjacency, int_adjacency, rtol=0).all():
            if self._has_causal_signature(stat):
                return GraphType.CAUSAL_WEIGHTED, adjacency

            return GraphType.WEIGHTED, adjacency

        graph_type = self._get_integer_adjacency_type(
            adjacency=adjacency,
            is_causal=is_causal
        )

        if graph_type in (GraphType.CAUSAL_UNWEIGHTED, GraphType.UNWEIGHTED):
            return graph_type, int_adjacency

        return graph_type, adjacency
    

    def _get_integer_adjacency_type(
            self,
            adjacency: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]],
            is_causal: bool):

        unweighted_causal_components = (-2,-1,0,1,2)
        unweighted_components = (-1,0,1)
        amax = adjacency.max()
        amin = adjacency.min()

        if is_causal:
            if (amax in unweighted_causal_components
                and amin in unweighted_causal_components):
                return GraphType.CAUSAL_UNWEIGHTED

            return GraphType.CAUSAL_WEIGHTED

       
        if amax in unweighted_components and amin in unweighted_components:
            return GraphType.UNWEIGHTED

        return GraphType.WEIGHTED


    def calculate(
            self,
            statistic: Statistic
    ) -> (np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] |
          np.ndarray[tuple[()], np.dtype[np.floating | np.integer]] |
          np.floating |
          np.integer |
          float |
          int):
        
        adjacency = self._get_statistic_result(statistic)

        if not isinstance(adjacency, np.ndarray):
            raise ValueError("GraphReducer can only reduce statistics "
                "that return an ndarray result.")

        if adjacency.ndim < 2:
            raise ValueError("GraphReducer can only reduce statistics that "
                "return a 2D array result.")

        graph_type = self._get_adjacency_type(adjacency, statistic)
        supported_graph_types = self._supported_graph_types()

        if graph_type not in supported_graph_types:
            raise ValueError(
                "{name} can only reduce adjacency matrices of the following graph types:\n"
                "\t{graph_types}".format(name=self.get_name(),
                                         graph_types=supported_graph_types))
        
        return super().calculate(statistic)
    

class GraphSummaryReducer(GraphReducer):

    def __init__(
            self,
            summary_type: SummaryType = "mean"):

        self._summary_type = summary_type
        super().__init__()


    def reduce(
            self,
            data: np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
    ) -> np.floating | float:

        result = super().reduce(data)

        if isinstance(result, np.ndarray):
            return summarize(result, self._summary_type)

        return result

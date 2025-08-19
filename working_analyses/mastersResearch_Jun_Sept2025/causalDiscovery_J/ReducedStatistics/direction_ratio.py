from pyspoc import ReducedStatistic
import numpy as np
from pyspoc.dataset import Dataset
from causallearn.search.ConstraintBased.PC import pc

class DirectedRatio(ReducedStatistic):
    """
    Runs the PC algorithm on the dataset and counts the ratio of directed edges to undirected edges.
    """

    def __init__(self, alpha: float = 0.05):
        name = "DirectedEdgeRatio"
        identifier = ""
        labels = ["directed_edge_ratio"]
        
        super().__init__()
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "my_new_reducer_name"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["my_new_reducer_label_1",
                "my_new_reducer_label_2",
                "my_new_reducer_label_n"]

    def compute(self, data: np.ndarray) -> float:
        # Run the PC algorithm and extract the custom adjacency matrix (see below/causal-learn documentation)
        # i->j: A_ji=1; A_ij = -1
        # i-j: A_ji = A_ij = -1
        # i<->j: A_ji = A_ij = 1
        cg = pc(data, alpha=self.alpha, indep_test='fisherz', verbose=False)
        adj_matrix = cg.G.graph

        total_edges = 0
        directed_edges = 0

        # Iterate through all pairs of nodes in the adjacency matrix and count the *uni-directed edges*
        for i in range(adj_matrix.shape[0]):
            for j in range(i + 1, adj_matrix.shape[0]):

                # Edge exists iff both entries are nonzero; check one
                if adj_matrix[i,j] != 0:
                    total_edges += 1

                    # Directed edge if values are opposite (1 and -1)
                    if (adj_matrix[i,j] == 1 and adj_matrix[j,i] == -1) or (adj_matrix[i,j] == -1 and adj_matrix[j,i] == 1):
                        directed_edges += 1

        # avoid division by zero
        smoothed_ratio = directed_edges / (total_edges +1e-8)
        return np.array(smoothed_ratio)
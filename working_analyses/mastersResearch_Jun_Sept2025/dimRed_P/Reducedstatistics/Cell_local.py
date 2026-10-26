import numpy as np
from cell_auxiliary_functions import reducer_reader, get_knn, get_Jaccard
from sklearn.preprocessing import scale
from pyspoc import ReducedStatistic


class Cell_local(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, l_dist = 'l1', n_neighbours = 15):

        # Calling base class initialiser.
        super().__init__()

        self.l_dist = l_dist
        self.n_neighbours = n_neighbours
        self.method = method
        self.reduced_dimensionality = reduced_dimensionality
        self.reducer = reducer_reader(self.method)(self.reduced_dimensionality)

    @property
    def name(self) -> str:
        return "Cell_local"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["my_new_reducer_label_1",
                "my_new_reducer_label_2",
                "my_new_reducer_label_n"]

    def compute(self, data: np.ndarray) -> float:

        # Center and scals data
        scaled_data = scale(data)

        # Compute nearest neighbours for the original data
        _, orig_indices = get_knn(data, n_neigh = self.n_neighbours, l_dist=self.l_dist)

        # Perform dimensionality reduction
        Z = self.reducer.fit_transform(scaled_data)

        # Recompute nearest neighbours in the dimensionally reduced data
        _, reduced_indices = get_knn(Z, n_neigh = self.n_neighbours, l_dist=self.l_dist)

        # Compare both sets using the jaccard distance
        jac_distances = get_Jaccard(orig_indices, reduced_indices)

        # Take the mean across all samples so as to have just one value
        output = np.mean(jac_distances)

        return output
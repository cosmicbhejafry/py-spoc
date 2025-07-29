import numpy as np
import tools as tl
from cell_auxiliary_functions import reducer_reader
from sklearn.preprocessing import scale
from pyss import ReducedStatistic
from typing import Union


class cell_local(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, set_p = 1, n_neighbours = 15):

        # Calling base class initialiser.
        super().__init__()

        self.set_p = set_p
        self.n_neighbours = n_neighbours
        self.method = method
        self.reduced_dimensionality = reduced_dimensionality
        self.reducer = reducer_reader(self.method)(self.reduced_dimensionality)

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

    def compute(self, data: np.ndarray) -> Union[np.ndarray, float]:

        # log-normalise data
        log_data = np.log1p(data)

        # Center and scale log-normalised data
        scaled_data = scale(log_data)

        # Compute nearest neighbours for the original data
        orig_indices = tl.getNeighbors(log_data, n_neigh = self.n_neighbours, p=self.set_p)

        # Perform dimensionality reduction
        Z = self.reducer.fit_transform(scaled_data)

        # Recompute nearest neighbours in the dimensionally reduced data
        reduced_indices = tl.getNeighbors(Z, n_neigh = self.n_neighbours, p=self.set_p)

        # Compare both sets using the jaccard distance
        jac_distances = tl.getJaccard(orig_indices, reduced_indices)

        # Take the mean across all samples so as to have just one value
        output = np.mean(jac_distances)

        return output
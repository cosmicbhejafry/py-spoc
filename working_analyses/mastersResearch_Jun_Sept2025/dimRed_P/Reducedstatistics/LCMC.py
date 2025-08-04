import numpy as np
from cell_auxiliary_functions import reducer_reader
from coranking_auxiliary_functions import ranking_matrix, coranking_matrix, compute_QNN, compute_LCMC, compute_auc_LCMC
from sklearn.metrics import pairwise_distances
from pyspoc import ReducedStatistic


class LCMC(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2):

        # Calling base class initialiser.
        super().__init__()

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality
        self.reducer = reducer_reader(self.method)(self.reduced_dimensionality)

    @property
    def name(self) -> str:
        return "Q_local"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["my_new_reducer_label_1",
                "my_new_reducer_label_2",
                "my_new_reducer_label_n"]

    def compute(self, data: np.ndarray) -> float:

        # Dimensionally reduce the data
        X = data
        Z = self.reducer.fit_transform(X)

        # Construct the distance matrices
        D  = pairwise_distances(X)
        D_z = pairwise_distances(Z)

        # Compute ranking and co-ranking matrices
        R = ranking_matrix(D)
        R_z = ranking_matrix(D_z)
        Q = coranking_matrix(R, R_z)

        # Compute QNN, and LCMC
        QNN = compute_QNN(Q)
        LCMC = compute_LCMC(QNN)
        
        # Return the average LCMC
        return compute_auc_LCMC(LCMC)
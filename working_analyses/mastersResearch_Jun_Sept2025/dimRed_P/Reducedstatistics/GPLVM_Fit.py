import numpy as np
import GPy 
from pyspoc import ReducedStatistic


class GPLVM_Fit(ReducedStatistic):

    def __init__(self, ARD_initial_dimension = 15):

        # Calling base class initialiser.
        super().__init__()

        self.method = method
        self.ARD_initial_dimension = ARD_initial_dimension

    @property
    def name(self) -> str:
        return "GPLVM-Fit"

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

        # Compute Continuity and take the AUC (average in this case)
        C = compute_continuity(Q)
        auc_C = compute_auc_C(C)

        return auc_C
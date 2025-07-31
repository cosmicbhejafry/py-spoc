import numpy as np
from cell_auxiliary_functions import reducer_reader
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr
from pyspoc import ReducedStatistic


class Distance_similarity(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, correlation_coefficient = 'Pearson'):

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality
        self.reducer = reducer_reader(self.method)(self.reduced_dimensionality)

        correlation_dict = {'Pearson': pearsonr,
                            'Spearman': spearmanr}

        self.correlation_coefficient = correlation_dict[correlation_coefficient]

        # Calling base class initialiser.
        super().__init__()

    @property
    def name(self) -> str:
        return "Distance_similarity"

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

        # Residual Variance of the two distance matrices
        Res_Var = 1 - (self.correlation_coefficient(D.flatten(), D_z.flatten())[0])**2

        return Res_Var
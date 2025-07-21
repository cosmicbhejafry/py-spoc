import numpy as np
import umap
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr
from pyss import ReducedStatistic
from typing import Union


class Distance_similarity(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, correlation_coefficient = 'Pearson'):

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality

        reducer_dict = {
            'PCA'   : PCA,
            'NMF'   : NMF,
            'TSNE'  : TSNE,
            'Isomap': Isomap,
            'UMAP'  : umap.UMAP,
        }

        correlation_dict = {'Pearson': pearsonr,
                            'Spearman': spearmanr}

        self.correlation_coefficient = correlation_dict[correlation_coefficient]
        self.reducer = reducer_dict[method](n_components=self.reduced_dimensionality)

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

    def compute(self, data: np.ndarray) -> Union[np.ndarray, float]:

        # Dimensionally reduce the data
        X = data
        Z = self.reducer.fit_transform(X)

        # Construct the distance matrices
        D  = pairwise_distances(X)
        D_z = pairwise_distances(Z)

        # Residual Variance of the two distance matrices
        Res_Var = 1 - (self.correlation_coefficient(D.flatten(), D_z.flatten())[0])**2

        return Res_Var
import numpy as np
import umap
from coranking_auxiliary_functions import ranking_matrix, coranking_matrix, slice_Q, compute_continuity, compute_auc_C
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import pairwise_distances
from pyss import ReducedStatistic
from typing import Union


class Continuity(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2):

        # Calling base class initialiser.
        super().__init__()

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality

        reducer_dict = {
            'PCA'   : PCA,
            'NMF'   : NMF,
            'TSNE'  : TSNE,
            'Isomap': Isomap,
            'UMAP'  : umap.UMAP,
        }

        self.reducer = reducer_dict[method](n_components=self.reduced_dimensionality)

    @property
    def name(self) -> str:
        return "Continuity"

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

        # Compute ranking and co-ranking matrices
        R = ranking_matrix(D)
        R_z = ranking_matrix(D_z)
        Q = coranking_matrix(R, R_z)

        # Compute Continuity and take the AUC (average in this case)
        C = compute_continuity(Q)
        auc_C = compute_auc_C(C)

        return auc_C
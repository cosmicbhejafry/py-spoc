import numpy as np
from cell_auxiliary_functions import reducer_reader
from sklearn.metrics import pairwise_distances
from pyspoc import ReducedStatistic
from scipy.stats import pearsonr, spearmanr
from working_analyses.mastersResearch_Jun_Sept2025.dimRed_P.Reducedstatistics.coranking_auxiliary_functions import *


class Coranking_metrics(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, correlation_coefficient = 'Pearson'):

        # Calling base class initialiser.
        super().__init__()

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality
        self.reducer = reducer_reader(self.method)(self.reduced_dimensionality)
        correlation_dict = {'Pearson': pearsonr,
                            'Spearman': spearmanr}

        self.correlation_coefficient = correlation_dict[correlation_coefficient]

    @property
    def name(self) -> str:
        return "Coranking_metrics"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return [
        "Distance-Similarity (DS)",
        "Continuity (AUC)",
        "Trustworthiness (AUC)",
        "QNN (AUC)",
        "LCMC (AUC)",
        "Q_local",
        "Q_global",
    ]

    def compute(self, data: np.ndarray) -> float:

        output = []

        # Dimensionally reduce the data
        X = data
        Z = self.reducer.fit_transform(X)

        # Construct the distance matrices
        D  = pairwise_distances(X)
        D_z = pairwise_distances(Z)

        # Distance similarity
        DS = self.correlation_coefficient(D.flatten(), D_z.flatten())[0]**2
        output.append(DS)

        # Compute ranking and co-ranking matrices
        R = ranking_matrix(D)
        R_z = ranking_matrix(D_z)
        Q = coranking_matrix(R, R_z)

        # Compute Continuity and take the AUC (average in this case)
        C = compute_continuity(Q)
        auc_C = compute_auc_C(C)
        output.append(auc_C)

        # Compute Trustworthiness and take the AUC 
        T = compute_trustworthiness(Q)
        auc_T = compute_auc_T(T)
        output.append(auc_T)

        # Compute QNN and take the AUC
        QNN = compute_QNN(Q)
        auc_QNN = compute_auc_QNN(QNN)
        output.append(auc_QNN)

        # Compute LCMC and take the AUC
        LCMC = compute_LCMC(QNN)
        auc_LCMC = compute_auc_LCMC(LCMC)
        output.append(auc_LCMC)

        # Compute Q_local
        kmax = compute_kmax(LCMC)
        Qlocal = compute_Qlocal(QNN, kmax)
        output.append(Qlocal)
        
        # Compute Q_global
        Qglobal = compute_Qglobal(QNN, kmax)
        output.append(Qglobal)

        return np.array(output)
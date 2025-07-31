import numpy as np
from cell_auxiliary_functions import reducer_reader, rankClus
from co_ranking_auxiliary_functions import ranking_matrix, coranking_matrix, compute_QNN, compute_LCMC, compute_auc_LCMC
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pyspoc import ReducedStatistic

class Cell_cluster(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2, l_dist = 'l1', pca_for_clusters = 50):

        # Calling base class initialiser.
        super().__init__()

        self.l_dist = l_dist
        self.pca_for_clusters = pca_for_clusters
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

    def compute(self, data: np.ndarray) -> float:

        # log-normalise data
        log_data = np.log1p(data)

        # Center and scale log-normalised data
        scaled_data = scale(log_data)

        # If pca_for_clusters is passed, reduce to that amount of principal components before computing the clusters
        if self.pca_for_clusters:
            pca = PCA(n_components=self.pca_for_clusters)
            X = pca.fit_transform(scaled_data)   # scaled_data is your preprocessed data
        else:
            X = scaled_data

        # To pick the number of clusters we iterate from 2 to 30 trying to find the one with the least BIC
        bics = []
        for k in range(2,30):
            gm = GaussianMixture(n_components=k).fit(X)
            bics.append(gm.bic(X))
        best_K = np.argmin(bics) + 2 # Add 2 since our loop starts at 2

        # Then refit gmm with the optimal number of clusters
        gmm = GaussianMixture(
            n_components=best_K,
        ).fit(X)

        # Predict labels for our data
        cluster_labels = gmm.predict(X)

        # Compute average pairwise distances between clusters
        D = rankClus(X, cluster_labels, self.l_dist)

        # Perform dimensionality reduction
        Z = self.reducer.fit_transform(scaled_data)

        # Compute average pairwise distances between clusters in the reduced space
        D_z = rankClus(Z, cluster_labels, self.l_dist)

        # Compute ranking and co-ranking matrices in the high- and low- dimensional spaces
        R = ranking_matrix(D)
        R_z = ranking_matrix(D_z)
        Q = coranking_matrix(R, R_z)

        # Compute Q_NN and LCMC too
        QNN = compute_QNN(Q)
        LCMC = compute_LCMC(QNN)

        # Return the AUC of the LCMC 
        return compute_auc_LCMC(LCMC)
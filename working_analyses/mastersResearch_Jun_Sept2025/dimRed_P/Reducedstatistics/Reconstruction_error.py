import numpy as np
from sklearn.decomposition import PCA, NMF
from pyspoc import ReducedStatistic
from typing import Union


class Reconstruction_error(ReducedStatistic):

    def __init__(self, method = 'PCA', reduced_dimensionality = 2):

        self.method = method
        self.reduced_dimensionality = reduced_dimensionality

        reducer_dict = {'PCA': PCA,
                        'NMF': NMF}

        self.reducer = reducer_dict[method](n_components=self.reduced_dimensionality)

        # Calling base class initialiser.
        super().__init__()

    @property
    def name(self) -> str:
        return "Reconstruction_error"

    @property
    def identifier(self) -> str:
        return "my_new_reducer_identifier"

    @property
    def labels(self) -> list[str]:
        return ["my_new_reducer_label_1",
                "my_new_reducer_label_2",
                "my_new_reducer_label_n"]

    def recon_MSE(self,X,X_reconstructed):
        '''
        Reconstruction relateive MSE
        '''
        assert X.shape == X_reconstructed.shape
        return np.linalg.norm(X-X_reconstructed, ord='fro')**2/(X.shape[0]*X.shape[1])

    def calculate_recon_error(self,X, X_reconstructed):
        '''
        Recontruction errors
        mse : mean squared error between X and X_reconstructed
        ams : mean square of X

        Return:
        rmse (=mse/ams) : relative mean squared error
        '''
        assert X.shape == X_reconstructed.shape

        mse = recon_MSE(X, X_reconstructed) # mean square error
        ams = recon_MSE(X, np.zeros(X.shape)) # mean square of original data matrix
        rmse = mse/ams # relative mean square error

        return rmse

    def recon_rMSE(self,X, X_reconstructed):
        '''
        relative mean squared error
        '''
        return recon_MSE(X, X_reconstructed) / recon_MSE(X, np.zeros(X.shape))

    def compute(self, data: np.ndarray) -> Union[np.ndarray, float]:

        # Dimensionally reduce the data and reconstruct the original one from the result
        X = data
        Z = self.reducer.fit_transform(X)
        X_reconstructed = self.reducer.inverse_transform(Z)

        # Compute the relative mean square error between the original dataset and the reconstructed one
        rmse = self.calculate_recon_error(X, X_reconstructed)

        return rmse
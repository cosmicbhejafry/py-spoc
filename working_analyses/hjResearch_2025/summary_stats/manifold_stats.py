"""
refactoring manifold stats, vectorizing, etc.
"""
from pyspoc import ReducedStatistic
import numpy as np
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap, trustworthiness
from sklearn.metrics import mean_squared_error

class DimReductionMethodsBase(ReducedStatistic):

    SUPPORTED_METHODS = ['PCA', 'TSNE', 'ISOMAP', 'NMF']

    ## TODO: include **kwargs figure out correct syntax
    ## TODO: design Q, better to have self.X_reduced or better to return it from the func? (linked to how caching works --> ask garry?)
    
    def __init__(self, method : str = 'PCA',embedding_dim=2):
        
        self.method = method.upper()
        self.embedding_dim = embedding_dim
        self.X_reduced_fit = None
        self.X_reduced = None
        
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Invalid metric '{method}'. Must be one of: {', '.join(self.SUPPORTED_METHODS)}"
            )
        
    def calculate(self, dataset: Dataset):
        result = super().calculate(dataset)
        return result


    def _fit_method(self, X: np.ndarray) -> None:
        
        if self.method == 'PCA':
            X_reduced_fit = PCA(n_components=self.embedding_dim).fit(X)
            self.X_reduced_fit = X_reduced_fit
        
        if self.method == 'TSNE':            
            X_reduced_fit = TSNE(n_components=self.embedding_dim, perplexity=30, random_state=0,method='barnes_hut').fit(X)
            self.X_reduced_fit = X_reduced_fit

        if self.method == 'ISOMAP':            
            X_reduced_fit = Isomap(n_components=self.embedding_dim).fit(X)
            self.X_reduced_fit = X_reduced_fit

        if self.method == 'NMF':
            X_reduced_fit = NMF(n_components=self.embedding_dim, init='nndsvd').fit(X)
            self.X_reduced_fit = X_reduced_fit

    def _get_reduced_X(self,X: np.ndarray) -> None:
        
        if self.X_reduced_fit is None:
            self._fit_method(self.method,X)

        self.X_reduced = self.X_reduced_fit.transform()
    
    def _reconstruction_error(self,X: np.ndarray) -> float:
        
        # fit the method if not done already
        if self.X_reduced_fit is None:
            self._fit_method(X)
        
        if self.method=='ISOMAP':
            return self.X_reduced_fit.reconstruction_error()
        
        else:
            if self.X_reduced is None:
                self._get_reduced_X(X)
                
            if self.method=='PCA':
                X_recovered = PCA.inverse_transform(self.X_reduced)
                return mean_squared_error(X,X_recovered)

            if self.method=='NMF':
                X_recovered = NMF.inverse_transform(self.X_reduced)
                return mean_squared_error(X,X_recovered)
            
            if self.method=='TSNE':
                return ValueError('no reconstruction for tsne')
    
    def _ranking_matrix(self,X:np.ndarray):
        pass
    
    def _coranking_matrix(self):
        pass
    
    
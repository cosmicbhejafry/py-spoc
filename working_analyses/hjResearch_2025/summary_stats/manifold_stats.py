"""
refactoring manifold stats, vectorizing, etc.

the pyDrMetrics implementations they seem a bit inconsistent to me, they dont match the inbuilt sklearn metrics

my current implementations are closer to this package:
https://github.com/samueljackson92/coranking/tree/master
associated with this paper:
Lee, John A., and Michel Verleysen. "Quality assessment of dimensionality reduction: Rank-based criteria." Neurocomputing 72.7 (2009): 1431-1443.

TODO: discuss with G- the cell statistics are associated with clustering, check if G has implemented before redoing?
"""
from pyspoc import ReducedStatistic
import numpy as np
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap, trustworthiness, UMAP
from sklearn.metrics import pairwise_distances

from scipy.stats import rankdata

class DimReductionMethodsBase(ReducedStatistic):

    METHODS = {
        "PCA":    (PCA,    {}),
        "TSNE":   (TSNE,   {"perplexity": 30, "init": "pca"}),
        "ISOMAP": (Isomap, {}),
        # "UMAP":   (UMAP,   {"n_neighbors": 15, "min_dist": 0.1}), # excluding umap for now as it needs extra dependencies
        # "NMF":    (NMF,    {"init": "nndsvd", "max_iter": 500}), # excluding NMF for now asit needs additional checks on the input matrix
    }
    
    # ------------- CLASS DEFINITIONS - start -------------    
    def __init__(self, method : str = 'PCA',embedding_dim=2,random_state=None,**kwargs):
        super().__init__()
        
        self.method = method.upper()
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.set_self_dist_inf_ranks=True
                
        if self.method not in self.METHODS.keys():
            raise ValueError(
                f"Invalid metric '{self.method}'. Must be one of: {', '.join(self.METHODS.keys())}"
            )
        # self.X_reduced_fit = None
        # self.X_reduced = None
        
    def calculate(self, dataset: Dataset):
        result = super().calculate(dataset)
        return result
    # ------------- CLASS DEFINITIONS - end -------------

    # ------------- FUNCTIONS ON X_REDICED - start -------------    
    def _fit_method(self, X: np.ndarray) -> np.ndarray:
        
        method_fn, method_kwargs = self.METHODS[self.method][0], self.METHODS[self.method][1]
        params = {**method_kwargs, **self.kwargs, "n_components": self.embedding_dim}
        
        if self.method == 'TSNE':
            if self.random_state is not None:
                params['random_state'] = self.random_state

        X_reduced = method_fn(**params).fit_transform(X)
        
        return X_reduced

    def _residual_variance(self, X : np.ndarray, X_reduced : np.ndarray):
        pass

    def _reconstruction_error(self, X : np.ndarray, X_reduced : np.ndarray):
        pass
    
    # ------------- FUNCTIONS ON X_REDICED - end -------------    

    # ------------- RANKING AND CO-RANKING BASED FUNCTIONS - start -------------
    def _ranking_matrix(self, X : np.ndarray, metric = 'euclidean',solver='nilrad') -> np.ndarray:
                    
        dist_X = pairwise_distances(X,metric=metric)
        
        # DEFAULT: set self-distances to inf, avoid ties / floatpoint errors
        # affects slicing of Q
        if self.set_self_dist_inf_ranks:
            np.fill_diagonal(dist_X, np.inf)
            # slice Q as Q[:-1,:-1] to remove self neighbours
        
        if solver=='nilrad':
            # Computing ranking_matrix using np.argsort twice on every row [Nilrad]            
            R = [np.argsort(np.argsort(row)) for row in dist_X]
            return np.array(R, dtype = 'uint')        
        elif solver=='scipy':
            R = rankdata(dist_X, axis=1, method="ordinal").astype(int) - 1
            return np.array(R, dtype = 'uint')
        else:            
            raise ValueError('not supported solver')
                    
    def _coranking_matrix(self,R1 : np.ndarray ,R2 : np.ndarray,solver='bincount-ravel') -> np.ndarray:
        assert R1.shape == R2.shape
        
        N = R1.shape[0]
        if solver == 'bincount-ravel':
            Q = np.bincount((R1 * N + R2).ravel(), minlength=N*N).reshape(N,N)        
            return Q
        elif solver=='for-loop':
            # [from coranking_auxiliary_functions.py]
            Q = np.zeros(R1.shape)
            m = len(Q)
            for i in range(m):
                for j in range(m):
                    k = int(R1[i,j])
                    l = int(R2[i,j])
                    Q[k,l] += 1
            return Q
        else:
            raise ValueError('not supported solver')
    
    def _drop_coranking_self_nb(self, Q : np.ndarray) -> np.ndarray:
        if self.set_self_dist_inf_ranks:
            # self-distances set to np.inf while computing R and Q
            return Q[:-1,:-1]
        else:
            # self-distances computed as 0 while computing R and Q
            return Q[1:,1:]
    
    # def _trustworthiness_k(self,Q:np.ndarary,k):
    # could write functions for each k separately, but more efficient to loop it i think
    #     pass
    
    def _get_k_lst_TC(self,N,k_mode,k_min,k_max):
        if k_mode == 'N/2':
            return range(1,(N + 1) // 2)
            
        elif k_mode == 'all':
            return range(1,N-1)
        
        elif k_mode == 'custom-k_min-k_max':
            if k_min is None or k_max is None or not (1 <= k_min < k_max <= N - 1):
                raise ValueError(f"need 1 <= k_min < k_max <= {N - 1}, got k_min={k_min}, k_max={k_max}")    
            
            return range(k_min,k_max)
        
        else:
            raise ValueError('invalid k_mode for trustworthiness curve')
        
            
    def _trustworthiness(self, Q : np.ndarray, k_mode = 'N/2',k_min=None,k_max=None):
        # refer to sec 3. https://www.sciencedirect.com/science/article/pii/S0925231209000101        
        # matches the sklearn trustworthiness implementation for k < N/2
        # default is N/2
        # i think this function could benefit from further refactoring or checks
        # im not sure if the k_mode, k_min, k_max is best but that is what i could figure right now       
        
        N = Q.shape[0]
        Q_sliced = self._drop_coranking_self_nb(Q)
        
        k_lst = self._get_k_lst_TC(N,k_mode,k_min,k_max)
        T = np.zeros(len(k_lst))
        true_rank = np.arange(1,N) #for W's
        
        for ix,k in enumerate(k_lst):
            # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
            Qs = Q_sliced[k:, :k]
            
            # a column vector of weights. weight = rank error = actual_rank - k            
            W = (true_rank[k:]-k).reshape(-1,1)
                
            if k < N/2:
                denom = N*k*(2*N - 3*k - 1)            
            else:
                denom = N * (N-k) * (N-k-1)
            
            T[ix] = 1 - 2*np.sum(Qs * W) / denom
            
        return T, np.array(k_lst)

    def _continuity(self, Q : np.ndarray, k_mode = 'N/2',k_min=None,k_max=None):
        # refer to sec 3. https://www.sciencedirect.com/science/article/pii/S0925231209000101        
        # default is N/2
        # similar to _trustworthiness
        
        N = Q.shape[0]
        Q_sliced = self._drop_coranking_self_nb(Q)
        
        k_lst = self._get_k_lst_TC(N,k_mode,k_min,k_max)
        
        C = np.zeros(len(k_lst))
        true_rank = np.arange(1,N) #for W's
        
        for ix,k in enumerate(k_lst):
            # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
            Qs = Q_sliced[:k, k:]
            
            # a column vector of weights. weight = rank error = actual_rank - k            
            W = (true_rank[k:]-k).reshape(1,-1)
                
            if k < N/2:
                denom = N*k*(2*N - 3*k - 1)            
            else:
                denom = N * (N-k) * (N-k-1)
            
            C[ix] = 1 - 2*np.sum(Qs * W) / denom
            
        return C, np.array(k_lst)

    def _lcmc(self, Q : np.ndarray,k_mode='all',k_min=None,k_max=None):
        # K/(1-N) + 1/NK * Qnn
        
        N = Q.shape[0]
        Q_sliced = self._drop_coranking_self_nb(Q)
        
        k_lst = self._get_k_lst_TC(N,k_mode,k_min,k_max)
        k_arr = np.array(k_lst)
        # lambda-loop version
        # lcmc_k = lambda k : k/(1-N) + np.sum(Q_sliced[:k,:k])/(N*k)
        # np.array([lcmc_k(k) for k in k_lst])

        # vectorize using cumulative sums
        block = np.cumsum(np.cumsum(Q_sliced[:-1,:-1], 0), 1)         
        lcmc = k_arr/(1 - N) + block[k_arr-1,k_arr-1]/(N*k_arr)       
        return lcmc, k_arr

    def _Q_NN(self, Q : np.ndarray, k_mode='all',k_min=None,k_max=None):
        N = Q.shape[0]
        Q_sliced = self._drop_coranking_self_nb(Q)
        
        k_lst = self._get_k_lst_TC(N,k_mode,k_min,k_max)
        k_arr = np.array(k_lst)
        # vectorize using cumulative sums
        block = np.cumsum(np.cumsum(Q_sliced[:-1,:-1], 0), 1)         
        qnn = block[k_arr-1,k_arr-1]/(N*k_arr)       
        return qnn, k_arr
              
    def _Q_local_global(self, Q : np.ndarray, lcmc = None):
        
        N = Q.shape[0]
        if lcmc is None:
            # find argmax of lcmc            
            lcmc, k_arr = self._lcmc(Q, k_mode='all')
        else:
            k_lst = self._get_k_lst_TC(N,'all')
            k_arr = np.array(k_lst)
        
        i_argmax = int(np.argmax(lcmc))
        k_argmax = int(k_arr[i_argmax])

        qnn = lcmc - k_arr/(1-N)
        q_local =  np.mean(qnn[:i_argmax + 1])
        q_global =  np.mean(qnn[i_argmax:])       
        
        return np.array([q_local,q_global]), k_argmax
    
    # ------------- RANKING AND CO-RANKING BASED FUNCTIONS - end -------------

# --------------------------

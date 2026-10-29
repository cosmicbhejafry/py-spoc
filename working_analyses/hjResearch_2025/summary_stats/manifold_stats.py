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
from typing import TypeGuard, Literal, Any, overload

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap, trustworthiness
from sklearn.metrics import pairwise_distances

from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr

class DimReductionMethodsBase(ReducedStatistic):

    METHODS = {
        "PCA":    (PCA,    {}),
        "TSNE":   (TSNE,   {"perplexity": 30, "init": "pca"}),
        "ISOMAP": (Isomap, {}),
        # "UMAP":   (UMAP,   {"n_neighbors": 15, "min_dist": 0.1}), # excluding umap for now as it needs extra dependencies
        # "NMF":    (NMF,    {"init": "nndsvd", "max_iter": 500}), # excluding NMF for now asit needs additional checks on the input matrix
    }
    
    # GOODNESS_OF_FIT = {
        # "PCA" : lambda model, X, Xr : reconstruction error,
        # "TSNE" : lambda model, X, Xr : model.kl_divergence_,
        # "ISOMAP" : lambda model, X, Xr : reconstruction_error()
    # }
    
    # ------------- CLASS DEFINITIONS - start -------------    
    def __init__(self, method : str = 'PCA',embedding_dim=2,
                 random_state=None,
                 dim: Literal["n", "p"] = "p",
                 nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise"):
        super().__init__(dim=dim,nan_policy=nan_policy)
        
        self.method = method.upper()
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.kwargs = {} #removing **kwargs from defn right now
        
        self.set_self_dist_inf_ranks=True
                
        if self.method not in self.METHODS.keys():
            raise ValueError(
                f"Invalid metric '{self.method}'. Must be one of: {', '.join(self.METHODS.keys())}"
            )
        # self.X_reduced_fit = None
        # self.X_reduced = None
        
    def _calculate(self, data : np.ndarray):
        result = super()._calculate(data)
        return result
    # ------------- CLASS DEFINITIONS - end -------------

    # ------------- FUNCTIONS ON X_REDICED - start -------------    
    def _fit_method(self, X: np.ndarray) -> np.ndarray:
        
        method_fn, method_kwargs = self.METHODS[self.method][0], self.METHODS[self.method][1]
        params = {**method_kwargs, **self.kwargs, "n_components": self.embedding_dim}
        
        if self.method == 'TSNE': #todo- replace w/ something more general?
            if self.random_state is not None:
                params['random_state'] = self.random_state
                
        model = method_fn(**params)
        X_reduced = model.fit_transform(X)
        
        # fit_score = float(self.GOODNESS_OF_FIT[self.method](model, X, X_reduced)) 
        # todo- can incorporate fit scores
        
        return X_reduced

    def _reconstruction_error(self):
        # reconstruction errors or goodness of fit in a way ? 
        if self.method == "PCA":
            pass
            # inverse transform
            # find RE

        if self.method == "Isomap":
            pass
            # use pairwise distances and 
            # reimplement the reconstruction_error() function for sklearn
        return np.nan
    
    def _pairwise_distance_mtx(self,X,metric='euclidean'):
        dist_X = pairwise_distances(X,metric=metric)       
        # DEFAULT: set self-distances to inf, avoid ties / floatpoint errors
        # affects slicing of Q
        # slice Q as Q[:-1,:-1] to remove self neighbours        
        if self.set_self_dist_inf_ranks:
            np.fill_diagonal(dist_X, np.inf)
        return dist_X
    
    def _residual_variance(self, dist_X : np.ndarray, dist_X_reduced : np.ndarray, corr_method = 'pearson'):
        upper_tri = np.triu_indices(dist_X.shape[0], k=1)
        pd_x = dist_X[upper_tri]
        pd_x_r = dist_X_reduced[upper_tri]
        
        if corr_method=='pearson':
            return pearsonr(pd_x.flatten(), pd_x_r.flatten())[0]
        if corr_method=='spearman':
            return spearmanr(pd_x.flatten(), pd_x_r.flatten())[0]
        else:
            raise ValueError('corr_method not supported')
    
    # ------------- FUNCTIONS ON X_REDICED - end -------------    

    # ------------- RANKING AND CO-RANKING BASED FUNCTIONS - start -------------
    def _ranking_matrix(self, dist_X : np.ndarray, solver='nilrad') -> np.ndarray:        
        if solver=='nilrad':
            # Computing ranking_matrix using np.argsort twice on every row [Nilrad]            
            R = [np.argsort(np.argsort(row)) for row in dist_X]
            return np.array(R, dtype = 'uint')        
        elif solver=='scipy':
            R = rankdata(dist_X, axis=1, method="ordinal").astype(int) - 1
            return np.array(R, dtype = 'uint')
        else:            
            raise ValueError('not supported solver')
    
    def _cluster_dist_matrix(self, X : np.ndarray, cluster_labels : np.array,metric='euclidean'):
        
        assert X.shape[0] == cluster_labels.shape[0]        
        
        unique_clusters = np.unique(cluster_labels)
        num_cluster = len(unique_clusters)
        cluster_dist_X = np.zeros((num_cluster,num_cluster))

        for ix_i,i in enumerate(unique_clusters):
            sub_1 = X[cluster_labels == i,:]
            
            for ix_j,j in enumerate(unique_clusters):                
                sub_2 = X[cluster_labels == j,:]
                
                if i == j:
                    cluster_dist_X[ix_i,ix_j] = np.inf
                    
                else:
                    cluster_dist_X[ix_i,ix_j] = np.mean(pairwise_distances(sub_1,sub_2,metric=metric))  
                    
        return cluster_dist_X
    
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
    
    # could write functions for each k separately, but more efficient to loop it i think    
    def _get_k_lst_TC(self,N,k_mode,k_min=None,k_max=None):
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
        
            
    def _trustworthiness(self, Q : np.ndarray, k_mode = 'N/2',k_min=None,k_max=None) -> tuple[np.ndarray, np.ndarray]:
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

    def _continuity(self, Q : np.ndarray, k_mode = 'N/2',k_min=None,k_max=None) -> tuple[np.ndarray, np.ndarray]:
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

    def _lcmc(self, Q : np.ndarray,k_mode='all',k_min=None,k_max=None) -> tuple[np.ndarray, np.ndarray]:
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

    def _Q_NN(self, Q : np.ndarray, k_mode='all',k_min=None,k_max=None) -> tuple[np.ndarray, np.ndarray]:
        N = Q.shape[0]
        Q_sliced = self._drop_coranking_self_nb(Q)
        
        k_lst = self._get_k_lst_TC(N,k_mode,k_min,k_max)
        k_arr = np.array(k_lst)
        # vectorize using cumulative sums
        block = np.cumsum(np.cumsum(Q_sliced[:-1,:-1], 0), 1)         
        qnn = block[k_arr-1,k_arr-1]/(N*k_arr)       
        return qnn, k_arr
                      
    def _Q_local_global(self, Q : np.ndarray, lcmc = None,k_arr=None) -> tuple[np.ndarray, np.ndarray]:
        N = Q.shape[0]
        if lcmc is None:
            # find argmax of lcmc            
            lcmc, k_arr = self._lcmc(Q, k_mode='all')
            
        i_argmax = int(np.argmax(lcmc))
        k_argmax = int(k_arr[i_argmax])
        
        qnn = lcmc - k_arr/(1-N)
        q_local =  np.mean(qnn[:i_argmax + 1])
        q_global =  np.mean(qnn[i_argmax:])       

        return np.array([q_local,q_global]), k_argmax

    def _jaccard_sim_kNN(self,R : np.ndarray,R_reduced : np.ndarray,k : int):
        # output is a vector of size N
        nn_X = R < k
        nn_X_reduced = R_reduced < k
        nn_intersect = np.count_nonzero(nn_X & nn_X_reduced, axis=1)
        nn_union = np.count_nonzero(nn_X | nn_X_reduced, axis=1)
        jaccard_sim = nn_intersect / nn_union
        
        return jaccard_sim
    
    def _kNN_distances_arr(self, R, k, dist_X, dist_X_reduced):
        # output is two matrices of size N,k each
        
        #for each i, list of col indices j that are within kNN
        # nn_X_index = np.nonzero(R < k)[1].reshape(-1, k) # (unsorted)
        
        nn_X_index = np.argsort(R, axis=1)[:, :k] # (sorted)

        nn_X_dist = np.take_along_axis(dist_X,nn_X_index,axis=1)
        nn_X_reduced_dist = np.take_along_axis(dist_X_reduced,nn_X_index,axis=1)        
        # apply summary functions on these dists. of nearest neighbours
                
        return nn_X_dist, nn_X_reduced_dist       

    # ------------- RANKING AND CO-RANKING BASED FUNCTIONS - end -------------

# --------------------------

class DimReductionStatistics(DimReductionMethodsBase):
    name = "DimReductionStatistics"
    identifier = "DimReductionStatistics"    
    # labels = ["non-linear", "topology"]
    
    # stat , needs ranks or coranking, function outputs
    STATISTICS_DICT = {}
    
    STATISTICS_DICT["residual_variance"] = ("_stat_residual_variance",False,["rv"])
    STATISTICS_DICT["trustworthiness_auc"] = ("_stat_trustworthiness_auc",True,["T_auc","T_logWt_auc"])
    STATISTICS_DICT["continuity_auc"] = ("_stat_continuity_auc",True,["C_auc","C_logWt_auc"])
    STATISTICS_DICT["lcmc_auc"] = ("_stat_lcmc_auc",True,["lcmc_auc","lcmc_logWt_auc"])
    STATISTICS_DICT["q_local_global"] = ("_stat_q_local_global",True,["q_local","q_global"])

    # STATISTICS_DICT["jaccard_knn"] = ("_stat_jaccard_knn",True,[])
    # STATISTICS_DICT["knn_dist_minmax_ratio"] = ("_stat_knn_dist_minmax_ratio",True,[])
    # STATISTICS_DICT["knn_dist_var"] = ("_stat_knn_dist_var",True,[])
    
    def __init__(self,method : str = 'PCA',embedding_dim=2,random_state=None,
                 stats=None,metric='euclidean',
                 dim: Literal["n", "p"] = "p",nan_policy: Literal["raise", "row-omit", "col-omit"] = "raise"):
        # removed **kwargs for now
        super().__init__(method=method,embedding_dim=embedding_dim,random_state=random_state,
                         dim=dim,nan_policy=nan_policy)
        
        self.metric = metric
        
        self.stats = list(stats) if stats is not None else ["residual_variance"]
        
        unknown = set(self.stats) - set(self.STATISTICS_DICT.keys())
        if len(unknown) > 0:
            raise ValueError(f"unknown stats {unknown}\nsupported list- {self.STATISTICS_DICT.keys()}")

        self.get_ranks = any(self.STATISTICS_DICT[s][1] for s in self.stats)

    def _compute(self, data: np.ndarray):
        X_reduced = self._fit_method(data)
        
        dist_X = self._pairwise_distance_mtx(data,metric=self.metric)
        dist_X_red = self._pairwise_distance_mtx(X_reduced,metric=self.metric)
        
        dr_obj_dict = {"dist_X" : dist_X, "dist_X_reduced" : dist_X_red, 
                       "R" : None, "R_reduced" : None,                        
                       "Q" : None, 
                       "k_mode_TC" : 'N/2', 'k_mode_lcmc' : 'all',
                       "trustworthiness" : None, 
                       "continuity" : None,
                       "lcmc" : None}

        if self.get_ranks:
            dr_obj_dict["R"] = self._ranking_matrix(dist_X)
            dr_obj_dict["R_reduced"] = self._ranking_matrix(dist_X_red)
            dr_obj_dict["Q"] = self._coranking_matrix(dr_obj_dict["R"],dr_obj_dict["R_reduced"])
        
        
        out_vals = []
        for s in self.stats:
            stat_v = getattr(self,self.STATISTICS_DICT[s][0])(dr_obj_dict)
            out_vals.extend(stat_v)
            
        return np.array(out_vals).flatten()

            
    def _stat_residual_variance(self, dr_obj):
        return [self._residual_variance(dr_obj["dist_X"], dr_obj["dist_X_reduced"])]

    def _get_aucs(self,tup_v_k):
        val,k_arr = tup_v_k[0], tup_v_k[1]
        return [np.mean(val), np.sum(val / k_arr) / np.sum(1 / k_arr)]

    def _stat_trustworthiness_auc(self, dr_obj):
        if dr_obj['trustworthiness'] is None:
            T,k_arr = self._trustworthiness(dr_obj["Q"],k_mode=dr_obj['k_mode_TC'])
            dr_obj['trustworthiness'] = (T,k_arr)
    
        return self._get_aucs(dr_obj['trustworthiness'])
  
    def _stat_continuity_auc(self, dr_obj):
        if dr_obj['continuity'] is None:
            C,k_arr = self._continuity(dr_obj["Q"],k_mode=dr_obj['k_mode_TC'])
            dr_obj['continuity'] = (C,k_arr)
    
        return self._get_aucs(dr_obj['continuity'])

    def _stat_lcmc_auc(self, dr_obj):
        if dr_obj['lcmc'] is None:
            lcmc,k_arr = self._lcmc(dr_obj["Q"],k_mode=dr_obj['k_mode_lcmc'])
            dr_obj['lcmc'] = (lcmc,k_arr)

        return self._get_aucs(dr_obj['lcmc'])

    def _stat_q_local_global(self, dr_obj):
        if dr_obj['lcmc'] is None:
            lcmc,k_arr = self._lcmc(dr_obj["Q"],k_mode=dr_obj['k_mode_lcmc'])
            dr_obj['lcmc'] = (lcmc,k_arr)
        
        q_lg,lcmc_kmax = self._Q_local_global(dr_obj["Q"],dr_obj["lcmc"][0],dr_obj["lcmc"][1])
        dr_obj['lcmc_argmax'] = lcmc_kmax
        return list(q_lg)

    def _stat_jaccard_knn(self,dr_obj,k=2):
        self._jaccard_sim_kNN(dr_obj["R"],dr_obj["R_reduced"],k)
        pass

    def _stat_knn_dist_minmax_ratio(self):
        pass
    
    def _stat_knn_dist_var(self):
        pass
        
# ---------------------------------
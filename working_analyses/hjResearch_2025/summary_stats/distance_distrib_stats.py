"""
contains distance distribution statistics, reduced statistics on row-wise pairwise or centroids
"""

# interactive testing done in working_analyses/hj_research/[...]/distance_stats_testing.ipynb

from pyspoc import ReducedStatistic
import numpy as np
from scipy.stats import skew, kurtosis, entropy, trim_mean, ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist,pdist
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset
from pyspoc._tag import Tag, Type, Signature

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

class DistanceDistributionBase(ReducedStatistic, ABC):

    VALID_METRICS = [
        'canberra', 'chebyshev', 
        'cityblock', 'correlation', 'cosine', 
        'euclidean', 'mahalanobis', 
        'seuclidean', 'sqeuclidean','minkowski'
    ]

    VALID_POINTS = {'all', 'global_mean_centroid', 'global_median_centroid', 'global_trimmed_mean_centroid'}    

    def __init__(self, metric: str = "euclidean",minkowski_p=None):
        super().__init__()
        
        self.metric = metric
        self.minkowski_p = minkowski_p
        
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {', '.join(self.VALID_METRICS)}"
            )
        if metric == 'minkowski' and minkowski_p is None:
            raise ValueError("metric='minkowski' requires 'minkowski_p' to be passed")

        # self.point = point
    
    # def _compute(self, data):
    #     result = super()._compute(data)
    #     return result

    def _dist_from_point(self, X: np.ndarray, point) -> np.ndarray:

        if point=='all':
            return self._pairwise_dists(X)

        center = self._get_center_from_point(X,point)

        if self.metric.startswith('minkowski'):
            return cdist(XA=X,XB=center,metric=self.metric,p=self.minkowski_p)
        
        return cdist(XA=X,XB=center,metric=self.metric)

    # TODO: IMPLEMENT PAIRWISE DISTANCE FUNCTION
    def _pairwise_dists(self, X: np.ndarray) -> np.ndarray:

        if self.metric.startswith('minkowski'):
            metricval = self.metric.split(' ')[0]
            pval = float(self.metric.split(' ')[1])
            return pdist(X, metric = metricval,p=pval)

        return pdist(X, metric = self.metric)
    
    def _distance_histogram(self,dists, bins=10, range=None):
        # TODO: Q - should smooth??
        hist, _ = np.histogram(dists, bins=bins, range=range, density=False)
        hist = hist.astype(np.float64)
        hist += 1e-10  # smoothing to avoid zero bins
        return hist / np.sum(hist)

    def _distance_softmax_transform(self,distances):
        # TODO: distance softmax implement?
        pass

    def _get_center_from_point(self,X,point=None):

        if point is None:
            return np.zeros(shape=(1,X.shape[1]))
        
        elif point=='global_mean_centroid':
            return np.mean(X,axis=0).reshape((1,X.shape[1]))
        
        elif point=='global_median_centroid':
            return np.median(X,axis=0).reshape((1,X.shape[1]))

        elif point=='global_trimmed_mean_centroid':
            return trim_mean(X, 0.1, axis=0).reshape((1, X.shape[1]))
        
        # elif point=='global_geom_median_centroid': #TODO: requires https://github.com/daleroberts/hdmedians/tree/master
        #     return np.median(X,axis=0).reshape((1,X.shape[1]))
        
        else:
            return point
        
    # TODO: ADD SCALING FUNCTIONS IF NEEDED
    # TODO: UNIT TEST

class CentroidDistanceDistributionCompare(DistanceDistributionBase):
    _long_name = "CentroidDistanceDistributionCompare"
    _identifier = "centroid-distance-distrib"
    _type = Type.DESCRIPTIVE
    _signatures = (Signature.SYMMETRY,)
    _tags = (Tag.DISTANCE, Tag.NONPARAMETRIC)

    VALID_MODES = {
        'KL_BOTH' : "_mode_kl_both",
        'KS_STATISTIC' : "_mode_ks_statistic",
        'WASSERSTEIN_DIST' : "_mode_wasserstein"
    }

    def __init__(self,metric,point1 = 'global_mean_centroid',
                 point2 = 'global_median_centroid',minkowski_p=None,
                 modes=['KL_BOTH']):
        super().__init__(metric=metric,minkowski_p=minkowski_p)
        self.point1 = point1
        self.point2 = point2
        
        self.modes = [v.upper() for v in list(modes)] if modes is not None else ["KL_BOTH"]
        unknown = set(self.modes) - set(self.VALID_MODES.keys())
        if unknown:
            raise ValueError(f"unkonown {unknown}, supported- {list(self.VALID_MODES.keys())}")

    def _compute(self, data: np.ndarray) -> np.ndarray:
        """
        difference in distance distributions from two centroids
        choose between KL (both sides), KS test statistic, and Wasserstein distance 
        """
        
        dists1 = self._dist_from_point(data,self.point1).flatten()
        dists2 = self._dist_from_point(data, self.point2).flatten()
        
        out_vec = []
        for m in self.modes:
            stat_out = getattr(self,self.VALID_MODES[m])(dists1,dists2)
            # print('stat_out', stat_out)
            out_vec.extend(stat_out)
            
        # print('out_vec',out_vec)
        return np.array(out_vec).flatten()

    def _mode_kl_both(self, dists1, dists2):
        # global_bin
        edges = np.histogram_bin_edges(np.concatenate([dists1, dists2]), bins="fd")
        hist1 = self._distance_histogram(dists1,bins=edges)
        hist2 = self._distance_histogram(dists2,bins=edges)
        
        # compute KL
        kl_12 = entropy(hist1,hist2)
        kl_21 = entropy(hist2,hist1)
        
        return [kl_12,kl_21]

    def _mode_ks_statistic(self, dists1, dists2):
        return [ks_2samp(dists1, dists2).statistic]

    def _mode_wasserstein(self, dists1, dists2):
        return [wasserstein_distance(dists1, dists2)]
    
class GlobalCentroidDiffs(DistanceDistributionBase):
    def __init__(self,metric,point1 = 'global_mean_centroid',point2 = 'global_median_centroid',minkowski_p=None):
        super().__init__(metric=metric,minkowski_p=minkowski_p)
        self.point1 = point1
        self.point2 = point2

    def _compute(self, data: np.ndarray) -> Union[np.ndarray, float]:

        """
        return RMSE, and feature-wise max, min of difference bw 2 centroids     
        (intuition- max difference of mean and median across marginal distributions per feature)     
        """
        center1 = self._get_center_from_point(data,self.point1)
        center2 = self._get_center_from_point(data,self.point2)

        feat_wise_diff = center1 - center2
        rmse = np.sqrt(np.mean(feat_wise_diff ** 2))
        
        return np.array([rmse,np.max(np.abs(feat_wise_diff)), np.min(np.abs(feat_wise_diff))])
    
class DistanceDistributionBasicSummarize(DistanceDistributionBase):
    name = "Distance from Point - Peaks"
    identifier = "dist-point-peaks"

    summary_fns_dict = {
        "mean" : lambda dists : np.mean(dists),
        "median" : lambda dists : np.median(dists),
        "iqr" : lambda dists : np.quantile(dists,0.75) - np.quantile(dists,0.25),
        "std" : lambda dists : np.std(dists),
        "min" : lambda dists : np.min(dists),
        "max" : lambda dists : np.max(dists),
        "skew" : lambda dists : skew(dists),
        "kurtosis" : lambda dists : kurtosis(dists),
    }
        
    def __init__(self,metric, point, stats: list[str] = ["skew"],minkowski_p=None):
        super().__init__(metric=metric,minkowski_p=minkowski_p)
        self.point = point
        self.stats = stats


    def _compute(self, data: np.ndarray) -> np.ndarray:
        
        """
        Compute distance distribution, choose between:
            pairwise
            distribution from global mean centroid
            distribution from global median centroid
            origin
            or, provide custom np.ndarray
        
        Valid metrics: 
            'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
            'euclidean', 'mahalanobis', 'seuclidean', 'sqeuclidean'
        
        Summarize computed distance distribution using: 
            mean, median, IQR, std, min, max, skew, kurtosis        
        """
                
        
        dists = self._dist_from_point(data, self.point).flatten()

        if set(self.stats) <= (set(self.summary_fns_dict.keys())):        
            # summary.append(np.median(dists))
            summary = [self.summary_fns_dict[v](dists) for v in self.stats]          
            print()

        else:
            raise ValueError(f"Unsupported stats: {self.stats}")

        print(summary)
        
        return np.array(summary).flatten()

class DistanceDistributionModesSummarize(DistanceDistributionBase):
    name = "Distance from Point - Summary Stats"
    identifier = "dist-point-summ"

    def __init__(self,metric, point, type='kde',minkowski_p=None):
        super().__init__(metric=metric,minkowski_p=minkowski_p)

        self.point=point
        self.type = type

    def _compute(self, data: np.ndarray) -> np.ndarray:
        """
        Compute distance distribution, choose between:
            pairwise
            distribution from global mean centroid
            distribution from global median centroid
            origin
            or, provide custom np.ndarray
        
        Valid metrics: 
            'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 
            'euclidean', 'mahalanobis', 'seuclidean', 'sqeuclidean'
        
        Summarize computed distance distribution using: 
            number of peaks in the gaussian KDE using silverman method    
        """
            
        dists = self._dist_from_point(data, self.point)

        dists1 = dists.flatten()

        if self.type == 'kde':
            kde = gaussian_kde(dists1, bw_method='silverman')
            xs = np.linspace(min(dists1), max(dists1), 1000)
            ys = kde(xs)
            peaks, _ = find_peaks(ys)
            n_modes = len(peaks)

        # elif self.type == 'max_line_intersections':
        #     # TODO: implement
        #     n_modes = -1.0

        # TODO: USE INFLECTION POINTS?
        # d2y = np.gradient(dy, x)      # Second derivative

        else:
            raise ValueError()

        return np.array(n_modes)

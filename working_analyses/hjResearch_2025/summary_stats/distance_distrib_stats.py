"""
contains distance distribution statistics, reduced statistics on row-wise pairwise or centroids
"""

from pyspoc import ReducedStatistic
import numpy as np
from scipy.stats import skew, kurtosis, entropy, trim_mean
from scipy.spatial.distance import cdist,pdist
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

class DistanceDistributionBase(ReducedStatistic, ABC):

    VALID_METRICS = [
        'canberra', 'chebyshev', 
        'cityblock', 'correlation', 'cosine', 
        'euclidean', 'mahalanobis', 
        'seuclidean', 'sqeuclidean'
    ]

    VALID_POINTS = {'all', 'global_mean_centroid', 'global_median_centroid', 'global_trimmed_mean_centroid'}    

    def __init__(self, metric: str = "euclidean",minkowski_p=None):

        self.metric = metric
            
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {', '.join(self.VALID_METRICS)}"
            )
        if metric == 'minkowski' and minkowski_p is None:
            raise ValueError("metric='minkowski' requires 'minkowski_p' to be passed")

        # self.point = point
    
    def _compute(self, data):
        result = super()._compute(data)
        return result

    def _dist_from_point(self, X: np.ndarray, point) -> np.ndarray:

        if point=='all':
            return self._pairwise_dists(X)

        center = self._get_center_from_point(X,point)

        if self.metric.startswith('minkowski'):
            metricval = self.metric.split(' ')[0]
            pval = float(self.metric.split(' ')[1])
            return cdist(XA=X,XB=center,metric=metricval,p=pval)
        
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
    # TODO: ADD CACHING OF DISTRIBUTION SO THAT IT'S EASIER 
    # TODO: UNIT TEST


class KLCentroidDistances(DistanceDistributionBase):
    # TODO: move, inherit from distance distribution base
    name = "KLCentroidDistances"
    identifier = ""
    # labels = ["skew"]

    def __init__(self,metric,point1 = 'global_mean_centroid',point2 = 'global_median_centroid',minkowski_p=None):
        super().__init__(metric=metric,minkowski_p=minkowski_p)
        self.point1 = point1
        self.point2 = point2

    def _compute(self, data: np.ndarray) -> np.ndarray:
        """
        Return KL Divergences of distance distributions from two input points 
        (q- how would this behave in high dim data? might be very trivial bcz of curse of dimensionality?)
        """
        
        dists1 = self._dist_from_point(data,self.point1)
        dists2 = self._dist_from_point(data, self.point2)
        
        # global_bin
        global_bin = None
        hist1 = self._distance_histogram(dists1,bins=global_bin)
        hist2 = self._distance_histogram(dists2,bins=global_bin)

        # compute KL
        kl_mean_median = entropy(hist1,hist2)
        kl_median_mean = entropy(hist2,hist1)
        
        return np.asarray([kl_mean_median, kl_median_mean])
    
class GlobalCentroidDiffs(DistanceDistributionBase):
    def __init__(self,metric,point1 = 'global_mean_centroid',point2 = 'global_median_centroid',minkowski_p=None):
        super().__init__(metric=metric,minkowski_p=minkowski_p)
        self.point1 = point1
        self.point2 = point2

    @property
    def name(self) -> str:
        return "GlobalCentroidDiffs"

    @property
    def identifier(self) -> str:
        return "GlobalCentroidDiffs"

    @property
    def labels(self) -> list[str]:
        return ["skew"]

    def _compute(self, data: np.ndarray) -> Union[np.ndarray, float]:

        """
        For each feature of the input data matrix, compute the max absolute difference of input points         
        (intuition- max difference of mean and median across marginal distributions per feature)        
        """
        center1 = self._get_center_from_point(data,self.point1)
        center2 = self._get_center_from_point(data,self.point2)

        feat_wise_diff = center1 - center2                
        return np.array([np.max(np.abs(feat_wise_diff)), np.min(np.abs(feat_wise_diff))])

    
class DistanceDistributionBasicSummarize(DistanceDistributionBase):
    name = "Distance from Point - Peaks"
    identifier = "dist-point-peaks"
    labels = ["scalar", "distance"]

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
                
        
        dists = self._dist_from_point(data, self.point)

        if set(self.stats) <= (set(self.summary_fns_dict.keys())):        
            # summary.append(np.median(dists))
            summary = [self.summary_fns_dict[v](dists) for v in self.stats]            
            print()

        else:
            raise ValueError(f"Unsupported stats: {self.stats}")
                    
        return np.array(summary).flatten()

class DistanceDistributionModesSummarize(DistanceDistributionBase):
    name = "Distance from Point - Summary Stats"
    identifier = "dist-point-summ"
    labels = ["scalar", "distance"]

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

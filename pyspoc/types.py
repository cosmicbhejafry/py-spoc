from typing import Literal, TypeAlias



DistanceMetric : TypeAlias = Literal[
    "euclidean",
    "sqeuclidean",
    "nan_euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "seuclidean",
    "mahalanobis",

    "hamming",
    "canberra",
    "braycurtis",

    "jaccard",
    "matching",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",

    "cosine",
    "correlation",
    "yule"]

"""
    ## Supported Metrics
    
    For two vectors `x` and `y` of equal length, the following distance metrics
    are supported:

    ### Metrics intended for real vectors

    **euclidean** : `sqrt(sum((x - y)^2))`
    **sqeuclidean** : `sum((x - y)^2)`
    **nan_euclidean** : `sqrt(sum((x - y)^2))` (all `np.nan` removed)
    **manhattan** : `sum(|x - y|)`
    **chebyshev** : `max(|x - y|)`
    **minkowski** : `sum(w * |x - y|^p)^(1/p)` (`w` and `p` provided separately)
    **seuclidean** : `sqrt(sum((x - y)^2 / v))` (`v` is computed from the variances `x` and `y`)
    **mahalanobis** : `sqrt((x - y)' V^-1 (x - y))` (`V` is a supplied matrix)

    ### Metrics intended for integer vectors

    **hamming** : `count(x != y) / count(x)`
    **canberra** : `sum(|x - y| / (|x| + |y|))`
    **braycurtis** : `sum(|x - y|) / (sum(|x|) + sum(|y|))` (all `np.nan` removed)

    ### Metrics intended for boolean vectors

    **jaccard** : `count(x != y) / count(x or y)`
    **matching** : `count(x != y) / count(x)` (**hamming** distance for booleans)
    **dice** : `count(x != y) / (count(x and y) + count(x or y))`
    **rogerstanimoto** : `2 * count(x != y) / (count(x) + count(x != y))
    **sokalsneath** : `count(x != y) / (count(x != y) + count(x and y) / 2)`



    For full documentation on the distance metrics below, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
"""

SummaryType: TypeAlias = Literal[
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "stdev",
    "iqr",
    "range"]

PairwiseSymmetryType: TypeAlias = Literal[
    "exact", "negative", "reciprocal", "triangular"]

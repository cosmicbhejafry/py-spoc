from pyspoc import Reducer
import numpy as np
from typing import Union
from abc import ABC
from pyspoc.dataset import Dataset
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class PCAtSNEDiff():
    pass


# ari = adjusted_rand_score(labels_pca, labels_tsne)
# nmi = normalized_mutual_info_score(labels_pca, labels_tsne)

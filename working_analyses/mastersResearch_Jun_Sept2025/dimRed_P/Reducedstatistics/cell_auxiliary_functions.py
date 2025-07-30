import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.pipeline    import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold     import TSNE, Isomap
import umap

# dictionary that maps each reducer's name to its function
BASE = {
    'PCA'   : PCA,
    't-SNE'  : TSNE,
    'Isomap': Isomap,
    'UMAP'  : umap.UMAP,
}

def reducer_reader(name_str):
    """
    Turn "PCA 50D -> TSNE" or "TSNE" into the corresponding pipeline of funcitons.
    The outuput f can then be updated as f(ndims, **kw) where the kwargs are passed to the final stage.
    """
    stages = [s.strip() for s in name_str.split('->')]
    info = []
    for i, stage in enumerate(stages):
        m = re.match(r'^([A-Za-z-]+)(?:\s+(\d+)D)?$', stage)
        if not m or m.group(1) not in BASE:
            raise ValueError(f"Cannot parse stage '{stage}'")
        cls = BASE[m.group(1)]
        fixed = int(m.group(2)) if m.group(2) else None
        info.append((cls, fixed))

    def factory(ndims, **kw):
        steps = []
        for idx, (cls, fixed) in enumerate(info):
            # last stage uses `ndims` if not fixed
            n = fixed if fixed is not None else ndims
            params = {}
            params['n_components'] = n
            # only pass **kw to the final stage
            inst = cls(**(params if idx < len(info)-1 else {**params, **kw}))
            steps.append((f"{cls.__name__.lower()}{idx}", inst))
        return Pipeline(steps) if len(steps) > 1 else steps[0][1]

    return factory


def rankClus(embed, labels, met = 'l1'):
    """ Compute average pairwise distances between labeled groups in a latent space.
        Arguments:
            - embed (Nxd array): Coordinates of each sample in the latent space.
            - labels (N, array): Labels for each sample.
            - met (string): metric to use when computing distances.
        Returns:
            - d (number_of_labelsxnumber_of_labels array): A symmetric matrix where d[i, j] is the mean distance between all
                points in group i and all points in group j. Groups are ordered according to `np.unique(labels)`.
    """
    
    unique_labels = np.unique(labels)
    d = np.zeros((len(unique_labels),len(unique_labels)))

    for i in range(len(unique_labels)):
        avg_dists = [] # Here we will store each row of the final array d
        for j in unique_labels:
            # Now we compute the distance between cluster i and cluster j as the mean distance among their points
            sub_1 = embed[inLab == unique_labels[i],:]
            sub_2 = embed[inLab == j,:]
            avg_dists += [np.mean(pairwise_distances(sub_1,sub_2,metric=met).flatten().tolist())]

        d[i,:] = avg_dists
    
    return d

def get_knn(X, n_neighbors=10):
    """ Computes the nearest neighbours of every point in a dataset.
        Arguments:
            - X (Nxp array): Dataset for which we will compute the nearest neighbours.
            - n_neighbors (int): Number of neighbours we will consider.
        Returns:
            - dist (Nxn_neighbors array): Distances to the nearest neighbours for every point.
            - ind (Nxn_neighbors array): Indices of the nearest neighbours for every point.
    """
    # We use the euclidean distance since this is the default metric for determining neighbours in t-SNE and UMAP
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    return dist, idx


def max_min_ratio(distances):
    """ Computes the distance max/min ratio among nearest neighbours.
        Arguments:
            - distances (Nxn_neighbors array): Nearest neighbour distances for every point.
        Returns:
            - dmax / dmin (N, array): Distance ratio for every point.
    """
    dmax = distances.max(axis=1)
    dmin = distances.min(axis=1)
    return dmax / dmin

def max_min_stat(amb_dist, amb_inx, Z):
    """ Computes the distance max/min ratio for the ambient and latent space.
        Arguments:
            - amb_dist (Nxn_neighbors array): Nearest neighbour distances for the ambient space.
            - amb_inx (Nxn_neighbors array): Nearest neighbour indices for the ambient space.
            - Z (Nxp array): Reduced dataset.
        Returns:
            - amb_stat (N, array): Distance ratio for ambient space.
            - lat_stat (N, array): Distance ratio for latent space.
    """
    # Compute distance ratio in ambient space
    amb_stat = max_min_ratio(amb_dist)

    # Compute distance ratio in latent space, considering the same nearest neighbours that we found in the ambient space
    N = Z.shape[0]
    lat_dist = np.empty((N, k), dtype=float)
    for i in range(N):
        lat_dist[i] = np.linalg.norm(Z[i] - Z[amb_idx[i]], axis=1)
    latent_ratio = max_min_ratio(lat_dist)

    return amb_stat, lat_stat
  
def var_stat(amb_dist, amb_idx, Z):
    """
    Computes the distance variance for the ambient and latent space.
    Arguments:
     - amb_dist (N×n_neighbors array): Nearest neighbour distances for the ambient space.
     - amb_idx  (N×n_neighbors array): Nearest neighbour indices for the ambient space.
     - Z        (N×p array): Reduced dataset.
    Returns:
     - amb_stat (N, array): Distance variance for ambient space.
     - lat_stat (N, array): Distance variance for latent space.
    """
    # Compute variance in ambient space
    amb_stat = np.var(amb_dist, axis=1)

    # Compute distances in latent space, using same neighbours from ambient
    N = Z.shape[0]
    lat_dist = np.empty((N, k), dtype=float)
    for i in range(N):
        lat_dist[i] = np.linalg.norm(Z[i] - Z[amb_idx[i]], axis=1)

    # Compute variance in latent space
    lat_stat = np.var(lat_dist, axis=1)

    return amb_stat, lat_stat
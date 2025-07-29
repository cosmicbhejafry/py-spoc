import re
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
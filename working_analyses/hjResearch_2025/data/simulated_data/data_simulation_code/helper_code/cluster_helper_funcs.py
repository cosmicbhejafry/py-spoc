
import numpy as np

def get_cluster_sizes_power_law(decay_exp, n_clusters, n_samples, min_count=20):
    """
    get decaying cluster imbalance
    currently relies on manual tuning of the decay parameter
    default: apply a min count per cluster of 20
    """
    if n_samples < min_count * n_clusters:
        raise ValueError(f"n_samples={n_samples} cannot give {n_clusters} clusters "
                         f"at least {min_count} points each")
        
    w = 1 / np.arange(1, n_clusters + 1) ** decay_exp
    w /= w.sum()
    spare = n_samples - min_count * n_clusters
    counts = min_count + np.floor(w * spare).astype(int)
    counts[0] += n_samples - counts.sum()
    return counts.tolist()

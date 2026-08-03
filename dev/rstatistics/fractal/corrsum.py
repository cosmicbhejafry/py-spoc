# %%

n = data.shape[0]

for scale in scales:
    boxes = np.floor(data / scale).astype(int)
    n_uq = np.unique(boxes).shape[0]

    if n_uq / n < 0.5:
        sorted_boxes = 
        pairs = list()

        for i in range(n):
            datum = sorted_data[i]

            for j in range(i+1, n):
                adj_datum = sorted_data[j]
                
                if datum[0] < adj_datum[0] - scale or datum[0] > adj_datum[0] + scale:
                    break

                dist = np.linalg.norm(datum - adj_datum, ord=2)
                
                if dist < scale:
                    pairs.append((datum, adj_datum))

# %%
# Testing

import numpy as np

# def _assign_boxes(data: np.ndarray):

seed = 0
rng = np.random.default_rng(seed)
data = rng.random(size=(1000,10))
# %%

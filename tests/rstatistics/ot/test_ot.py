#%%
from sklearn.metrics.pairwise import (
    rbf_kernel,
    euclidean_distances
)
from functools import partial
import torch

from pyspoc._random.torch import make_torch_generator

gen = make_torch_generator(0)
x = torch.randn(1000,1, generator=gen)
y = torch.randn(1000,1, generator=gen)
#%%
from geomloss.ot import solve
blur = 0.5
gamma = 1 / (2 * blur ** 2)
#cost = partial(rbf_kernel, gamma=gamma)
cost = euclidean_distances
C_xy = torch.tensor(cost(x,y))
C_xx = torch.tensor(cost(x,y))
C_yy = torch.tensor(cost(x,y))
solution = solve(C=C_xy,  reg=0.001, max_iter=1000)
solution.value
#%%

#%%
from geomloss import SamplesLoss

loss = SamplesLoss(loss="sinkhorn")

L = loss(x,y)
#%%
import numpy as np
np.random.seed(0)
Pi = np.random.standard_normal(size=(10,10))
a = np.zeros(shape=Pi.shape[0], dtype=np.float64)
b = np.zeros(shape=Pi.shape[0], dtype=np.float64)

for i in range(10):
    a_i = 0
    b_i = 0

    for j in range(10):
        a_i += Pi[i,j]
    a[i] = a_i
    
    for j in range(10):
        q_ij = Pi[i,j] / a_i

        if q_ij > 0:
           b_i += q_ij * np.log(q_ij)

    b[i] = b_i

h = (a * b).sum()
h
#%%
a = Pi.sum(axis=1)
L = (Pi.T / a).T
b = np.sum(L * np.log(L), axis=1, where=L>0)
(b * a).sum()
#%%
a = Pi.sum(axis=1)
L = (Pi.T / a).T
np.sum(Pi * np.log(L), where=L>0)
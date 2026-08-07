#%%
import numpy as np
import matplotlib.pyplot as plt

from pyspoc.statistics.dimreduce.orthopcae import _estimator

#data = np.load("/home/gsc225/Projects/hcda/py-spoc/pyspoc/data/cml.npy")
data = np.load("/home/gsc225/Projects/hcda/py-spoc/pyspoc/data/forex.npy")
#data = np.random.normal(loc=3, scale=1, size=(10000,25))
#data = np.random.random(size=(10000,25))

# %%
#%%
# Assuming 'estimator' is your fitted OrthogonalPCAEstimator instance
model = opcae._model_

print("=== Encoder BatchNorm Metrics ===")
print("Running Mean:", model.encoder_bn.running_mean)
print("Running Var :", model.encoder_bn.running_var)
print("Learned Gamma (Scale):", model.encoder_bn.weight)
print("Learned Beta (Shift) :", model.encoder_bn.bias)

print("\n=== Decoder BatchNorm Metrics ===")
print("Running Mean:", model.decoder_bn.running_mean)
print("Running Var :", model.decoder_bn.running_var)
print("Learned Gamma (Scale):", model.decoder_bn.weight)
print("Learned Beta (Shift) :", model.decoder_bn.bias)
#%%

from pyspoc.rstatistics.dimreduce.orthopcae.rstatistics import (
    OrthogonalPCAEVarianceElbow, OrthogonalPCAEVarianceExplainedRatio)

orthopcae_ve = OrthogonalPCAEVarianceElbow(256, 5, train_steps=1000)
orthopcae_ve.compute(data)

# %%
orthopcae_var = OrthogonalPCAEVarianceExplainedRatio(256, 7, train_steps=1000)
results = orthopcae_var.compute(data)
results
# %%
plt.plot(results)
plt.show()
# %%
import pyspoc.rstatistics.dimreduce.orthopcae.rstatistics as rsf

est = _estimator.OrthogonalPCAEEstimator(256, 7, train_steps=1000)
est.fit(data)
# %%
rsf.extract_pcae_scree_data(est)

# %%
import torch
data_tensor = est._prepare_data(data)
sse_loss_fn = torch.nn.MSELoss(reduction="sum")
data_mean = data_tensor.mean(dim=0, keepdim=True).expand_as(data_tensor)
print(sse_loss_fn(data_mean, data_tensor).item())

normalized_tensor = est._normalize_data(data_tensor)

recon = est._model_(normalized_tensor)
print(sse_loss_fn(recon, data_tensor).item())

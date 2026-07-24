#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

from pyspoc.statistics.dimreduce.orthopcae import _estimator

#data = np.load("/home/gsc225/Projects/hcda/py-spoc/pyspoc/data/cml.npy")
data = np.load("/home/gsc225/Projects/hcda/py-spoc/pyspoc/data/forex.npy")
#data = np.random.normal(loc=3, scale=1, size=(10000,25))
#data = np.random.random(size=(10000,25))
#%%
test = _estimator.OrthogonalPCAEEstimator.get_or_create(data=data, batch_size=100, components=5)
test.compute
# %%
opcae = _estimator.OrthogonalPCAEEstimator(256,
                                          components=5,
                                          alpha=0.02,
                                          burn_in_steps_prop=0.2,
                                          train_steps=10000)
results = opcae.compute(data)
plt.plot(results["dimensions"], results["pseudo_variance_explained"])
results
#%%
# Assuming 'estimator' is your fitted OrthogonalPCAEstimator instance
model = opcae.model_

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
import torch
import numpy as np

def inspect_bottleneck_weights(estimator):
    """
    Extracts and profiles the learned projection weights of the bottleneck.
    Analogous to inspecting PCA component loading magnitudes.
    """
    # Put model into evaluation mode to freeze any runtime states
    estimator.model_.eval()
    
    # Extract the raw projection weights from the GPU/CPU
    # Shape: (max_bottleneck_dim, 64)
    W = estimator.model_.enc_project.weight.detach().cpu()
    
    max_bottleneck = W.shape[0]
    
    print("=== Bottleneck Component Profiler ===")
    print(f"{'Component':<12} | {'Weight L2 Norm (Importance)':<30} | {'Orthogonality Dot Product (vs Comp 0)':<35}")
    print("-" * 85)
    
    # Track metrics for library output or plotting
    component_norms = []
    
    for i in range(max_bottleneck):
        row = W[i]
        l2_norm = torch.norm(row, p=2).item()
        component_norms.append(l2_norm)
        
        # Calculate dot product against the very first component to check independence
        if i == 0:
            dot_vs_first = 1.0  # Perfect self-correlation
        else:
            dot_vs_first = torch.dot(W[0], row).item()
            
        print(f"Node {i:<8} | {l2_norm:<30.4f} | {dot_vs_first:<35.6f}")
        
    # Check overall matrix orthogonality matrix deviation (Frobenius norm)
    W_XT = torch.matmul(W, W.t())
    identity = torch.eye(max_bottleneck)
    frobenius_deviation = torch.norm(W_XT - identity, p='fro').item()
    
    print("-" * 85)
    print(f"Total Matrix Orthogonality Deviation (Frobenius): {frobenius_deviation:.4f}")
    
    return {
        "component_norms": component_norms,
        "orthogonality_matrix": W_XT.numpy()
    }

#%%
# Run the profiler
metrics = inspect_bottleneck_weights(opcae)
# %%

test = [1,2,3]
# %%
import numpy as np
data = np.load("/home/gsc225/Projects/hcda/py-spoc/pyspoc/data/forex.npy")
from pyspoc.rstatistics.dimreduce.orthopcae.rstatistics import (
    OrthogonalPCAEVarianceElbow,
    OrthogonalPCAEVarianceExplained)

opcaeve = OrthogonalPCAEVarianceExplained(
    batch_size=250,
    components=5,
    alpha=0.02,
    burn_in_steps_prop=0.2,
    train_steps=10000)

variance_explained = opcaeve.compute(data)
print(variance_explained)
#%%
opcaevee = OrthogonalPCAEVarianceElbow(
    batch_size=250,
    components=5,
    alpha=0.02,
    burn_in_steps_prop=0.2,
    train_steps=5000)

elbow = opcaevee.compute(data)
print(elbow)
# %%
test_dict = {"1": 2, "2": 3}
tuple(test_dict.values())
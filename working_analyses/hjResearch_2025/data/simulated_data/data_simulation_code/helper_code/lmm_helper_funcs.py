## note - right now each individual function uses a different rng. 
## it's probably fine to have 1 rng object in 'latent_metric_model_gp' and pass it across all the functions just for simplicity
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

# remove this function to reduce function call overhead
# def gp_kernel(Z, length_scale=1.0):
#     """compute the RBF kernel matrix for latent space Z"""
#     return 

def generate_gp_functions(Z, p, random_state,length_scale=1.0, kernel_sig=1.0):
    """
    generate p Gaussian process samples evaluated at Z
    """
    n = Z.shape[0]
    
    # rbf kernel
    K = kernel_sig * rbf_kernel(Z, Z, gamma=1.0 / (2 * length_scale**2))
    
    # Add small jitter for numerical stability
    K[np.diag_indices_from(K)] += 1e-6
    # K += 1e-6 * np.eye(n)

    rng = np.random.default_rng(seed=random_state)

    return rng.multivariate_normal(mean=np.zeros(n), cov=K, size=p).T

def generate_gp_functions_cholesky(Z, p, random_state,length_scale=1.0, kernel_sig=1.0):
    """
    generate p Gaussian process samples evaluated at Z using the cholesky decomp and then apply affine transform
    """
    n = Z.shape[0]
    
    # rbf kernel
    K = kernel_sig * rbf_kernel(Z, Z, gamma=1.0 / (2 * length_scale**2))
    
    # Add small jitter for numerical stability
    K[np.diag_indices_from(K)] += 1e-6
    # K += 1e-6 * np.eye(n)

    L = np.linalg.cholesky(K) # k shld be l @ l.t
    del(K)
    rng = np.random.default_rng(seed=random_state)
    return L @ rng.standard_normal((n, p))

def generate_noise(n, p, sigma,random_state):
    rng = np.random.default_rng(seed=random_state)
    return sigma * rng.standard_normal(size=(n, p))

def latent_metric_model_gp(n, p, Z, sigma, random_states, length_scale=1.0, sigma_f=1.0):
    """
    LMM sampling function

    takes as input the latent sampled space - just for ease in terms of parameters etc.

    supply dictionary of random states - to have diff ones for the latent sampling, gp sampling andf noise sampling
    todo- check logic of seeding ?? 
    """
    if 'gp' not in random_states.keys():
        raise ValueError(f'seed dict doesnt have gp key')
    if 'noise' not in random_states.keys():
        raise ValueError(f'seed dict doesnt have noise key')
    
    Y_gp = generate_gp_functions_cholesky(Z, p,random_states['gp'],length_scale=length_scale, kernel_sig=sigma_f)
    noise = generate_noise(n, p, sigma,random_states['noise'])
    Y = Y_gp + noise
    return Y
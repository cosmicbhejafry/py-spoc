from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import math

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._estimator import OrthogonalPCAEEstimator
    from ._module import OrthogonalPCAE


def train_adaptive_orthogonal_pcae(
        model: OrthogonalPCAE,
        data_loader: DataLoader,
        device: torch.device,
        current_epoch: int,
        epochs: int = 100,
        burn_in_epochs: int = 10,
        alpha: float = 0.1):
    """
    Trains the Orthogonal PCAE with an automatically scaling lambda.
    
    Args:
        model: The OrthogonalPCAE model instance.
        data_loader: PyTorch DataLoader containing the tabular dataset.
        epochs: Number of training epochs.
        alpha: Target ratio of orthogonality gradient force vs reconstruction force (default 10%).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_loss_fn = nn.MSELoss()
    
    # Track lambda over time for library diagnostics/logging
    lambda_history = []
    recon_loss_history = []
    ortho_loss_history = []
    optimal_epoch_loss = math.inf
    optimal_model = None
    optimal_epoch = None
    model.train()

    for epoch in range(current_epoch, current_epoch + epochs):

        mask = epoch > burn_in_epochs
                
        for batch in data_loader:
            
            batch_losses = []
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            
            # 1. Separate Forward Passes to isolate gradients
            # Pass 1: Reconstruction
            x_recon = model(batch_x, mask=mask)
            recon_loss = mse_loss_fn(x_recon, batch_x)
            
            # Pass 2: Orthogonality
            ortho_loss = model.get_orthogonality_loss()
            
            # 2. Compute Reconstruction Gradients
            # We retain the graph so we can backward through it again for total loss
            recon_loss.backward(retain_graph=True)

            # Extract gradient norm for the projection layer weight
            recon_grad = model.enc_project.weight.grad

            if recon_grad is None:
                raise ValueError(f"Reconstruction loss gradient is missing on epoch {epoch}.")

            recon_grad_norm = recon_grad.norm(p=2).item()

            # Store the reconstruction loss
            recon_loss_history.append((epoch, recon_loss.item()))
            
            # Clear gradients to isolate the next step
            optimizer.zero_grad()
            
            # 3. Compute Orthogonality Gradients
            ortho_loss.backward(retain_graph=True)
            
            # Extract gradient norm for the projection layer weight
            ortho_grad = model.enc_project.weight.grad

            if ortho_grad is None:
                raise ValueError(f"Orthogonal loss gradient is missing on epoch {epoch}.")

            ortho_grad_norm = ortho_grad.norm(p=2).item()

            # Store the orthogonality loss
            ortho_loss_history.append((epoch, ortho_loss.item()))
            
            # Clear gradients once more before final step
            optimizer.zero_grad()
            
            # 4. Calculate Adaptive Lambda
            if ortho_grad_norm > 1e-8:
                # Safe to divide: ortho_grad_norm is guaranteed to be positive here
                adaptive_lambda = alpha * recon_grad_norm / ortho_grad_norm
            else:
                # Stop penalising if we are already perfectly orthogonal
                adaptive_lambda = 0.0
                
            # Bound lambda to prevent extreme updates during early initialization noise
            adaptive_lambda = min(max(adaptive_lambda, 1e-5), 10.0)
            lambda_history.append(adaptive_lambda)
            
            # 5. Combined Final Backward Pass using calculated adaptive lambda
            total_loss = recon_loss + (adaptive_lambda * ortho_loss)
            total_loss.backward()
            batch_losses.append(total_loss.item())
            
            optimizer.step()

        epoch_loss = sum(batch_losses) / len(batch_losses)

        if epoch_loss < optimal_epoch_loss:
            optimal_epoch_loss = epoch_loss
            optimal_model = model
            optimal_epoch = epoch
            
    return {
        "model": model,
        "lambda_history": lambda_history,
        "recon_loss_history": recon_loss_history,
        "ortho_loss_history": ortho_loss_history,
        "optimal_model": optimal_model,
        "optimal_epoch": optimal_epoch,
        "optimal_epoch_loss": optimal_epoch_loss
    }


def extract_pcae_scree_data(
        model: OrthogonalPCAE,
        data_tensor: torch.Tensor) -> dict[str, Any]:
    """
    Computes a progressive reconstruction loss distribution.
    Analogous to PCA's cumulative variance explained curve.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    elbow_dim = 0
    dimensions = list(range(1, model.max_bottleneck + 1))
    loss_distribution = []
    
    with torch.no_grad():

        data_mean = data_tensor.mean(dim=0, keepdim=True).expand_as(data_tensor)
        baseline_mse = loss_fn(data_mean, data_tensor).item()
        
        # Calculate loss at each progressive dimension isolation step
        for d in dimensions:
            x_recon = model(data_tensor, active_dim=d)
            mse = loss_fn(x_recon, data_tensor).item()
            loss_distribution.append(mse)

    # Get baseline variance of the dataset (reconstructing with 0 dimensions)
    # This acts similarly to the total sum of squares (TSS).
    #baseline_mse = loss_distribution[0]
            
    # Calculate pseudo "Variance Explained" percentage
    # R^2 style: (Baseline Error - Bottleneck Error) / Baseline Error
    
    if np.isclose(baseline_mse, 0.0):
        variance_explained = np.zeros(
            len(loss_distribution),
            dtype=float,
        )
    else:
        variance_explained = np.array([
            max(0.0, baseline_mse - mse) / baseline_mse
            for mse in loss_distribution
        ])
        # Locate the optimal elbow point using standard knee-detection
        elbow_dim = _find_elbow_point(dimensions, loss_distribution)
    
    return {
        "dimensions": dimensions,
        "reconstruction_loss": loss_distribution,
        "pseudo_variance_explained": variance_explained,
        "optimal_bottleneck_dimension": int(elbow_dim),
        "baseline_loss": baseline_mse
    }

def _find_elbow_point(x: list[int], y: list[float]) -> int:
    """Standard maximum distance geometric elbow detector."""
    n_points = min(len(x), len(y))

    if n_points == 0:
        raise ValueError("Cannot find elbow point in empty data.")

    if n_points <= 2:
        return x[-1]

    coords = np.vstack((x, y)).T
    first_pt, last_pt = coords[0], coords[-1]
    line_vec = last_pt - first_pt
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    vec_from_first = coords - first_pt
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    
    distances = np.linalg.norm(vec_to_line, axis=1)
    return x[np.argmax(distances)]


def _inspect_bottleneck_weights(estimator: OrthogonalPCAEEstimator):
    """
    Extracts and profiles the learned projection weights of the bottleneck.
    Analogous to inspecting PCA component loading magnitudes.
    """
    # Put model into evaluation mode to freeze any runtime states

    if estimator.model_ is None:
        raise ValueError("The estimator has not been fitted yet. Call compute() first.")

    estimator.model_.eval()
    
    # Extract the raw projection weights from the GPU/CPU
    # Shape: (max_bottleneck_dim, 64)
    W = estimator.model_.enc_project.weight.detach().cpu()
    
    max_bottleneck = W.shape[0]
    
    print("=== Bottleneck Component Profiler ===")
    print(f"{'Component':<12} | "
          "{'Weight L2 Norm (Importance)':<30} | "
          "{'Orthogonality Dot Product (vs Comp 0)':<35}")
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

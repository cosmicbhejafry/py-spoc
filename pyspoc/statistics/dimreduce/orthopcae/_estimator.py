from __future__ import annotations

import torch
import numpy as np
import math

from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Literal, Optional, Any

from ._module import OrthogonalPCAE
from . import _func as f
from pyspoc._estimators.fitting import LazyFittedCachedEstimatorMixin
from pyspoc._argchecking import RuntimeTypeCheckedMixin


class OrthogonalPCAEEstimator(
    RuntimeTypeCheckedMixin,
    LazyFittedCachedEstimatorMixin[dict[str, Any]]):

    def __init__(
            self,
            batch_size: int,
            max_bottleneck_dim: Optional[int],
            train_steps: int = 10000,
            burn_in_steps_prop: float = 0.1,
            alpha: float = 0.1,
            compute_model_type: Literal["current", "optimal"] = "optimal",
            shuffle: bool = True,
            random_seed: int | None = None):

        """The user-facing API wrapper for the library."""

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.train_steps = train_steps
        self.burn_in = burn_in_steps_prop
        self.max_bottleneck_dim = max_bottleneck_dim
        self.compute_model_type = compute_model_type
        self.alpha = alpha

        self.current_epoch = 0
        self.lambda_history = []
        self.recon_loss_history = []
        self.ortho_loss_history = []

        # Set the DataLoader rng
        self._loader_rng = torch.Generator(device="cpu")

        if random_seed is not None:
            self._loader_rng.manual_seed(random_seed)
        else:
            self._loader_rng.seed()
        
        # Determine the best hardware accelerator available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_ = None
        self.train_epochs_ = None
        self.mean_ = None
        self.scale_ = None
        self.optimal_epoch_ = None
        self.optimal_model_ = None
        self.optimal_loss_ = math.inf


    def _fit_estimator(self, data: np.ndarray):
        n, p = data.shape

        self.batch_size = min(self.batch_size, n)
        self.train_epochs_ = self._get_train_epochs(n)
        self.burn_in_epochs_ = int(
            np.ceil(self.train_epochs_ * self.burn_in)
        )

        self.model_ = OrthogonalPCAE(
            p,
            self.max_bottleneck_dim,
            random_seed=self.random_seed,
        ).to(self.device)

        self._train(data, self.train_epochs_)


    def _normalize_data(
            self,
            X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X)
        else:
            X_tensor = X
        
        # Get normalization metrics for data
        if self.mean_ is None:
            self.mean_ = X_tensor.mean(dim=0)

        if self.scale_ is None:
            self.scale_ = X_tensor.std(dim=0)
            self.scale_[self.scale_ == 0.0] = 1.0

        # Normalize and prepare
        X_scaled = (X_tensor - self.mean_) / self.scale_

        return X_scaled
    

    def _prepare_data(
            self,
            X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Validates, casts, and wraps a raw NumPy array into a PyTorch DataLoader.
        
        Args:
            X: Input tabular dataset as a NumPy array.
            batch_size: Number of records per training step.
            shuffle: Whether to shuffle the data indices (True for training, False for evaluation).
        """
        # Ensure data is a standard floating point NumPy array
        if isinstance(X, torch.Tensor):
            
            if X.dtype == torch.float32:
                return X
            
            return X.to(torch.float32)
        
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        # Enforce float32 precision (PyTorch default neural network precision)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            
        # Ensure memory layout is C-contiguous to prevent internal PyTorch slicing warnings
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        # Convert to PyTorch Tensor
        tensor_X = torch.from_numpy(X)

        return tensor_X


    def _prepare_loader(self, tensor_X: torch.Tensor) -> DataLoader:
        
        # Construct the base dataset object
        dataset = TensorDataset(tensor_X)
        
        # Drop the last batch during training if it's too small
        # (prevents high-variance gradient steps)
        drop_last = self.shuffle and (len(dataset) > self.batch_size)
        
        # Build the structured high-performance data loader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=drop_last,
            generator=self._loader_rng,
            pin_memory=(self.device.type == "cuda") # Speeds up CPU-to-GPU data transfers
        )
        
        return loader

    
    def _get_model(self, model_type: Literal["current", "optimal"]) -> OrthogonalPCAE:
        if self.model_ is None:
            raise ValueError("Internal model has not yet been trained. Call compute() first.")
        
        if self.optimal_model_ is None:
            return self.model_
        else:
            return self.model_ if model_type == "current" else self.optimal_model_
    
        
    def _get_train_epochs(self, n: int) -> int:
        batch_count = np.ceil(n / self.batch_size)
        train_epochs = np.ceil(self.train_steps / batch_count)
        return int(train_epochs)
       
    
    def _train(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor],
            epochs: int):
        """Handles data loading and training execution cleanly outside the nn.Module."""

        tensor_X = self._prepare_data(X_tabular)
        tensor_X = self._normalize_data(tensor_X)
        data_loader = self._prepare_loader(tensor_X)
        burn_in_epochs = max(0, self.burn_in_epochs_ - self.current_epoch)
        
        # Execute the training algorithm
        trained_metrics = f.train_adaptive_orthogonal_pcae(
            self.model_,
            data_loader,
            self.device,
            current_epoch=self.current_epoch,
            epochs=epochs,
            burn_in_epochs=burn_in_epochs,
            alpha=self.alpha)

        # Grab training data
        self.lambda_history.extend(trained_metrics["lambda_history"])
        self.recon_loss_history.extend(trained_metrics["recon_loss_history"])
        self.ortho_loss_history.extend(trained_metrics["ortho_loss_history"])
        self.current_epoch += self.train_epochs_

        # Store optimal model and relevant metrics from training
        optimal_model = trained_metrics["optimal_model"]
        optimal_epoch = trained_metrics["optimal_epoch"]
        optimal_loss = trained_metrics["optimal_epoch_loss"]

        # Update overall optimal model if better
        if optimal_model is not None:
            if optimal_loss < self.optimal_loss_:
                self.optimal_model_ = optimal_model
                self.optimal_loss_ = optimal_loss
                self.optimal_epoch_ = optimal_epoch

    
    def _reconstruct(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor],
            model_type: Literal["current", "optimal"]) -> Union[np.ndarray, torch.Tensor]:
                       
        if self.model_ is None:
            raise ValueError("Internal model has not yet been trained. Call compute() first.")
        
        if self.optimal_model_ is None:
            model = self.model_
        else:
            model = self.model_ if model_type == "current" else self.optimal_model_

        model.eval()

        with torch.no_grad():
            tensor_X = self._prepare_data(X_tabular).to(self.device)
            tensor_X_recon = model(tensor_X).detach()

        if isinstance(X_tabular, torch.Tensor):
            return tensor_X_recon
        
        return tensor_X_recon.cpu().numpy()
        
        
    def _transform(self, X_tabular: Union[np.ndarray, torch.Tensor]):
        """Inference step: Extract the ordered bottleneck coordinates."""
        if self.model_ is None:
            raise ValueError("Internal model has not yet been trained. Call compute() first.")

        self.model_.eval()
        # Returns the compressed coordinates
        ...

        
    def _compute_fitted(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor]) -> dict[str, Any]:
        
        """Returns the scree-plot metadata, elbow, and R² scores."""
        
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The estimator must be fitted before calling compute_statistics.")
            
        # --- Apply the exact training constraints to the test data ---
        tensor_X = self._prepare_data(X_tabular)
        tensor_X = self._normalize_data(tensor_X)
        tensor_X = tensor_X.to(self.device)
        model = self._get_model(self.compute_model_type)
        return f.extract_pcae_scree_data(model, tensor_X)

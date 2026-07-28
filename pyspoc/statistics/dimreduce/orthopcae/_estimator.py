from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Literal

from ._model import OrthogonalPCAE
from pyspoc._estimators.fitting import LazyFittedCachedEstimatorMixin
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._random import RandomSeedMixin
from pyspoc._random.torch import make_torch_generator
from pyspoc.settings import settings


class OrthogonalPCAEEstimator(
    RuntimeTypeCheckedMixin,
    RandomSeedMixin,
    LazyFittedCachedEstimatorMixin):

    _freeze_random_seed = True

    def __init__(
            self,
            batch_size: int,
            max_bottleneck_dim: int,
            train_steps: int = 10000,
            burn_in_steps_prop: float = 0.1,
            alpha: float = 0.1,
            shuffle: bool = True,
            random_seed: int | None = None):

        """The user-facing API wrapper for the library."""

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._train_steps = train_steps
        self._burn_in = burn_in_steps_prop
        self._max_bottleneck_dim = max_bottleneck_dim
        self._alpha = alpha

        self._current_epoch = 0
        self._lambda_history = []
        self._recon_loss_history = []
        self._ortho_loss_history = []

        # Data ordering has an independent RNG stream from the model.
        self._loader_rng = make_torch_generator(self.random_seed)

        # Determine the best hardware accelerator available
        self._training_device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self._model_ = None
        self._train_epochs_ = None
        self._burn_in_epochs_ = None
        self._mean_ = None
        self._scale_ = None
        self._final_loss_ = None

    @property
    def batch_size(self) -> int:
        """Return the effective training batch size."""
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """Return whether the DataLoader shuffles training observations."""
        return self._shuffle

    @property
    def train_steps(self) -> int:
        """Return the requested number of optimizer steps."""
        return self._train_steps

    @property
    def burn_in_steps_prop(self) -> float:
        """Return the proportion of training reserved for burn-in."""
        return self._burn_in

    @property
    def max_bottleneck_dim(self) -> int | None:
        """Return the configured maximum bottleneck dimension."""
        return self._max_bottleneck_dim

    @property
    def alpha(self) -> float:
        """Return the target orthogonality-to-reconstruction gradient ratio."""
        return self._alpha

    @property
    def current_epoch(self) -> int:
        """Return the number of completed training epochs."""
        return self._current_epoch

    @property
    def lambda_history(self) -> tuple[float, ...]:
        """Return an immutable snapshot of adaptive lambda values."""
        return tuple(self._lambda_history)

    @property
    def recon_loss_history(self) -> tuple[tuple[int, float], ...]:
        """Return an immutable snapshot of reconstruction losses."""
        return tuple(self._recon_loss_history)

    @property
    def ortho_loss_history(self) -> tuple[tuple[int, float], ...]:
        """Return an immutable snapshot of orthogonality losses."""
        return tuple(self._ortho_loss_history)

    @property
    def loader_rng(self) -> torch.Generator:
        """Return an independent snapshot of the DataLoader generator state."""
        return self._loader_rng.clone_state()

    @property
    def training_device(self) -> torch.device:
        """Return the device used for model training."""
        return self._training_device

    @property
    def train_epochs(self) -> int | None:
        """Return the resolved number of training epochs, if available."""
        return self._train_epochs_

    @property
    def burn_in_epochs(self) -> int | None:
        """Return the resolved number of burn-in epochs, if available."""
        return self._burn_in_epochs_

    @property
    def mean(self) -> torch.Tensor | None:
        """Return a defensive copy of the fitted normalization mean."""
        return None if self._mean_ is None else self._mean_.clone()

    @property
    def scale(self) -> torch.Tensor | None:
        """Return a defensive copy of the fitted normalization scale."""
        return None if self._scale_ is None else self._scale_.clone()

    @property
    def final_loss(self) -> float | None:
        """Return the final observed training loss."""
        return self._final_loss_


    def _fit_estimator(self, data: np.ndarray):
        n, p = data.shape

        self._batch_size = min(self._batch_size, n)
        self._train_epochs_ = self._get_train_epochs(n)
        self._burn_in_epochs_ = int(
            np.ceil(self._train_epochs_ * self._burn_in)
        )

        self._model_ = OrthogonalPCAE(
            p,
            self._max_bottleneck_dim,
            random_seed=self.random_seed,
        ).to(self._training_device)

        self._train(data, self._train_epochs_)
        self._model_.eval()
        inference_device = self.get_inference_device()
        self._model_.to(inference_device)


    def _normalize_data(
            self,
            X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X)
        else:
            X_tensor = X

        # Get normalization metrics for data
        if self._mean_ is None:
            self._mean_ = X_tensor.mean(dim=0)

        if self._scale_ is None:
            self._scale_ = X_tensor.std(dim=0)
            self._scale_[self._scale_ == 0.0] = 1.0

        # Normalize and prepare
        X_scaled = (X_tensor - self._mean_) / self._scale_

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


    def get_inference_device(self) -> torch.device:
        match settings.current.torch_estimator_inference_device:

            case "training":
                return self._training_device

            case _:
                return torch.device("cpu")

    def _prepare_loader(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor],
            device: torch.device,
            shuffle: bool) -> DataLoader:

        tensor_X = self._prepare_data(X_tabular)

        tensor_X = self._normalize_data(tensor_X)

        tensor_X = tensor_X.cpu()

        # Construct the base dataset object
        dataset = TensorDataset(tensor_X)

        # Drop the last batch during training if it's too small
        # (prevents high-variance gradient steps)
        drop_last = shuffle and (len(dataset) > self._batch_size)

        # Build the structured high-performance data loader
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=self._loader_rng,
            pin_memory=(device.type == "cuda") # Speeds up CPU-to-GPU data transfers
        )

        return loader


    def _get_model(self) -> OrthogonalPCAE:
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return self._model_

    def _get_model_device(self) -> torch.device:
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return next(self._model_.parameters()).device

    def _get_train_epochs(self, n: int) -> int:
        if self._shuffle and n > self._batch_size:
            # Training drops a partial shuffled batch, so only complete
            # batches contribute optimizer steps.
            batch_count = n // self._batch_size
        else:
            batch_count = np.ceil(n / self._batch_size)

        train_epochs = np.ceil(self._train_steps / batch_count)
        return int(train_epochs)


    def _train(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor],
            epochs: int):
        """Handles data loading and training execution cleanly outside the nn.Module."""

        model_device = self._get_model_device()
        data_loader = self._prepare_loader(X_tabular, model_device, self._shuffle)

        if self._burn_in_epochs_ is None:
            raise RuntimeError("Burn-in epochs must be resolved before training.")

        # Execute the training algorithm
        trained_metrics = self._train_adaptive_orthogonal_pcae(
            self._get_model(),
            data_loader,
            self._training_device,
            epochs=epochs,
            alpha=self._alpha)

        # Grab training data
        self._lambda_history.extend(trained_metrics["lambda_history"])
        self._recon_loss_history.extend(trained_metrics["recon_loss_history"])
        self._ortho_loss_history.extend(trained_metrics["ortho_loss_history"])
        self._final_loss_ = trained_metrics["final_loss"]

    def _reconstruct(
            self,
            X_tabular: Union[np.ndarray, torch.Tensor],
            model_type: Literal["current", "optimal"]) -> Union[np.ndarray, torch.Tensor]:

        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        model = self._model_
        model_device = self._get_model_device()

        with torch.no_grad():
            tensor_X = self._prepare_data(X_tabular).to(model_device)
            tensor_X_recon = model(tensor_X).detach()

        if isinstance(X_tabular, torch.Tensor):
            return tensor_X_recon

        return tensor_X_recon.cpu().numpy()


    def _transform(self, X_tabular: Union[np.ndarray, torch.Tensor]):
        """Inference step: Extract the ordered bottleneck coordinates."""
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        self._model_.eval()
        # Returns the compressed coordinates
        ...


    def _train_adaptive_orthogonal_pcae(
            self,
            model: OrthogonalPCAE,
            data_loader: DataLoader,
            device: torch.device,
            epochs: int = 100,
            alpha: float = 0.1):
        """
        Trains the Orthogonal PCAE with an automatically scaling lambda.

        Args:
            model: The OrthogonalPCAE model instance.
            data_loader: PyTorch DataLoader containing the tabular dataset.
            epochs: Number of training epochs.
            alpha: Target ratio of orthogonality gradient force vs reconstruction force
                (default 10%).
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()

        # Track lambda over time for library diagnostics/logging
        lambda_history = []
        recon_loss_history = []
        ortho_loss_history = []

        model.train()
        epoch_list = list(range(self.current_epoch, self.current_epoch + epochs))

        for epoch in epoch_list:

            # Stochastic latent masking disabled during burn-in.
            mask = epoch >= self.burn_in_epochs
            batch_losses = []

            for batch in data_loader:
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

            self._current_epoch += 1
            epoch_loss = sum(batch_losses) / len(batch_losses)

        return {
            "model": model,
            "final_loss": epoch_loss,
            "lambda_history": lambda_history,
            "recon_loss_history": recon_loss_history,
            "ortho_loss_history": ortho_loss_history
        }


    def _inspect_bottleneck_weights(self):
        """
        Extracts and profiles the learned projection weights of the bottleneck.
        Analogous to inspecting PCA component loading magnitudes.
        """
        # Put model into evaluation mode to freeze any runtime states

        if self._model_ is None:
            raise ValueError("The estimator has not been fitted yet. Call compute() first.")

        self._model_.eval()

        # Extract the raw projection weights from the GPU/CPU
        # Shape: (max_bottleneck_dim, 64)
        W = self._model_.enc_project.weight.detach().cpu()

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

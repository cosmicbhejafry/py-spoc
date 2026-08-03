"""Cached, lazily fitted estimator for the Orthogonal PCA autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Any

from ._model import OrthogonalPCAE
from pyspoc._estimators.fitting import LazyFittedCachedEstimatorMixin
from pyspoc._argchecking import RuntimeTypeCheckedMixin
from pyspoc._random import RandomSeedMixin
from pyspoc._random.torch import make_torch_generator
from pyspoc.settings import settings


class OrthogonalPCAEEstimator(
    RuntimeTypeCheckedMixin, RandomSeedMixin, LazyFittedCachedEstimatorMixin
):
    """Manage the training and inference lifecycle of an OrthogonalPCAE model.

    The estimator owns normalization statistics, random generators, training
    diagnostics, and the private PyTorch model. Cache and fit synchronization
    are supplied by :class:`LazyFittedCachedEstimatorMixin`.

    Parameters
    ----------
    batch_size : int
        Number of observations per optimizer batch.
    max_bottleneck_dim : int
        Maximum number of ordered latent dimensions.
    train_steps : int, default=10000
        Target number of optimizer steps.
        Epoch count is a derived attribute. Subject to rounding conditional
        on ``shuffle``, the number of epochs is approximately ``train_steps``
        multiplied by ``batch_size`` divided by the number of data observations.
    burn_in_steps_prop : float, default=0.1
        Initial proportion of epochs trained without stochastic masking.
    alpha : float, default=0.1
        Target orthogonality-to-reconstruction gradient ratio in loss function.
    shuffle : bool, default=True
        Whether the training loader shuffles batches.
    random_seed : int or None, optional
        Per-estimator seed override. If ``None``, use the library setting.
    """

    # Freeze the resolved seed in the cache arguments. This ensures later
    # setting changes cannot silently alter the identity of this estimator.
    _freeze_random_seed = True

    def __init__(
        self,
        batch_size: int,
        max_bottleneck_dim: int,
        train_steps: int = 10000,
        burn_in_steps_prop: float = 0.1,
        alpha: float = 0.1,
        shuffle: bool = True,
        random_seed: int | None = None,
    ):
        """Initialize estimator configuration and unfitted state.

        Parameters
        ----------
        batch_size : int
            Number of observations per optimizer batch.
        max_bottleneck_dim : int
            Maximum number of ordered latent dimensions.
        train_steps : int, default=10000
            Target number of optimizer steps.
            Epoch count is a derived attribute. Subject to rounding conditional
            on ``shuffle``, the number of epochs is approximately ``train_steps``
            multiplied by ``batch_size`` divided by the number of data observations.
        burn_in_steps_prop : float, default=0.1
            Initial proportion of epochs trained without stochastic masking.
        alpha : float, default=0.1
            Target orthogonality-to-reconstruction gradient ratio in loss function.
        shuffle : bool, default=True
            Whether the training loader shuffles batches.
        random_seed : int or None, optional
            Per-estimator seed override. If ``None``, use the library setting.
        """

        # Configuration participates in cache equivalence and remains stable
        # after construction, except for an effective batch-size reduction
        # when the fitted dataset contains fewer observations.
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._train_steps = train_steps
        self._burn_in = burn_in_steps_prop
        self._max_bottleneck_dim = max_bottleneck_dim
        self._alpha = alpha

        # Retain diagnostics across explicit additional training runs.
        self._current_epoch = 0
        self._lambda_history = []
        self._recon_loss_history = []
        self._ortho_loss_history = []

        # Data ordering has an independent RNG stream from the model.
        self._loader_rng = make_torch_generator(self.random_seed)

        # Determine the best hardware accelerator available
        self._training_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Dataset-dependent fitted state is populated only by _fit_estimator().
        self._model_ = None
        self._train_epochs_ = None
        self._burn_in_epochs_ = None
        self._mean_ = None
        self._scale_ = None
        self._final_loss_ = None

    @property
    def batch_size(self) -> int:
        """Return the effective training batch size.

        Returns
        -------
        int
            Configured batch size, capped by the fitted observation count.
        """
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """Return whether training observations are shuffled.

        Returns
        -------
        bool
            ``True`` when the training loader randomizes observation order.
        """
        return self._shuffle

    @property
    def train_steps(self) -> int:
        """Return the requested number of optimizer steps.

        Returns
        -------
        int
            Approximate optimization-step target.
        """
        return self._train_steps

    @property
    def burn_in_steps_prop(self) -> float:
        """Return the proportion of training reserved for burn-in.

        Returns
        -------
        float
            Fraction of resolved training epochs without stochastic masking.
        """
        return self._burn_in

    @property
    def max_bottleneck_dim(self) -> int | None:
        """Return the configured maximum bottleneck dimension.

        Returns
        -------
        int or None
            Maximum latent dimension, or ``None`` if not configured.
        """
        return self._max_bottleneck_dim

    @property
    def alpha(self) -> float:
        """Return the target gradient ratio.

        Returns
        -------
        float
            Orthogonality-to-reconstruction gradient ratio.
        """
        return self._alpha

    @property
    def current_epoch(self) -> int:
        """Return the number of completed training epochs.

        Returns
        -------
        int
            Epoch index from which any additional training will continue.
        """
        return self._current_epoch

    @property
    def lambda_history(self) -> tuple[float, ...]:
        """Return an immutable snapshot of adaptive penalty values.

        Returns
        -------
        tuple of float
            Adaptive lambda value from each optimization batch.
        """
        return tuple(self._lambda_history)

    @property
    def recon_loss_history(self) -> tuple[tuple[int, float], ...]:
        """Return an immutable snapshot of reconstruction losses.

        Returns
        -------
        tuple of tuple of (int, float)
            Epoch and reconstruction-loss pairs for optimization batches.
        """
        return tuple(self._recon_loss_history)

    @property
    def ortho_loss_history(self) -> tuple[tuple[int, float], ...]:
        """Return an immutable snapshot of orthogonality losses.

        Returns
        -------
        tuple of tuple of (int, float)
            Epoch and orthogonality-loss pairs for optimization batches.
        """
        return tuple(self._ortho_loss_history)

    @property
    def loader_rng(self) -> torch.Generator:
        """Return a copy of the DataLoader random generator.

        Returns
        -------
        torch.Generator
            Independent generator with the current loader RNG state.
        """
        return self._loader_rng.clone_state()

    @property
    def training_device(self) -> torch.device:
        """Return the device used for model training.

        Returns
        -------
        torch.device
            Best accelerator selected during estimator construction, or CPU.
        """
        return self._training_device

    @property
    def train_epochs(self) -> int | None:
        """Return the dataset-specific, derived training epoch count.

        Returns
        -------
        int or None
            Resolved epoch count, or ``None`` before fitting.
        """
        return self._train_epochs_

    @property
    def burn_in_epochs(self) -> int | None:
        """Return the dataset-specific, derived burn-in epoch count.

        Returns
        -------
        int or None
            Resolved burn-in epochs, or ``None`` before fitting.
        """
        return self._burn_in_epochs_

    @property
    def mean(self) -> torch.Tensor | None:
        """Return a defensive copy of the fitted normalization mean.

        Returns
        -------
        torch.Tensor or None
            Per-feature mean, or ``None`` before fitting.
        """
        return None if self._mean_ is None else self._mean_.clone()

    @property
    def scale(self) -> torch.Tensor | None:
        """Return a defensive copy of the fitted normalization scale.

        Returns
        -------
        torch.Tensor or None
            Per-feature standard deviation, or ``None`` before fitting.
        """
        return None if self._scale_ is None else self._scale_.clone()

    @property
    def final_loss(self) -> float | None:
        """Return the final observed training loss.

        Returns
        -------
        float or None
            Mean loss from the final epoch, or ``None`` before fitting.
        """
        return self._final_loss_

    # Override the abstract fitting hook from LazyFittedCachedEstimatorMixin.
    # Its public fit() method owns thread-safe locking, data matching,
    # and the fitted-state transition around this method.
    def _fit_estimator(self, data: np.ndarray) -> None:
        """Construct and train the private model for a dataset.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional training data already accepted by the fitting
            lifecycle.

        Returns
        -------
        None
            The model, normalization state, and diagnostics are stored on this
            estimator.

        Notes
        -----
        This method overrides the abstract
        :meth:`LazyFittedCachedEstimatorMixin._fit_estimator` hook. Call
        :meth:`fit` rather than invoking this method directly so locking and
        cache invariants remain enforced.
        """
        n, p = data.shape

        # Convert the optimizer-step target into whole epochs for this dataset.
        self._batch_size = min(self._batch_size, n)
        self._train_epochs_ = self._get_train_epochs(n)
        self._burn_in_epochs_ = int(np.ceil(self._train_epochs_ * self._burn_in))

        # Model weight initialization uses a separate RNG stream from loader
        # shuffling, while both streams begin from the resolved estimator seed.
        self._model_ = OrthogonalPCAE(
            p,
            self._max_bottleneck_dim,
            random_seed=self.random_seed,
        ).to(self._training_device)

        self._train(data, self._train_epochs_)
        # Publish only an evaluation-mode model and move it to the configured
        # inference device to avoid retaining unnecessary accelerator memory.
        self._model_.eval()
        inference_device = self.get_inference_device()
        self._model_.to(inference_device)

    def _normalize_data(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Standardize data using normalization state from the first fit.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Observations to normalize.

        Returns
        -------
        torch.Tensor
            Standardized data with zero-variance feature scales replaced by
            one.

        Notes
        -----
        The mean and scale are initialized once and subsequently reused so
        inference remains consistent with training.
        """

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

    def _prepare_data(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert array-like tabular data to a contiguous float tensor.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Tabular observations.

        Returns
        -------
        torch.Tensor
            Float32 tensor suitable for the PyTorch model. Existing tensors
            remain on their current device; NumPy inputs produce CPU tensors.
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
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        # Convert to PyTorch Tensor
        tensor_X = torch.from_numpy(X)

        return tensor_X

    def get_inference_device(self) -> torch.device:
        """Resolve the post-training model device from current settings.

        Returns
        -------
        torch.device
            Training device when the setting is ``"training"``; otherwise CPU.
        """
        match settings.current.torch_estimator_inference_device:
            case "training":
                return self._training_device

            case _:
                return torch.device("cpu")

    def _prepare_loader(
        self, X_tabular: Union[np.ndarray, torch.Tensor], device: torch.device, shuffle: bool
    ) -> DataLoader:
        """Build a normalized data loader for training or evaluation.

        Parameters
        ----------
        X_tabular : numpy.ndarray or torch.Tensor
            Raw observations.
        device : torch.device
            Destination device used to determine whether pinned host memory is
            beneficial.
        shuffle : bool
            Whether to randomize observation order.

        Returns
        -------
        torch.utils.data.DataLoader
            Loader backed by normalized CPU tensors.
        """

        tensor_X = self._prepare_data(X_tabular)

        tensor_X = self._normalize_data(tensor_X)

        # DataLoader storage remains on CPU; individual batches are transferred
        # by the training loop.
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
            pin_memory=(device.type == "cuda"),  # Speeds up CPU-to-GPU data transfers
        )

        return loader

    def _get_model(self) -> OrthogonalPCAE:
        """Return the fitted private model.

        Returns
        -------
        OrthogonalPCAE
            Trained model owned by this estimator.

        Raises
        ------
        ValueError
            If fitting has not created a model.
        """
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return self._model_

    def _get_model_device(self) -> torch.device:
        """Return the device containing the fitted model.

        Returns
        -------
        torch.device
            Device of the model parameters.

        Raises
        ------
        ValueError
            If fitting has not created a model.
        """
        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        return next(self._model_.parameters()).device

    def _get_train_epochs(self, n: int) -> int:
        """Convert a target optimizer-step count into complete epochs.

        Parameters
        ----------
        n : int
            Number of training observations.

        Returns
        -------
        int
            Smallest whole epoch count meeting the requested step target.
        """
        if self._shuffle and n > self._batch_size:
            # Training drops a partial shuffled batch, so only complete
            # batches contribute optimizer steps.
            batch_count = n // self._batch_size
        else:
            batch_count = np.ceil(n / self._batch_size)

        train_epochs = np.ceil(self._train_steps / batch_count)
        return int(train_epochs)

    def _train(self, X_tabular: Union[np.ndarray, torch.Tensor], epochs: int):
        """Train the model and append diagnostics to estimator history.

        Parameters
        ----------
        X_tabular : numpy.ndarray or torch.Tensor
            Training observations.
        epochs : int
            Number of additional epochs to execute.

        Returns
        -------
        None
            Model parameters and estimator histories are updated in place.

        Raises
        ------
        RuntimeError
            If burn-in epochs have not been resolved by the fitting hook.
        """

        model_device = self._get_model_device()
        data_loader = self._prepare_loader(X_tabular, model_device, self._shuffle)

        if self._burn_in_epochs_ is None:
            raise RuntimeError("Burn-in epochs must be resolved before training.")

        # Execute the training algorithm
        trained_metrics = self._train_adaptive_orthogonal_pcae(
            self._get_model(), data_loader, self._training_device, epochs=epochs, alpha=self._alpha
        )

        # Grab training data
        self._lambda_history.extend(trained_metrics["lambda_history"])
        self._recon_loss_history.extend(trained_metrics["recon_loss_history"])
        self._ortho_loss_history.extend(trained_metrics["ortho_loss_history"])
        self._final_loss_ = trained_metrics["final_loss"]

    def _reconstruct(
        self, X_tabular: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Reconstruct observations with the fitted model.

        Parameters
        ----------
        X_tabular : numpy.ndarray or torch.Tensor
            Observations to reconstruct.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Reconstructed observations, preserving the input container type.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """

        if self._model_ is None:
            raise ValueError("Internal model has not yet been trained. Call fit() first.")

        model = self._model_
        model_device = self._get_model_device()

        # Disable autograd because reconstruction is an inference-only helper.
        with torch.no_grad():
            tensor_X = self._prepare_data(X_tabular).to(model_device)
            tensor_X_recon = model(tensor_X).detach()

        if isinstance(X_tabular, torch.Tensor):
            return tensor_X_recon

        return tensor_X_recon.cpu().numpy()

    def _transform(self, X_tabular: Union[np.ndarray, torch.Tensor]):
        """Extract ordered bottleneck coordinates from fitted observations.

        Parameters
        ----------
        X_tabular : numpy.ndarray or torch.Tensor
            Observations to transform.

        Returns
        -------
        None
            Latent-coordinate extraction is not yet implemented.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
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
        epochs: int,
        alpha: float,
    ) -> dict[str, Any]:
        """Train with an adaptive orthogonality penalty.

        Parameters
        ----------
        model : OrthogonalPCAE
            Model whose parameters are optimized.
        data_loader : torch.utils.data.DataLoader
            Normalized training batches.
        device : torch.device
            Device on which each optimization batch is processed.
        epochs : int
            Number of additional epochs.
        alpha : float
            Target ratio of orthogonality-gradient magnitude to
            reconstruction-gradient magnitude in loss function.

        Returns
        -------
        dict[str, Any]
            Trained model, final loss, and per-step lambda, reconstruction-loss,
            and orthogonality-loss histories.

        Raises
        ------
        ValueError
            If either loss fails to produce a projection-layer gradient.

        Notes
        -----
        Separate backward passes measure the two gradient magnitudes before a
        final combined backward pass updates the model.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()

        # Track lambda over time for library diagnostics/logging
        lambda_history = []
        recon_loss_history = []
        ortho_loss_history = []

        # Restore training behavior for batch normalization and stochastic
        # bottleneck masking.
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
            "ortho_loss_history": ortho_loss_history,
        }

    def _inspect_bottleneck_weights(self) -> dict[str, Any]:
        """Print and return diagnostics for learned bottleneck projections.

        Returns
        -------
        dict[str, Any]
            ``component_norms`` contains the L2 norm of every projection row;
            ``orthogonality_matrix`` contains the detached ``W @ W.T`` array.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.

        Notes
        -----
        This is a researcher-facing diagnostic helper analogous to inspecting
        PCA loading magnitudes. It prints a formatted report to standard output.
        """
        # Put model into evaluation mode to freeze any runtime states

        if self._model_ is None:
            raise ValueError("The estimator has not been fitted yet. Call fit() first.")

        self._model_.eval()

        # Extract the raw projection weights from the GPU/CPU
        # Shape: (max_bottleneck_dim, 64)
        W = self._model_.enc_project.weight.detach().cpu()

        max_bottleneck = W.shape[0]

        print("=== Bottleneck Component Profiler ===")
        print(
            f"{'Component':<12} | "
            "{'Weight L2 Norm (Importance)':<30} | "
            "{'Orthogonality Dot Product (vs Comp 0)':<35}"
        )
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
        frobenius_deviation = torch.norm(W_XT - identity, p="fro").item()

        print("-" * 85)
        print(f"Total Matrix Orthogonality Deviation (Frobenius): {frobenius_deviation:.4f}")

        return {"component_norms": component_norms, "orthogonality_matrix": W_XT.numpy()}

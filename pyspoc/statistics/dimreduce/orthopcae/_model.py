"""PyTorch model used by the Orthogonal PCA autoencoder estimator."""

import torch
import torch.nn as nn
import math

from pyspoc._random.torch import make_torch_generator


class OrthogonalPCAE(nn.Module):
    """Autoencoder with ordered, orthogonally regularized latent dimensions.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    bottleneck_dim : int
        Number of ordered latent dimensions.
    random_seed : int
        Seed for model-local weight initialization and mask sampling.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int, random_seed: int):
        """Initialize network layers and deterministic model-local state.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        bottleneck_dim : int
            Number of latent coordinates exposed by the bottleneck.
        random_seed : int
            Seed used by the model-local CPU random generator.
        """

        super().__init__()

        self.bottleneck = bottleneck_dim
        self.residual_dim = bottleneck_dim - 1
        self._rng = make_torch_generator(random_seed)

        self.mask_pool = self._get_refreshed_mask_pool()

        self.encoder_bn = nn.BatchNorm1d(64)
        self.decoder_bn = nn.BatchNorm1d(64)

        # Stream A: Purely Linear Mean/Intercept Tracker (1 Node)
        self.enc_mean = nn.Linear(input_dim, bottleneck_dim, bias=False)

        # Stream B: Non-Linear Orthogonal Bottleneck (Remaining Nodes)

        # Encoder Hidden Layer
        self.enc_hidden = nn.Linear(input_dim, 64)

        # The project layer must be saved explicitly to calculate orthogonality
        self.enc_project = nn.Linear(64, bottleneck_dim, bias=False)

        self.encoder = nn.Sequential(self.enc_hidden, self.encoder_bn, nn.Tanh(), self.enc_project)

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64), self.decoder_bn, nn.Tanh(), nn.Linear(64, input_dim)
        )

        self._reset_parameters()

    def get_device(self) -> torch.device:
        """Return the device containing the model parameters.

        Returns
        -------
        torch.device
            Device of the first registered parameter.
        """
        return next(self.parameters()).device

    def get_dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by the model parameters.

        Returns
        -------
        torch.dtype
            Data type of the first registered parameter.
        """
        return next(self.parameters()).dtype

    def _get_refreshed_mask_pool(self) -> torch.Tensor:
        """Generate one random permutation of all active dimensions.

        Returns
        -------
        torch.Tensor
            One-based bottleneck dimensions sampled without replacement.
        """
        return torch.randperm(self.bottleneck, generator=self._rng) + 1

    def _pop_mask_idx(self) -> int:
        """Remove and return the next dimension from the mask pool.

        Returns
        -------
        int
            One-based active bottleneck dimension.
        """
        idx = self.mask_pool[0].item()
        self.mask_pool = self.mask_pool[1:]
        return int(idx)

    def _reset_parameters(self):
        """Initialize all linear layers using the model-local generator.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        # Reimplement PyTorch's usual linear-layer initialization with an
        # explicit generator so model construction does not consume global RNG.
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue

            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=self._rng)

            if module.bias is not None:
                fan_in = module.weight.shape[1]
                bound = 1 / math.sqrt(fan_in)

                nn.init.uniform_(module.bias, -bound, bound, generator=self._rng)

    # Override torch.nn.Module.forward(), the framework-defined inference hook
    # invoked through ``model(...)``.
    def forward(self, x: torch.Tensor, *, active_dim=None, mask=False):
        """Reconstruct observations through an optionally masked bottleneck.

        Parameters
        ----------
        x : torch.Tensor
            Batch with shape ``(observations, input_features)``.
        active_dim : int or None, optional
            Number of leading latent dimensions to retain. If ``None``, use
            all dimensions unless stochastic training masking is requested.
        mask : bool, default=False
            Whether to sample an active dimension while the module is in
            training mode.

        Returns
        -------
        torch.Tensor
            Reconstructed batch with the same shape as ``x``.

        Notes
        -----
        This method overrides :meth:`torch.nn.Module.forward`.
        """
        # The linear mean stream preserves a dataset-level intercept while the
        # nonlinear stream learns ordered residual coordinates.
        x_mean = x.mean(dim=0, keepdim=True).expand_as(x)
        z_mean = self.enc_mean(x_mean)
        z_residual = self.encoder(x)

        if active_dim is None:
            if self.training and mask:
                # Stochastic step: pick a random bottleneck upper bound
                if self.mask_pool.numel() == 0:
                    self.mask_pool = self._get_refreshed_mask_pool()

                active_dim = self._pop_mask_idx()

            else:
                active_dim = self.bottleneck

        else:
            active_dim = min(max(1, active_dim), self.bottleneck)

        if active_dim < self.bottleneck:
            # Functional masking (safe for backpropagation)
            mask = torch.zeros_like(z_residual)
            mask[:, :active_dim] = 1.0
            z_residual = z_residual * mask

        z = z_residual + z_mean
        x_recon = self.decoder(z)
        return x_recon

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Measure deviation from orthonormal projection weights.

        Returns
        -------
        torch.Tensor
            Scalar Frobenius norm of ``W @ W.T - I``.

        Notes
        -----
        Penalizing this value encourages distinct bottleneck directions to be
        mutually orthogonal.
        """
        W = self.enc_project.weight  # Shape: (bottleneck_dim, 64)

        # Compute correlation/covariance structure of the weights
        # Product shape: (bottleneck_dim, bottleneck_dim)
        W_XT = torch.matmul(W, W.t())

        # Create an identity matrix of the same size
        identity = torch.eye(
            self.bottleneck,
            device=W.device,
            dtype=W.dtype,
        )

        # Penalise deviations from the identity matrix (Frobenius norm)
        return torch.norm(W_XT - identity, p="fro")

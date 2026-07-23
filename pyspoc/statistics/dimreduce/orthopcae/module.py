import torch
import torch.nn as nn
import numpy as np

class OrthogonalPCAE(nn.Module):
    def __init__(self, input_dim: int, max_bottleneck_dim: int):
        super().__init__()
                
        self.max_bottleneck = max_bottleneck_dim
        self.residual_dim = max_bottleneck_dim - 1

        self.mask_pool = self._get_refreshed_mask_pool()
        
        self.encoder_bn = nn.BatchNorm1d(64)
        self.decoder_bn = nn.BatchNorm1d(64)
        
        # Stream A: Purely Linear Mean/Intercept Tracker (1 Node)
        self.enc_mean = nn.Linear(input_dim, max_bottleneck_dim, bias=False)

        # Stream B: Non-Linear Orthogonal Bottleneck (Remaining Nodes)

        # Encoder Hidden Layer
        self.enc_hidden = nn.Linear(input_dim, 64)

        # The project layer must be saved explicitly to calculate orthogonality
        self.enc_project = nn.Linear(64, max_bottleneck_dim, bias=False)
               
        self.encoder = nn.Sequential(
            self.enc_hidden,
            self.encoder_bn,
            nn.Tanh(),
            self.enc_project
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(max_bottleneck_dim, 64),
            self.decoder_bn,
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

    def _get_refreshed_mask_pool(self) -> np.ndarray:
        return np.random.choice(self.max_bottleneck,
                                self.max_bottleneck,
                                replace=False) + 1


    def _pop_mask_idx(self) -> int:
        idx = self.mask_pool[0]
        self.mask_pool = self.mask_pool[1:]
        return idx


    def forward(self, x, *, active_dim=None, mask=False):
        
        x_mean = x.mean(axis=0, keepdim=True).expand_as(x)
        
        z_mean = self.enc_mean(x_mean)
        
        z_residual = self.encoder(x)
               
        if active_dim is None:
        
            if self.training and mask:
                # Stochastic step: pick a random bottleneck upper bound
                if self.mask_pool.size == 0:
                    self.mask_pool = self._get_refreshed_mask_pool()

                active_dim = self._pop_mask_idx()

            else:
                active_dim = self.max_bottleneck

        else:
            active_dim = min(max(1, active_dim), self.max_bottleneck)


        if active_dim < self.max_bottleneck:
            # Functional masking (safe for backpropagation)
            mask = torch.zeros_like(z_residual)
            mask[:, :active_dim] = 1.0
            z_residual = z_residual * mask
                    
        z = z_residual + z_mean
        x_recon = self.decoder(z)
        return x_recon
    

    def get_orthogonality_loss(self):
        """
        Penalises weight correlation in the projection layer.
        Forces the bottleneck directions to be mutually orthogonal.
        """
        W = self.enc_project.weight  # Shape: (max_bottleneck_dim, 64)
        
        # Compute correlation/covariance structure of the weights
        # Product shape: (max_bottleneck_dim, max_bottleneck_dim)
        W_XT = torch.matmul(W, W.t())
        
        # Create an identity matrix of the same size
        identity = torch.eye(self.max_bottleneck, device=W.device)
        
        # Penalise deviations from the identity matrix (Frobenius norm)
        return torch.norm(W_XT - identity, p='fro')

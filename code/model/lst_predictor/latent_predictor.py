"""
Latent-space LST predictor for diffusion guidance.
Predicts LST p95 from semantic/satellite VAE latents.
"""
###### import libraries ######
# Data Science/ML libraries
import torch
import torch.nn as nn


class LatentLSTPredictor(nn.Module):
    """
    Predicts LST statistics (p95) from VAE latent representations.
    
    Used for:
    - Phase 2: Soft guidance during diffusion sampling
    - Phase 3: Hard check after sampling (accept/reject)
    
    Architecture: CNN encoder → Global pool → MLP → Scalar
    """
    
    def __init__(
        self,
        z_channels: int = 3,
        latent_size: int = 64,
        hidden_dims: list = [64, 128, 256],
    ):
        """
        Args:
            z_channels: Number of latent channels (3 for semantic, 4 for satellite)
            latent_size: Spatial size of latent (64 for 2x VAE downsampling)
            hidden_dims: Hidden dimensions for CNN encoder
        """
        super().__init__()
        
        self.z_channels = z_channels
        self.latent_size = latent_size
        
        # CNN encoder with progressive downsampling
        encoder_layers = []
        in_ch = z_channels
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, h_dim), h_dim),
                nn.SiLU(),
            ])
            in_ch = h_dim
        
        # Global average pooling
        encoder_layers.append(nn.AdaptiveAvgPool2d(1))
        encoder_layers.append(nn.Flatten())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # MLP head for scalar prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1] (normalized LST)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent tensor [B, z_channels, H, W]
            
        Returns:
            LST prediction [B, 1] in [0, 1] range
        """
        features = self.encoder(z)
        return self.head(features)
    
    def predict_celsius(self, z: torch.Tensor, lst_max: float = 80.0) -> torch.Tensor:
        """
        Predict LST in Celsius (for interpretability).
        
        Args:
            z: Latent tensor [B, z_channels, H, W]
            lst_max: Maximum LST value used for normalization
            
        Returns:
            LST prediction [B, 1] in Celsius
        """
        return self.forward(z) * lst_max
"""
Latent-space Temperature predictor for diffusion guidance.
Predicts Temperature p95 from semantic/satellite VAE latents.
"""
###### import libraries ######
# Standard libraries
import os
from typing import Optional, Dict, Any

# Data Science/ML libraries
import torch
import torch.nn as nn


class LatentTemperaturePredictor(nn.Module):
    """
    Predicts Temperature statistics (p95) from VAE latent representations.
    
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
        dropout: float = 0.1,
    ):
        """
        Args:
            z_channels: Number of latent channels (3 for semantic, 4 for satellite)
            latent_size: Spatial size of latent (64 for 2x VAE downsampling)
            hidden_dims: Hidden dimensions for CNN encoder
            dropout: Dropout probability for regularization (0.0 to disable)
        """
        super().__init__()
        
        self.z_channels = z_channels
        self.latent_size = latent_size
        self.dropout = dropout
        
        # CNN encoder with progressive downsampling
        encoder_layers = []
        in_ch = z_channels
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, h_dim), h_dim),
                nn.SiLU(),
            ])
            # Add dropout after each conv block (except last)
            if dropout > 0 and i < len(hidden_dims) - 1:
                encoder_layers.append(nn.Dropout2d(dropout))
            in_ch = h_dim
        
        # Global average pooling
        encoder_layers.append(nn.AdaptiveAvgPool2d(1))
        encoder_layers.append(nn.Flatten())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # MLP head for scalar prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(dropout if dropout > 0 else 0.1),  # Always have some dropout in head
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1] (normalized Temperature)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent tensor [B, z_channels, H, W]
            
        Returns:
            Temperature prediction [B, 1] in [0, 1] range
        """
        features = self.encoder(z)
        return self.head(features)
    
    def predict_celsius(self, z: torch.Tensor, temp_max: float = 80.0) -> torch.Tensor:
        """
        Predict Temperature in Celsius (for interpretability).
        
        Args:
            z: Latent tensor [B, z_channels, H, W]
            temp_max: Maximum Temperature value used for normalization
            
        Returns:
            Temperature prediction [B, 1] in Celsius
        """
        return self.forward(z) * temp_max


def load_latent_temperature_predictor(
    config: Dict[str, Any],
    mode: str,
    device: torch.device,
    checkpoint_dir: Optional[str] = None
) -> Optional[LatentTemperaturePredictor]:
    """
    Load a trained LatentTemperaturePredictor from checkpoint.
    
    Args:
        config: Full config dict containing 'latent_temperature_predictor' section
        mode: Predictor mode ('semantic' or 'satellite')
        device: Device to load model on
        checkpoint_dir: Directory containing checkpoint (defaults to results/<task_name>)
        
    Returns:
        Loaded LatentTemperaturePredictor or None if checkpoint not found
    """
    predictor_config = config.get('latent_temperature_predictor', {})
    modes_config = predictor_config.get('modes', {})
    
    if mode not in modes_config:
        print(f"⚠ Mode '{mode}' not found in latent_temperature_predictor.modes config")
        return None
    
    mode_config = modes_config[mode]
    
    # Get architecture params
    z_channels = mode_config.get('z_channels', 3)
    latent_size = mode_config.get('latent_size', 64)
    hidden_dims = predictor_config.get('hidden_dims', [64, 128, 256])
    dropout = predictor_config.get('dropout', 0.1)
    
    # Get checkpoint path
    checkpoint_name = mode_config.get('checkpoint_name', f'latent_temperature_predictor_{mode}.pth')
    
    if checkpoint_dir is None:
        train_config = config.get('train_params', {})
        task_name = train_config.get('task_name', 'urban_inpainting')
        repo_dir =config.get('repo_dir', '')
        if not repo_dir:
            raise ValueError("repo_dir must be specified in config if checkpoint_dir is not provided")
        checkpoint_dir = os.path.join(repo_dir, 'results', task_name)
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Latent Temperature predictor checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = LatentTemperaturePredictor(
        z_channels=z_channels,
        latent_size=latent_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print info
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'unknown'))
    print(f"✓ Loaded latent Temperature predictor ({mode})")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Val loss: {val_loss:.6f}" if isinstance(val_loss, float) else f"  Epoch: {epoch}")
    print(f"  Architecture: z_channels={z_channels}, latent_size={latent_size}, hidden_dims={hidden_dims}")
    
    return model
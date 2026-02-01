"""
VAE-based perceptual loss for semantic diffusion training.

Uses trained semantic VAE encoder to extract features and compare
predicted vs ground truth in feature space (perceptual loss) rather than pixel space.

Benefits:
- Ideally captures semantic similarity (buildings, streets, vegetation patterns)
- More robust to small spatial offsets than pixel MSE
- Gets domain-specific knowledge from VAE training
- No domain mismatch (ImageNet features don't understand urban layouts)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class VAEPerceptualLoss(nn.Module):
    """
    Perceptual loss using VAE encoder features.
    
    Extracts intermediate features from a trained VAE encoder and computes
    weighted MSE loss between features of predicted and target images.
    
    Args:
        vae: Trained VAE model with encoder
        feature_layers: List of encoder layer indices to extract features from
                       (e.g., [0, 1, 2] for first 3 downsampling layers)
        feature_weights: Per-layer weights for feature loss
                        (e.g., [0.5, 1.0, 2.0] - early→deep semantic emphasis)
        normalize_features: If True, L2-normalize features before computing loss
    """
    
    def __init__(
        self,
        vae: nn.Module,
        feature_layers: List[int] = [0, 1, 2],
        feature_weights: Optional[List[float]] = None,
        normalize_features: bool = False,
    ):
        super().__init__()
        
        # Store frozen VAE encoder for feature extraction
        self.vae = vae
        self.vae.eval()  # Always in eval mode
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.normalize_features = normalize_features
        
        # Default: equal weights, or user-specified
        if feature_weights is None:
            self.feature_weights = [1.0] * len(feature_layers)
        else:
            assert len(feature_weights) == len(feature_layers), \
                f"feature_weights length {len(feature_weights)} != feature_layers length {len(feature_layers)}"
            self.feature_weights = feature_weights
        
        # Normalize weights to sum to 1
        total = sum(self.feature_weights)
        self.feature_weights = [w / total for w in self.feature_weights]
        
        # Storage for intermediate features
        self.features = {}
        self.hooks = []
        
        # Register forward hooks to capture intermediate features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on VAE encoder layers to capture features."""
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                self.features[layer_idx] = output
            return hook
        
        # Support custom VAE architecture with encoder_layers (ModuleList)
        if hasattr(self.vae, 'encoder_layers'):
            encoder_layers = self.vae.encoder_layers
            for layer_idx in self.feature_layers:
                if layer_idx < len(encoder_layers):
                    handle = encoder_layers[layer_idx].register_forward_hook(
                        hook_fn(layer_idx)
                    )
                    self.hooks.append(handle)
                else:
                    raise ValueError(
                        f"Layer index {layer_idx} out of range for encoder_layers "
                        f"(has {len(encoder_layers)} layers)"
                    )
        else:
            raise AttributeError(
                f"VAE must have 'encoder_layers' attribute for feature extraction. "
                f"Found attributes: {[attr for attr in dir(self.vae) if not attr.startswith('_')]}"
            )
    
    def extract_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate features from VAE encoder.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Dictionary mapping layer_idx -> feature tensor
        """
        self.features = {}  # Clear previous features
        
        with torch.no_grad():
            _ = self.vae.encode(x)  # Forward through encoder to trigger hooks
        
        return self.features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_per_layer: bool = False
    ) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target ground truth images [B, C, H, W]
            return_per_layer: If True, also return per-layer losses
        
        Returns:
            Weighted perceptual loss scalar (or dict if return_per_layer=True)
        """
        
        # Extract features from both pred and target
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        # Compute weighted MSE loss across feature layers
        total_loss = 0.0
        per_layer_losses = {}
        
        for layer_idx, weight in zip(self.feature_layers, self.feature_weights):
            pred_feat = pred_features[layer_idx]
            target_feat = target_features[layer_idx]
            
            # Optional: L2-normalize features (useful for stability)
            if self.normalize_features:
                pred_feat = nn.functional.normalize(pred_feat, dim=1)
                target_feat = nn.functional.normalize(target_feat, dim=1)
            
            # MSE loss between features
            layer_loss = nn.functional.mse_loss(pred_feat, target_feat)
            per_layer_losses[layer_idx] = layer_loss.item()
            
            # Weighted accumulation
            total_loss += weight * layer_loss
        
        if return_per_layer:
            return total_loss, per_layer_losses
        else:
            return total_loss
    
    def __del__(self):
        """Remove hooks on deletion to prevent memory leaks."""
        for handle in self.hooks:
            handle.remove()


def create_perceptual_loss(
    vae: nn.Module,
    config: dict
) -> Optional[VAEPerceptualLoss]:
    """
    Factory function to create perceptual loss from config.
    
    Args:
        vae: Trained VAE model
        config: Config dict with keys:
                - use_perceptual: bool
                - perceptual_layers: List[int]
                - perceptual_feature_weights: Optional[List[float]]
    
    Returns:
        VAEPerceptualLoss instance or None if disabled
    """
    
    if not config.get('use_perceptual', False):
        return None
    
    feature_layers = config.get('perceptual_layers', [0, 1, 2])
    feature_weights = config.get('perceptual_feature_weights', None)
    
    return VAEPerceptualLoss(
        vae=vae,
        feature_layers=feature_layers,
        feature_weights=feature_weights,
        normalize_features=False  # Can add to config if needed
    )

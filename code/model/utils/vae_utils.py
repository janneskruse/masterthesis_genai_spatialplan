"""
Utility functions for VAE training and inference.
"""

import os
import torch
from torchvision.utils import save_image, make_grid
from typing import Dict, List, Optional


def save_vae_reconstruction_samples(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    layer_names: List[str],
    layers_registry: Dict,
    save_dir: str,
    step: int,
    n_samples: int = 8,
    save_rgb_composite: bool = True
) -> None:
    """
    Save VAE reconstruction samples with layer-aware visualization.
    
    This function handles visualization for different layer types:
    - Binary layers: Applies sigmoid to logits for visualization
    - RGB layers: Normalizes from [-1, 1] to [0, 1]
    - Continuous layers: Applies adaptive normalization (height-based or min-max)
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstructed tensor [B, C, H, W] (logits for binary, values for continuous)
        layer_names: List of layer names corresponding to channels
        layers_registry: Global layers configuration dict
        save_dir: Directory to save visualization images
        step: Training step number for filename
        n_samples: Number of samples to visualize (default: 8)
        save_rgb_composite: Whether to save RGB composite if RGB layers present (default: True)
        
    Example:
        >>> save_vae_reconstruction_samples(
        ...     input_tensor=input_tensor,
        ...     recon_tensor=recon,
        ...     layer_names=['buildings', 'streets', 'lst'],
        ...     layers_registry=config['layers'],
        ...     save_dir='./samples',
        ...     step=1000
        ... )
    """
    
    n_samples = min(n_samples, input_tensor.shape[0])
    
    # Visualize each layer separately based on layer registry
    vis_grids = []
    
    for ch_idx, layer_name in enumerate(layer_names):
        input_ch = input_tensor[:n_samples, ch_idx:ch_idx+1, :, :]
        recon_ch = recon_tensor[:n_samples, ch_idx:ch_idx+1, :, :]
        
        # Get layer properties from registry
        layer_info = layers_registry.get(layer_name, {})
        layer_type = layer_info.get('type', 'continuous')
        
        # Determine visualization method based on layer type
        if layer_type == 'binary':
            # Binary channels: input is 0/1, recon is logits
            input_vis = torch.clamp(input_ch, 0, 1)
            recon_vis = torch.sigmoid(recon_ch)  # Apply sigmoid to logits
        elif layer_type == 'rgb':
            # RGB channels: normalize from [-1, 1] to [0, 1]
            input_vis = torch.clamp(input_ch, -1., 1.)
            input_vis = (input_vis + 1) / 2
            recon_vis = torch.clamp(recon_ch, -1., 1.)
            recon_vis = (recon_vis + 1) / 2
        else:
            # Continuous channels: normalize to [0, 1] range for visualization
            # Use channel-specific normalization based on data range
            if 'height' in layer_name.lower():
                # Height channels: normalize by max expected height
                max_height = 100.0
                input_vis = torch.clamp(input_ch / max_height, 0, 1)
                recon_vis = torch.clamp(recon_ch / max_height, 0, 1)
            else:
                # Other continuous: min-max normalization
                input_vis = (input_ch - input_ch.min()) / (input_ch.max() - input_ch.min() + 1e-8)
                recon_vis = (recon_ch - recon_ch.min()) / (recon_ch.max() - recon_ch.min() + 1e-8)
        
        # Create comparison for this layer
        comparison_ch = torch.cat([input_vis, recon_vis], dim=0)
        grid_ch = make_grid(comparison_ch, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
        vis_grids.append(grid_ch)
    
    # Save each layer separately
    for ch_idx, layer_name in enumerate(layer_names):
        save_path = os.path.join(save_dir, f'recon_step_{step}_{layer_name.replace(":", "_")}.png')
        save_image(vis_grids[ch_idx], save_path)
    
    # Also save RGB composite if RGB layers are present
    if save_rgb_composite:
        rgb_indices = [i for i, name in enumerate(layer_names) if 'rgb' in name.lower()]
        if len(rgb_indices) >= 3:
            # Take first 3 RGB channels
            rgb_input = input_tensor[:n_samples, rgb_indices[0]:rgb_indices[2]+1, :, :]
            rgb_recon = recon_tensor[:n_samples, rgb_indices[0]:rgb_indices[2]+1, :, :]
            
            rgb_input = torch.clamp(rgb_input, -1., 1.)
            rgb_input = (rgb_input + 1) / 2
            rgb_recon = torch.clamp(rgb_recon, -1., 1.)
            rgb_recon = (rgb_recon + 1) / 2
            
            comparison_rgb = torch.cat([rgb_input, rgb_recon], dim=0)
            grid_rgb = make_grid(comparison_rgb, nrow=n_samples, padding=2, pad_value=1.0)
            
            save_path = os.path.join(save_dir, f'recon_step_{step}_RGB_composite.png')
            save_image(grid_rgb, save_path)

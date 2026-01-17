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
    
    This function handles visualization for different layer types using
    normalization settings from the layer registry:
    - Binary layers: Applies sigmoid to logits for visualization
    - RGB/continuous layers: Uses layer-specific normalization config
      (percentile, minmax, clip, custom)
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstructed tensor [B, C, H, W] (logits for binary, values for continuous)
        layer_names: List of layer names corresponding to channels
        layers_registry: Global layers configuration dict with normalization settings
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
        normalize_method = layer_info.get('normalize', 'minmax')
        
        # Determine visualization method based on layer type
        if layer_type == 'binary':
            # Binary channels: input is 0/1, recon is logits
            input_vis = torch.clamp(input_ch, 0, 1)
            recon_vis = torch.sigmoid(recon_ch)  # Apply sigmoid to logits
            
        else:
            # Continuous/RGB channels: use layer-specific normalization config
            if normalize_method == 'percentile':
                # Percentile normalization (for RGB): data is already in [0, 1]
                input_vis = torch.clamp(input_ch, 0, 1)
                recon_vis = torch.clamp(recon_ch, 0, 1)
                
            elif normalize_method == 'clip':
                # Clip normalization: normalize using clip_range from config
                clip_range = layer_info.get('clip_range', [-1, 1])
                min_val, max_val = clip_range
                # Normalize to [0, 1] for visualization
                input_vis = (torch.clamp(input_ch, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
                recon_vis = (torch.clamp(recon_ch, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
                
            elif normalize_method == 'custom':
                # Custom normalization: use normalize_params from config
                normalize_params = layer_info.get('normalize_params', {'min': 0, 'max': 100})
                min_val = normalize_params.get('min', 0)
                max_val = normalize_params.get('max', 100)
                # Normalize to [0, 1] for visualization
                input_vis = torch.clamp((input_ch - min_val) / (max_val - min_val + 1e-8), 0, 1)
                recon_vis = torch.clamp((recon_ch - min_val) / (max_val - min_val + 1e-8), 0, 1)
                
            elif normalize_method == 'minmax':
                # Min-max normalization: use actual data range
                # Check if mask_layer is specified (e.g., buildings_heights masked by buildings)
                mask_layer = layer_info.get('mask_layer', None)
                if mask_layer:
                    # For masked layers, normalize only non-zero regions
                    input_nonzero = input_ch[input_ch > 0]
                    recon_nonzero = recon_ch[recon_ch > 0]
                    if len(input_nonzero) > 0:
                        input_min, input_max = input_nonzero.min(), input_nonzero.max()
                        input_vis = (input_ch - input_min) / (input_max - input_min + 1e-8)
                        input_vis = torch.clamp(input_vis, 0, 1)
                    else:
                        input_vis = input_ch
                    
                    if len(recon_nonzero) > 0:
                        recon_min, recon_max = recon_nonzero.min(), recon_nonzero.max()
                        recon_vis = (recon_ch - recon_min) / (recon_max - recon_min + 1e-8)
                        recon_vis = torch.clamp(recon_vis, 0, 1)
                    else:
                        recon_vis = recon_ch
                else:
                    # Regular min-max normalization
                    input_vis = (input_ch - input_ch.min()) / (input_ch.max() - input_ch.min() + 1e-8)
                    recon_vis = (recon_ch - recon_ch.min()) / (recon_ch.max() - recon_ch.min() + 1e-8)
            else:
                # Fallback: min-max normalization
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
            
            # Get RGB layer normalization config
            rgb_layer_name = [name for name in layer_names if 'rgb' in name.lower()][0]
            rgb_layer_info = layers_registry.get(rgb_layer_name, {})
            rgb_normalize = rgb_layer_info.get('normalize', 'percentile')
            
            # Normalize RGB composite based on layer config
            if rgb_normalize == 'percentile':
                # Already in [0, 1]
                rgb_input = torch.clamp(rgb_input, 0, 1)
                rgb_recon = torch.clamp(rgb_recon, 0, 1)
            elif rgb_normalize == 'clip':
                # Use clip_range
                clip_range = rgb_layer_info.get('clip_range', [-1, 1])
                min_val, max_val = clip_range
                rgb_input = (torch.clamp(rgb_input, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
                rgb_recon = (torch.clamp(rgb_recon, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
            else:
                # Default: assume [-1, 1] normalization (legacy support)
                rgb_input = torch.clamp(rgb_input, -1., 1.)
                rgb_input = (rgb_input + 1) / 2
                rgb_recon = torch.clamp(rgb_recon, -1., 1.)
                rgb_recon = (rgb_recon + 1) / 2
            
            comparison_rgb = torch.cat([rgb_input, rgb_recon], dim=0)
            grid_rgb = make_grid(comparison_rgb, nrow=n_samples, padding=2, pad_value=1.0)
            
            save_path = os.path.join(save_dir, f'recon_step_{step}_RGB_composite.png')
            save_image(grid_rgb, save_path)

"""
Utility functions for visualizing and saving sample outputs during training/inference.
Shared between VAE reconstruction and diffusion sampling.
"""
###### import libraries ######
# Standard libraries
import os
from typing import Dict, List, Optional, Tuple

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

# Local imports
from model.utils.colors import get_colormap_for_layer, apply_colormap_to_tensor


def normalize_channel_for_visualization(
    channel: torch.Tensor,
    layer_info: Dict,
    is_reconstruction: bool = False
) -> torch.Tensor:
    """
    Normalize a single channel for visualization based on layer type.
    
    Args:
        channel: Channel tensor [B, 1, H, W]
        layer_info: Layer configuration dict from registry
        is_reconstruction: If True, channel is VAE reconstruction (may be logits for binary)
        
    Returns:
        Normalized channel [B, 1, H, W] in [0, 1] range
    """
    layer_type = layer_info.get('type', 'continuous')
    
    if layer_type == 'binary':
        # Binary channels
        if is_reconstruction:
            # Reconstruction is logits - apply sigmoid
            channel_vis = torch.sigmoid(channel)
        else:
            # Input is already 0/1
            channel_vis = torch.clamp(channel, 0, 1)
    else:
        # Continuous/RGB channels - check for masking
        mask_layer = layer_info.get('mask_layer', None)
        
        if mask_layer:
            # For masked layers (e.g., buildings_heights), normalize only non-zero regions
            nonzero_mask = channel > 0
            if nonzero_mask.any():
                ch_min = channel[nonzero_mask].min()
                ch_max = channel[nonzero_mask].max()
                
                if ch_max > ch_min:
                    channel_vis = (channel - ch_min) / (ch_max - ch_min + 1e-8)
                    channel_vis = torch.clamp(channel_vis, 0, 1)
                    # Zero out regions that were originally zero
                    channel_vis = channel_vis * nonzero_mask.float()
                else:
                    channel_vis = channel
            else:
                channel_vis = channel
        else:
            # Regular min-max normalization across the entire channel
            ch_min = channel.min()
            ch_max = channel.max()
            
            if ch_max > ch_min:
                channel_vis = (channel - ch_min) / (ch_max - ch_min + 1e-8)
            else:
                # Constant channel - keep as is
                channel_vis = channel
    
    return channel_vis


def save_channel_visualization(
    channel: torch.Tensor,
    save_path: str,
    n_samples: int = 8,
    normalize: bool = False,
    apply_colormap: bool = False,
    colormap_name: Optional[str] = None
) -> None:
    """
    Save a single channel as a grid visualization.
    
    Args:
        channel: Channel tensor [B, 1, H, W] (already normalized to [0, 1])
        save_path: Path to save image
        n_samples: Number of samples to show in grid
        normalize: Whether to apply additional normalization (usually False if pre-normalized)
        apply_colormap: Whether to apply colormap
        colormap_name: Name of colormap to use (only if apply_colormap=True)
    """
    n_samples = min(n_samples, channel.shape[0])
    channel_subset = channel[:n_samples]
    
    # Apply colormap if requested
    if apply_colormap and colormap_name:
        channel_subset = apply_colormap_to_tensor(channel_subset, colormap_name)
    
    # Create grid and save
    grid = make_grid(channel_subset, nrow=n_samples, normalize=normalize, padding=2, pad_value=1.0)
    save_image(grid, save_path)


def save_comparison_visualization(
    input_channel: torch.Tensor,
    recon_channel: torch.Tensor,
    save_path: str,
    n_samples: int = 8,
    normalize: bool = False,
    apply_colormap: bool = False,
    colormap_name: Optional[str] = None
) -> None:
    """
    Save input vs reconstruction comparison for a single channel.
    
    Args:
        input_channel: Input channel [B, 1, H, W] (normalized to [0, 1])
        recon_channel: Reconstruction channel [B, 1, H, W] (normalized to [0, 1])
        save_path: Path to save image
        n_samples: Number of samples to show in grid
        normalize: Whether to apply additional normalization
        apply_colormap: Whether to apply colormap
        colormap_name: Name of colormap to use
    """
    n_samples = min(n_samples, input_channel.shape[0])
    
    # Concatenate input and reconstruction vertically (2*n_samples rows)
    comparison = torch.cat([input_channel[:n_samples], recon_channel[:n_samples]], dim=0)
    
    # Apply colormap if requested
    if apply_colormap and colormap_name:
        comparison = apply_colormap_to_tensor(comparison, colormap_name)
    
    # Create grid (n_samples columns, 2 rows: input on top, recon on bottom)
    grid = make_grid(comparison, nrow=n_samples, normalize=normalize, padding=2, pad_value=1.0)
    save_image(grid, save_path)


def save_layerwise_samples(
    tensor: torch.Tensor,
    layer_names: List[str],
    layers_registry: Dict,
    save_dir: str,
    filename_prefix: str,
    n_samples: int = 8,
    is_reconstruction: bool = False,
    use_colormaps: bool = True,
    mask: Optional[torch.Tensor] = None
) -> None:
    """
    Save each layer of a multi-channel tensor as separate visualizations.
    
    This is the main function for diffusion sampling visualization.
    
    Args:
        tensor: Multi-channel tensor [B, C, H, W]
        layer_names: List of layer names corresponding to channels
        layers_registry: Layer configuration registry
        save_dir: Directory to save images
        filename_prefix: Prefix for filenames (e.g., 'sample_step_1000')
        n_samples: Number of samples to visualize
        is_reconstruction: If True, applies sigmoid to binary logits
        use_colormaps: Whether to apply colormaps to continuous layers
        mask: Optional mask tensor [B, 1, H, W] to overlay red border
    """
    
    n_samples = min(n_samples, tensor.shape[0])
    
    for ch_idx, layer_name in enumerate(layer_names):
        if ch_idx >= tensor.shape[1]:
            break
        
        channel = tensor[:, ch_idx:ch_idx+1, :, :]
        layer_info = layers_registry.get(layer_name, {})
        
        # Normalize channel for visualization
        channel_vis = normalize_channel_for_visualization(
            channel, layer_info, is_reconstruction
        )
        
        # Determine if we should apply colormap
        layer_type = layer_info.get('type', 'continuous')
        apply_cmap = (
            use_colormaps and 
            layer_type != 'binary' and 
            'rgb' not in layer_name.lower()
        )
        
        # Get colormap name if applicable
        cmap_name = None
        if apply_cmap:
            cmap_name = get_colormap_for_layer(layer_name)
        
        # Apply colormap if needed before mask overlay
        if apply_cmap and cmap_name:
            channel_vis = apply_colormap_to_tensor(channel_vis, cmap_name)
            apply_cmap = False  # Already applied
            cmap_name = None
        
        # Overlay red mask border if mask is provided
        if mask is not None:
            # Upsample mask to match channel resolution
            mask_upsampled = F.interpolate(
                mask[:n_samples],
                size=(channel_vis.shape[2], channel_vis.shape[3]),
                mode='nearest'
            )
            
            # Convert grayscale to RGB for red border overlay
            if channel_vis.shape[1] == 1:
                channel_vis = channel_vis.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            
            # Compute mask boundary (edge detection)
            mask_tensor = mask_upsampled.float()  # [B, 1, H, W]
            
            # Create erosion kernel (3x3 all ones) on same device as mask
            kernel = torch.ones(1, 1, 3, 3, device=mask_tensor.device)
            
            # Erode the mask (shrink it inward)
            mask_eroded = F.conv2d(mask_tensor, kernel, padding=1)
            mask_eroded = (mask_eroded == 9).float()  # Only keep pixels where all 9 neighbors were 1
            
            # Boundary = original mask - eroded mask (pixels on the edge)
            mask_boundary = mask_tensor - mask_eroded
            mask_boundary = (mask_boundary > 0).float()
            
            # Ensure mask_boundary is on same device as channel_vis
            mask_boundary = mask_boundary.to(channel_vis.device)
            
            # Apply red border (set R=1, G=0, B=0 where boundary)
            channel_vis[:, 0:1, :, :] = torch.where(mask_boundary > 0, torch.ones_like(channel_vis[:, 0:1, :, :]), channel_vis[:, 0:1, :, :])
            channel_vis[:, 1:2, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(channel_vis[:, 1:2, :, :]), channel_vis[:, 1:2, :, :])
            channel_vis[:, 2:3, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(channel_vis[:, 2:3, :, :]), channel_vis[:, 2:3, :, :])
        
        # Save this layer
        save_path = os.path.join(save_dir, f'{filename_prefix}_{layer_name}.png')
        save_channel_visualization(
            channel_vis,
            save_path,
            n_samples=n_samples,
            normalize=False,  # Already normalized
            apply_colormap=apply_cmap,
            colormap_name=cmap_name
        )


def save_layerwise_comparisons(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    channel_names: List[str],
    layer_names: List[str],
    layers_registry: Dict,
    save_dir: str,
    filename_prefix: str,
    n_samples: int = 8,
    use_colormaps: bool = True
) -> None:
    """
    Save input vs reconstruction comparisons for each layer.
    
    This is the main function for VAE reconstruction visualization.
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstruction tensor [B, C, H, W] (logits for binary)
        channel_names: List of channel names (e.g., ['rgb:red', 'buildings'])
        layer_names: List of layer names for each channel
        layers_registry: Layer configuration registry
        save_dir: Directory to save images
        filename_prefix: Prefix for filenames (e.g., 'recon_step_1000')
        n_samples: Number of samples to visualize
        use_colormaps: Whether to apply colormaps to continuous layers
    """
    n_samples = min(n_samples, input_tensor.shape[0])
    
    for ch_idx, (channel_name, layer_name) in enumerate(zip(channel_names, layer_names)):
        if ch_idx >= input_tensor.shape[1]:
            break
        
        input_ch = input_tensor[:, ch_idx:ch_idx+1, :, :]
        recon_ch = recon_tensor[:, ch_idx:ch_idx+1, :, :]
        
        layer_info = layers_registry.get(layer_name, {})
        
        # Normalize both input and reconstruction
        input_vis = normalize_channel_for_visualization(input_ch, layer_info, is_reconstruction=False)
        recon_vis = normalize_channel_for_visualization(recon_ch, layer_info, is_reconstruction=True)
        
        # Determine if we should apply colormap
        layer_type = layer_info.get('type', 'continuous')
        apply_cmap = (
            use_colormaps and 
            layer_type != 'binary' and 
            'rgb' not in layer_name.lower()
        )
        
        # Get colormap name if applicable
        cmap_name = None
        if apply_cmap:
            cmap_name = get_colormap_for_layer(layer_name)
        
        # Save comparison
        save_path = os.path.join(save_dir, f'{filename_prefix}_{channel_name.replace(":", "_")}.png')
        save_comparison_visualization(
            input_vis,
            recon_vis,
            save_path,
            n_samples=n_samples,
            normalize=False,  # Already normalized
            apply_colormap=apply_cmap,
            colormap_name=cmap_name
        )


def save_rgb_composite(
    tensor: torch.Tensor,
    layer_names: List[str],
    save_path: str,
    n_samples: int = 8,
    normalize_per_channel: bool = True
) -> None:
    """
    Save RGB composite visualization if tensor contains RGB channels.
    
    Args:
        tensor: Multi-channel tensor [B, C, H, W]
        layer_names: List of layer names
        save_path: Path to save RGB composite
        n_samples: Number of samples to visualize
        normalize_per_channel: Whether to normalize each RGB channel independently
    """
    # Find RGB channel indices
    rgb_indices = [i for i, layer_name in enumerate(layer_names) if 'rgb' in layer_name.lower()]
    
    if len(rgb_indices) < 3:
        return  # Not enough RGB channels
    
    try:
        n_samples = min(n_samples, tensor.shape[0])
        
        # Extract RGB channels
        rgb_tensor = tensor[:n_samples, rgb_indices[:3], :, :]
        
        if normalize_per_channel:
            # Normalize each channel independently for better color balance
            rgb_normalized = []
            for ch_idx in range(3):
                ch = rgb_tensor[:, ch_idx:ch_idx+1, :, :]
                ch_min = ch.min()
                ch_max = ch.max()
                
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
                else:
                    ch_norm = ch
                
                rgb_normalized.append(ch_norm)
            
            rgb_tensor = torch.cat(rgb_normalized, dim=1)
        
        # Create and save grid
        grid = make_grid(rgb_tensor, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
        save_image(grid, save_path)
        
    except Exception as e:
        print(f"[WARNING] Failed to save RGB composite: {e}")


def save_rgb_comparison(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    layer_names: List[str],
    save_path: str,
    n_samples: int = 8,
    normalize_per_channel: bool = True
) -> None:
    """
    Save input vs reconstruction RGB composite comparison.
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstruction tensor [B, C, H, W]
        layer_names: List of layer names
        save_path: Path to save RGB comparison
        n_samples: Number of samples to visualize
        normalize_per_channel: Whether to normalize each RGB channel independently
    """
    # Find RGB channel indices
    rgb_indices = [i for i, layer_name in enumerate(layer_names) if 'rgb' in layer_name.lower()]
    
    if len(rgb_indices) < 3:
        return  # Not enough RGB channels
    
    try:
        n_samples = min(n_samples, input_tensor.shape[0])
        
        # Extract RGB channels from both input and reconstruction
        rgb_input = input_tensor[:n_samples, rgb_indices[:3], :, :]
        rgb_recon = recon_tensor[:n_samples, rgb_indices[:3], :, :]
        
        if normalize_per_channel:
            # Normalize input
            rgb_input_normalized = []
            for ch_idx in range(3):
                ch = rgb_input[:, ch_idx:ch_idx+1, :, :]
                ch_min = ch.min()
                ch_max = ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
                else:
                    ch_norm = ch
                rgb_input_normalized.append(ch_norm)
            
            # Normalize reconstruction
            rgb_recon_normalized = []
            for ch_idx in range(3):
                ch = rgb_recon[:, ch_idx:ch_idx+1, :, :]
                ch_min = ch.min()
                ch_max = ch.max()
                if ch_max > ch_min:
                    ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
                else:
                    ch_norm = ch
                rgb_recon_normalized.append(ch_norm)
            
            rgb_input = torch.cat(rgb_input_normalized, dim=1)
            rgb_recon = torch.cat(rgb_recon_normalized, dim=1)
        
        # Concatenate input and reconstruction for comparison
        comparison = torch.cat([rgb_input, rgb_recon], dim=0)
        
        # Create and save grid
        grid = make_grid(comparison, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
        save_image(grid, save_path)
        
    except Exception as e:
        print(f"[WARNING] Failed to save RGB comparison: {e}")

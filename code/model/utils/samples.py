"""
==============================================================================
Utility functions for visualizing 
and saving sample outputs during training/inference.
Shared between VAE reconstruction and diffusion sampling.
==============================================================================
"""
###### import libraries ######
# Standard libraries
import os
from typing import Dict, List, Optional, Tuple

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from model.utils.colors import get_colormap_for_layer, apply_colormap_to_tensor, get_categorical_colormap
from model.utils.layer_config import is_categorical_layer


def normalize_channel_for_visualization(
    channel: torch.Tensor,
    layer_info: Dict,
    is_reconstruction: bool = False
) -> torch.Tensor:
    """
    Normalize a single channel for visualization based on layer type.
    
    For categorical layers, pass the full [B, num_classes, H, W] tensor.
    Returns [B, 1, H, W] class index map normalized to [0, 1].
    
    Args:
        channel: Channel tensor [B, 1, H, W] or [B, num_classes, H, W] for categorical
        layer_info: Layer configuration dict from registry
        is_reconstruction: If True, channel is VAE reconstruction (may be logits for binary)
        
    Returns:
        Normalized channel [B, 1, H, W] in [0, 1] range
    """
    layer_type = layer_info.get('type', 'continuous')
    
    if layer_type == 'categorical':
        num_classes = layer_info.get('num_classes', 1)
        
        if is_reconstruction:
            # Reconstruction is logits [B, num_classes, H, W] -> softmax -> argmax
            class_indices = torch.softmax(channel, dim=1).argmax(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            # Input is one-hot [B, num_classes, H, W] -> argmax
            class_indices = channel.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize class indices to [0, 1] for colormap application
        if num_classes > 1:
            channel_vis = class_indices.float() / (num_classes - 1)
        else:
            channel_vis = class_indices.float()
        return channel_vis
    
    elif layer_type == 'binary':
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
    colormap_name: Optional[str] = None,
    mask: Optional[torch.Tensor] = None
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
        colormap_name: Name of colormap to use,
        mask: Optional mask tensor [B, 1, H, W] to overlay red border
    """
    n_samples = min(n_samples, input_channel.shape[0])
    
    # Concatenate input and reconstruction vertically (2*n_samples rows)
    comparison = torch.cat([input_channel[:n_samples], recon_channel[:n_samples]], dim=0)
    
    # Apply colormap if requested
    if apply_colormap and colormap_name:
        comparison = apply_colormap_to_tensor(comparison, colormap_name)
        
    # Overlay red border if mask is provided
    if mask is not None:
        # Upsample mask to match channel resolution
        mask_upsampled = F.interpolate(
            mask[:n_samples],
            size=(comparison.shape[2], comparison.shape[3]),
            mode='nearest'
        )
        
        # Convert grayscale to RGB for red border overlay
        if comparison.shape[1] == 1:
            comparison = comparison.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
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
        
        # Repeat boundary for both input and reconstruction samples (comparison has 2*n_samples batch size)
        mask_boundary = mask_boundary.repeat(2, 1, 1, 1)
        
        # Ensure mask_boundary is on same device as comparison
        mask_boundary = mask_boundary.to(comparison.device)
        
        # Apply red border (set R=1, G=0, B=0 where boundary)
        comparison[:, 0:1, :, :] = torch.where(mask_boundary > 0, torch.ones_like(comparison[:, 0:1, :, :]), comparison[:, 0:1, :, :])
        comparison[:, 1:2, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(comparison[:, 1:2, :, :]), comparison[:, 1:2, :, :])
        comparison[:, 2:3, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(comparison[:, 2:3, :, :]), comparison[:, 2:3, :, :])
    
    # Create grid (n_samples columns, 2 rows: input on top, recon on bottom)
    grid = make_grid(comparison, nrow=n_samples, normalize=normalize, padding=2, pad_value=1.0)
    save_image(grid, save_path)


def _overlay_mask_border(
    channel_vis: torch.Tensor,
    mask: torch.Tensor,
    n_samples: int
) -> torch.Tensor:
    """
    Overlay red border on visualization at mask boundaries.
    
    Args:
        channel_vis: Visualization tensor [B, C, H, W] (1 or 3 channels)
        mask: Binary mask [B, 1, H, W]
        n_samples: Number of samples to process
        
    Returns:
        Tensor with red border overlay [B, 3, H, W]
    """
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
    
    return channel_vis


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
    Handles categorical layers by aggregating their one-hot channels into
    a single class-index visualization with a discrete colormap.
    
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
    categorical_processed = set()
    
    ch_idx = 0
    while ch_idx < len(layer_names):
        if ch_idx >= tensor.shape[1]:
            break
        
        layer_name = layer_names[ch_idx]
        layer_info = layers_registry.get(layer_name, {})
        layer_type = layer_info.get('type', 'continuous')
        
        # Handle categorical layers: aggregate all num_classes channels
        if is_categorical_layer(layer_info) and layer_name not in categorical_processed:
            categorical_processed.add(layer_name)
            num_classes = layer_info.get('num_classes', 1)
            
            # Extract all channels for this categorical layer
            cat_channels = tensor[:, ch_idx:ch_idx+num_classes, :, :]  # [B, num_classes, H, W]
            
            # Normalize: argmax to class index map [B, 1, H, W]
            channel_vis = normalize_channel_for_visualization(
                cat_channels, layer_info, is_reconstruction
            )
            
            # Apply discrete categorical colormap
            if use_colormaps:
                cmap = get_categorical_colormap(num_classes)
                channel_vis = apply_colormap_to_tensor(channel_vis, cmap)
            
            # Overlay mask border if provided
            if mask is not None:
                channel_vis = _overlay_mask_border(channel_vis, mask, n_samples)
            
            # Save this categorical layer as single visualization
            save_path = os.path.join(save_dir, f'{filename_prefix}_{layer_name}.png')
            save_channel_visualization(
                channel_vis, save_path, n_samples=n_samples,
                normalize=False, apply_colormap=False
            )
            
            ch_idx += num_classes
            continue
        
        # Skip channels belonging to already-processed categorical layer
        if is_categorical_layer(layer_info) and layer_name in categorical_processed:
            ch_idx += 1
            continue
        
        channel = tensor[:, ch_idx:ch_idx+1, :, :]
        
        # Normalize channel for visualization
        channel_vis = normalize_channel_for_visualization(
            channel, layer_info, is_reconstruction
        )
        
        # Determine if we should apply colormap
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
            channel_vis = _overlay_mask_border(channel_vis, mask, n_samples)
        
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
        
        ch_idx += 1


def save_layerwise_comparisons(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    channel_names: List[str],
    layer_names: List[str],
    layers_registry: Dict,
    save_dir: str,
    filename_prefix: str,
    n_samples: int = 8,
    use_colormaps: bool = True,
    mask: Optional[torch.Tensor] = None
) -> None:
    """
    Save input vs reconstruction comparisons for each layer.
    
    This is the main function for VAE reconstruction visualization.
    Handles categorical layers by aggregating their one-hot channels.
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstruction tensor [B, C, H, W] (logits for binary/categorical)
        channel_names: List of channel names (e.g., ['rgb:red', 'buildings', 'building_shapes:class_0'])
        layer_names: List of layer names for each channel
        layers_registry: Layer configuration registry
        save_dir: Directory to save images
        filename_prefix: Prefix for filenames (e.g., 'recon_step_1000')
        n_samples: Number of samples to visualize
        use_colormaps: Whether to apply colormaps to continuous layers
        mask: Optional mask tensor to overlay on visualizations
    """
    
    n_samples = min(n_samples, input_tensor.shape[0])
    categorical_processed = set()
    
    ch_idx = 0
    while ch_idx < len(channel_names):
        if ch_idx >= input_tensor.shape[1]:
            break
        
        channel_name = channel_names[ch_idx]
        layer_name = layer_names[ch_idx]
        layer_info = layers_registry.get(layer_name, {})
        layer_type = layer_info.get('type', 'continuous')
        
        # Handle categorical layers: aggregate all num_classes channels
        if is_categorical_layer(layer_info) and layer_name not in categorical_processed:
            categorical_processed.add(layer_name)
            num_classes = layer_info.get('num_classes', 1)
            
            # Extract all channels for this categorical layer
            input_cat = input_tensor[:, ch_idx:ch_idx+num_classes, :, :]  # [B, num_classes, H, W]
            recon_cat = recon_tensor[:, ch_idx:ch_idx+num_classes, :, :]  # [B, num_classes, H, W]
            
            # Normalize to class index maps [B, 1, H, W]
            input_vis = normalize_channel_for_visualization(input_cat, layer_info, is_reconstruction=False)
            recon_vis = normalize_channel_for_visualization(recon_cat, layer_info, is_reconstruction=True)
            
            # Apply discrete categorical colormap
            cmap = get_categorical_colormap(num_classes)
            
            # Save comparison
            save_path = os.path.join(save_dir, f'{filename_prefix}_{layer_name}.png')
            save_comparison_visualization(
                input_vis, recon_vis, save_path,
                n_samples=n_samples, normalize=False,
                apply_colormap=True, colormap_name=cmap, mask=mask
            )
            
            ch_idx += num_classes
            continue
        
        # Skip channels belonging to already-processed categorical layer
        if is_categorical_layer(layer_info) and layer_name in categorical_processed:
            ch_idx += 1
            continue
        
        input_ch = input_tensor[:, ch_idx:ch_idx+1, :, :]
        recon_ch = recon_tensor[:, ch_idx:ch_idx+1, :, :]
        
        # Normalize both input and reconstruction
        input_vis = normalize_channel_for_visualization(input_ch, layer_info, is_reconstruction=False)
        recon_vis = normalize_channel_for_visualization(recon_ch, layer_info, is_reconstruction=True)
        
        # Determine if we should apply colormap
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
            colormap_name=cmap_name,
            mask=mask
        )
        
        ch_idx += 1


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


def save_latent_visualization(
    latent: torch.Tensor,
    save_path: str,
    n_samples: int = 8,
):
    """
    Save visualization of VAE latent channels.
    
    Args:
        latent: Latent tensor [B, C, H, W]
        save_path: Path to save image
        n_samples: Number of samples to show
        mode: 'semantic' or 'satellite' for colormap hints
    """
    n_samples = min(n_samples, latent.shape[0])
    n_channels = latent.shape[1]
    
    # Create subplot grid: rows = channels, cols = samples
    fig, axes = plt.subplots(n_channels, n_samples, figsize=(2 * n_samples, 2 * n_channels))
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for ch in range(n_channels):
        for s in range(n_samples):
            ax = axes[ch, s]
            img = latent[s, ch].cpu().numpy()
            
            # Normalize for visualization
            vmin, vmax = np.percentile(img, [2, 98])
            img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            ax.imshow(img_norm, cmap=sns.color_palette("rocket", as_cmap=True))
            ax.axis('off')
            
            if s == 0:
                ax.set_ylabel(f'Ch {ch}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
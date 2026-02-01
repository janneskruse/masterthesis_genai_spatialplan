"""
Post-processing utilities for diffusion model outputs.

Functions for sharpening binary channels, thresholding, etc.
"""
###### import libraries ######
# Data Science/ML
import torch
from typing import List, Dict, Any, Optional


def sharpen_binary_channels(
    tensor: torch.Tensor,
    layer_names: List[str],
    layers_registry: Dict[str, Any],
    threshold: float = 0.5,
    inplace: bool = False
) -> torch.Tensor:
    """
    Apply binary thresholding to binary-type channels for crisp outputs.
    
    After diffusion sampling and VAE decoding, binary channels (buildings, streets, 
    vegetation) often have soft probabilities. This function applies a threshold to
    produce clean binary masks suitable for GIS vectorization.
    
    Args:
        tensor: Decoded semantic tensor [B, C, H, W] with values in [0, 1]
        layer_names: List of layer names corresponding to channels
        layers_registry: Layer configuration registry containing 'type' for each layer
        threshold: Threshold value (default 0.5)
        inplace: If True, modify tensor in place. If False, return a copy.
        
    Returns:
        Tensor with binary channels thresholded to {0, 1}
        
    Example:
        >>> decoded = vae.decode(sampled_latent)  # [B, 4, H, W]
        >>> layer_names = ['buildings', 'streets', 'buildings_heights', 'vegetation']
        >>> sharpened = sharpen_binary_channels(decoded, layer_names, layers_registry)
        >>> # buildings, streets, vegetation now have values in {0, 1}
        >>> # buildings_heights (continuous) unchanged
    """
    if not inplace:
        tensor = tensor.clone()
    
    for ch_idx, layer_name in enumerate(layer_names):
        if ch_idx >= tensor.shape[1]:
            break
            
        # Get layer config from registry
        layer_config = layers_registry.get(layer_name, {})
        layer_type = layer_config.get('type', 'continuous')
        
        # Only threshold binary-type layers
        if layer_type == 'binary':
            tensor[:, ch_idx] = (tensor[:, ch_idx] > threshold).float()
    
    return tensor


def apply_post_processing(
    tensor: torch.Tensor,
    layer_names: List[str],
    layers_registry: Dict[str, Any],
    post_process_config: Optional[Dict[str, Any]] = None,
    inplace: bool = False
) -> torch.Tensor:
    """
    Apply all configured post-processing steps to decoded tensor.
    
    Reads post_process_config and applies enabled transformations.
    
    Args:
        tensor: Decoded tensor [B, C, H, W]
        layer_names: List of layer names corresponding to channels
        layers_registry: Layer configuration registry
        post_process_config: Dict with post-processing options:
            - sharpen_binary: bool - Apply binary thresholding (default: False)
            - threshold: float - Threshold value for binary (default: 0.5)
        inplace: If True, modify tensor in place
        
    Returns:
        Post-processed tensor
    """
    if post_process_config is None:
        return tensor
    
    # Apply binary sharpening if enabled
    if post_process_config.get('sharpen_binary', False):
        threshold = post_process_config.get('threshold', 0.5)
        tensor = sharpen_binary_channels(
            tensor=tensor,
            layer_names=layer_names,
            layers_registry=layers_registry,
            threshold=threshold,
            inplace=inplace
        )
    
    return tensor

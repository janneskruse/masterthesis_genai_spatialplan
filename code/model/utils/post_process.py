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
    inplace: bool = False,
    mask: Optional[torch.Tensor] = None
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
            - normalize_mask_region: bool - Normalize generated region (inside mask) to [0,1]
              without destroying extreme values. Context (outside mask) stays unchanged.
              This fixes "faint context" visualization issue properly.
            - clip_output: bool - Hard clamp all channels to [0, 1] (default: False)
              Use normalize_mask_region instead when possible.
        inplace: If True, modify tensor in place
        mask: Optional mask tensor [B, 1, H, W] where 1=generated region, 0=context
              Required for normalize_mask_region option.
        
    Returns:
        Post-processed tensor
    """
    if post_process_config is None:
        return tensor
    
    if not inplace:
        tensor = tensor.clone()
    
    # Normalize only the generated region (inside mask) to [0,1]
    # This preserves extreme values as valid normalized values instead of clipping
    # Context (outside mask) stays unchanged since it's already [0,1] from dataset
    if post_process_config.get('normalize_mask_region', False) and mask is not None:
        for c in range(tensor.shape[1]):
            ch = tensor[:, c:c+1]  # [B, 1, H, W]
            
            # Get values inside mask
            mask_bool = mask > 0.5
            masked_values = ch[mask_bool]
            
            if masked_values.numel() > 0:
                ch_min = masked_values.min()
                ch_max = masked_values.max()
                
                if ch_max > ch_min:
                    # Normalize only inside mask: (x - min) / (max - min)
                    normalized = (ch - ch_min) / (ch_max - ch_min + 1e-8)
                    # Apply only to masked region, keep context unchanged
                    tensor[:, c:c+1] = torch.where(mask_bool, normalized, ch)
    
    # Fallback: hard clip to [0, 1] (less preferred, destroys information)
    if post_process_config.get('clip_output', False):
        tensor = torch.clamp(tensor, 0, 1)
    
    # Apply binary sharpening if enabled (on whole image)
    if post_process_config.get('sharpen_binary', False):
        threshold = post_process_config.get('threshold', 0.5)
        tensor = sharpen_binary_channels(
            tensor=tensor,
            layer_names=layer_names,
            layers_registry=layers_registry,
            threshold=threshold,
            inplace=True  # Already cloned above if needed
        )
    
    return tensor

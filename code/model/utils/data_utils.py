###### import libraries ######
# Standard libraries
from typing import Dict, List, Tuple, Union

# Data Science/ML
import torch
import numpy as np

# Local imports
from model.utils.layer_config import is_binary_layer

def apply_layer_mask(
    data: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply mask to layer data (e.g., mask building heights by building footprints).
    
    Args:
        data: Input tensor to mask
        mask: Binary mask tensor (same spatial dims as data)
        
    Returns:
        Masked tensor (data * mask)
    """
    return data * mask

def apply_filter(data, layer_config: Dict):
    """
    Apply filter to layer data (thresholds, binary conversion).
    
    Handles both torch.Tensor and numpy arrays efficiently.
    
    Args:
        data: Input tensor or numpy array
        layer_config: Layer configuration with optional 'filter' key
        
    Returns:
        Filtered data (binary {0, 1} if type='binary', otherwise continuous)
        Returns same type as input (torch.Tensor or numpy array)
    """
    is_torch = isinstance(data, torch.Tensor)
    is_binary = layer_config.get('type') == 'binary'
    filter_config = layer_config.get('filter', {})
    
    # No filter specified - just handle binary conversion if needed
    if not filter_config:
        if is_binary:
            if is_torch:
                return (data > 0.5).float()
            else:
                return (data > 0.5).astype(np.float32)
        return data
    
    # Create mask for filtering
    if is_torch:
        mask = torch.ones_like(data, dtype=torch.bool)
        
        if 'gte' in filter_config:
            mask = mask & (data >= filter_config['gte'])
        if 'lte' in filter_config:
            mask = mask & (data <= filter_config['lte'])
        if 'gt' in filter_config:
            mask = mask & (data > filter_config['gt'])
        if 'lt' in filter_config:
            mask = mask & (data < filter_config['lt'])
        
        # Return binary mask or filtered continuous data
        if is_binary:
            return mask.float()
        else:
            return torch.where(mask, data, torch.zeros_like(data))
    else:
        # NumPy path
        mask = np.ones_like(data, dtype=bool)
        
        if 'gte' in filter_config:
            mask = mask & (data >= filter_config['gte'])
        if 'lte' in filter_config:
            mask = mask & (data <= filter_config['lte'])
        if 'gt' in filter_config:
            mask = mask & (data > filter_config['gt'])
        if 'lt' in filter_config:
            mask = mask & (data < filter_config['lt'])
        
        # Return binary mask or filtered continuous data
        if is_binary:
            return mask.astype(np.float32)
        else:
            return np.where(mask, data, 0.0)

def apply_layer_transform(data, layer_config, layer_statistics=None, mask_data=None):
    """
    Apply transformations to layer data including filtering and normalization.
    
    Handles both torch.Tensor and numpy arrays.
    
    Args:
        data: numpy array or torch tensor of layer data
        layer_config: Layer configuration dict with optional 'filter', 'normalize', and 'mask_layer' keys
        layer_statistics: Optional dict with global statistics (min, max, mean, std, q01, q99, etc.)
        mask_data: Optional mask array to apply (for layers with mask_layer config)
        
    Returns:
        Transformed data (same type as input)
    """
    is_torch = isinstance(data, torch.Tensor)
    is_binary = is_binary_layer(layer_config)
    filter_config = layer_config.get('filter', {})
    
    # Step 1: Apply filters (thresholding, range filtering)
    if filter_config:
        data = apply_filter(data, layer_config)
    
    # Step 2: Convert to binary if needed
    if is_binary:
        if is_torch:
            data = (data > 0.5).float()
        else:
            data = (data > 0.5).astype(np.float32)
    
    # Step 3: Apply normalization with global statistics
    data = normalize_layer(data, layer_config, layer_statistics)
    
    # Step 4: Apply mask if specified (e.g., building heights masked by buildings)
    if mask_data is not None:
        mask_layer_name = layer_config.get('mask_layer', None)
        if mask_layer_name:
            # Apply mask: zero out values where mask is 0
            if is_torch:
                data = data * mask_data
            else:
                data = data * mask_data
    
    return data

def normalize_layer(data, layer_config, layer_statistics=None):
    """
    Normalize data layer based on configuration using global statistics.
    
    Supports multiple normalization strategies:
    - 'minmax': Normalize to [0, 1] using global min-max scaling
    - 'standardize': Z-score normalization using global mean and std
    - 'percentile': Clip to global percentiles then normalize to [0, 1]
    - 'clip': Clip to specified range (no scaling)
    - 'custom': Custom min/max normalization with specified bounds
    - None: No normalization (binary layers, already normalized data)
    
    Handles both torch.Tensor and numpy arrays.
    
    Args:
        data: Input data (torch.Tensor or numpy array)
        layer_config: Layer configuration with optional 'normalize' key
        layer_statistics: Optional dict with global statistics (min, max, mean, std, q01, q99, etc.)
                         If None, falls back to patch-wise statistics (not recommended)
        
    Returns:
        Normalized data (same type as input)
        
    Config examples:
        normalize: 'minmax'
        
        normalize: 'percentile'
        lower_percentile: 2
        upper_percentile: 98
        
        normalize: 'clip'
        clip_range: [-1, 1]
        
        normalize: 'custom'
        normalize_params:
          min: 250  # Kelvin
          max: 320  # Kelvin
    """
    is_torch = isinstance(data, torch.Tensor)
    normalize_method = layer_config.get('normalize', None)
    
    # No normalization needed
    if normalize_method is None:
        return data
    
    # Handle NaN values
    if is_torch:
        data = torch.nan_to_num(data, nan=0.0)
    else:
        data = np.nan_to_num(data, nan=0.0)
    
    # Apply normalization method
    if normalize_method == 'minmax':
        # Normalize to [0, 1] using global statistics
        if layer_statistics is not None:
            data_min = layer_statistics['min']
            data_max = layer_statistics['max']
        else:
            # Fallback to patch-wise (not recommended)
            if is_torch:
                data_min = data.min().item()
                data_max = data.max().item()
            else:
                data_min = data.min()
                data_max = data.max()
        
        if data_max > data_min:
            if is_torch:
                data = (data - data_min) / (data_max - data_min + 1e-8)
            else:
                data = (data - data_min) / (data_max - data_min + 1e-8)
    
    elif normalize_method == 'standardize':
        # Z-score normalization using global statistics: (x - mean) / std
        if layer_statistics is not None:
            mean = layer_statistics['mean']
            std = layer_statistics['std']
        else:
            # Fallback to patch-wise (not recommended)
            if is_torch:
                mean = data.mean().item()
                std = data.std().item()
            else:
                mean = data.mean()
                std = data.std()
        
        if std > 1e-8:
            if is_torch:
                data = (data - mean) / std
            else:
                data = (data - mean) / std
    
    elif normalize_method == 'percentile':
        # Clip to global percentiles then normalize to [0, 1]
        lower_percentile = layer_config.get('lower_percentile', 2)
        upper_percentile = layer_config.get('upper_percentile', 98)
        
        if layer_statistics is not None:
            # Use precomputed global percentiles
            # Map percentile to closest available (1, 2, 98, 99)
            if lower_percentile <= 1:
                p_low = layer_statistics['q01']
            else:
                p_low = layer_statistics['q02']
            
            if upper_percentile >= 99:
                p_high = layer_statistics['q99']
            else:
                p_high = layer_statistics['q98']
        else:
            # Fallback to patch-wise (not recommended)
            if is_torch:
                p_low = torch.quantile(data, lower_percentile / 100.0).item()
                p_high = torch.quantile(data, upper_percentile / 100.0).item()
            else:
                p_low = np.percentile(data, lower_percentile)
                p_high = np.percentile(data, upper_percentile)
        
        # Clip and normalize
        if is_torch:
            data = torch.clamp(data, p_low, p_high)
            if p_high > p_low:
                data = (data - p_low) / (p_high - p_low + 1e-8)
        else:
            data = np.clip(data, p_low, p_high)
            if p_high > p_low:
                data = (data - p_low) / (p_high - p_low + 1e-8)
    
    elif normalize_method == 'clip':
        # Clip to specified range
        clip_range = layer_config.get('clip_range', [-1, 1])
        if is_torch:
            data = torch.clamp(data, clip_range[0], clip_range[1])
        else:
            data = np.clip(data, clip_range[0], clip_range[1])
    
    elif normalize_method == 'custom':
        # Custom normalization with configurable min/max
        # Useful for domain-specific data like temperature in Kelvin
        normalize_params = layer_config.get('normalize_params', {})
        data_min = normalize_params.get('min', None)
        data_max = normalize_params.get('max', None)
        
        if data_min is None or data_max is None:
            raise ValueError(
                f"Custom normalization requires 'normalize_params' with 'min' and 'max' keys. "
                f"Got: {normalize_params}"
            )
        
        # Clip to specified range then normalize to [0, 1]
        if is_torch:
            data = torch.clamp(data, data_min, data_max)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min + 1e-8)
        else:
            data = np.clip(data, data_min, data_max)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")
    
    return data


# ===========================================================================
# TEMPERATURE CONTROL UTILITIES
# ===========================================================================

def masked_quantile(x: torch.Tensor, mask: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """
    Compute quantile within masked region.
    
    Args:
        x: Temperature tensor [B, 1, H, W] or [B, C, H, W]
        mask: Binary mask [B, 1, H, W] with 1 = inpaint region, 0 = context
        q: Quantile to compute (default 0.95 for p95)
        
    Returns:
        Quantile value per sample [B]
        
    Example:
        >>> temp = torch.randn(4, 1, 128, 128)
        >>> mask = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> p95 = masked_quantile(temp, mask, q=0.95)
        >>> p95.shape
        torch.Size([4])
    """
    B = x.shape[0]
    
    # If multi-channel, take first channel only (should be single-channel temperature)
    if x.ndim == 4 and x.shape[1] > 1:
        x = x[:, :1]
    
    out = []
    for b in range(B):
        xb = x[b].reshape(-1)  # Flatten spatial dimensions
        mb = mask[b].reshape(-1) > 0.5  # Boolean mask
        
        # Extract values inside mask
        vals = xb[mb]
        
        # Fallback to full image if mask is empty
        if vals.numel() == 0:
            vals = xb
        
        # Compute quantile
        out.append(torch.quantile(vals, q))
    
    return torch.stack(out, dim=0)


def normalize_scalar_like_layer(
    t_value: float, 
    layer_cfg: dict, 
    layer_stats: dict | None
) -> float:
    """
    Normalize a scalar threshold value using the same pipeline as layer data.
    
    This ensures consistency between user-provided thresholds (e.g., 35°C)
    and the normalized values used during training.
    
    Args:
        t_value: Scalar value in original units (e.g., degrees Celsius)
        layer_cfg: Layer configuration dict from layers registry
        layer_stats: Global statistics dict (min, max, mean, std, q01, q99, etc.)
        
    Returns:
        Normalized scalar value
        
    Example:
        >>> # LST layer with custom normalization [0, 80]°C
        >>> layer_cfg = {'normalize': 'custom', 'normalize_params': {'min': 0, 'max': 80}}
        >>> t_norm = normalize_scalar_like_layer(35.0, layer_cfg, None)
        >>> print(t_norm)  # 35/80 = 0.4375
    """
    # Wrap scalar as [1, 1, 1, 1] tensor
    t_tensor = torch.tensor([[[[t_value]]]], dtype=torch.float32)
    
    # Apply existing normalization function
    t_norm = normalize_layer(t_tensor, layer_cfg, layer_stats)
    
    # Extract scalar
    return float(t_norm.item())


def generate_tmax_training_target(
    t_current: torch.Tensor,
    strategy: str,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Generate temperature control training targets.
    
    Strategies:
    - 'relative': t_current + random_delta (learns continuous control around data)
    - 'sampled': random value from [0, 1] (explores full range)
    - 'fixed': constant target value (single-target training)
    
    Args:
        t_current: Current temperature statistic [B] (normalized)
        strategy: 'relative', 'sampled', or 'fixed'
        config: Training config dict with strategy-specific parameters
        device: Device to create tensors on
        
    Returns:
        Target temperature values [B] (normalized)
        
    Example:
        >>> t_current = torch.tensor([0.4, 0.6, 0.5, 0.7])
        >>> config = {'relative_delta_range': [-0.1, 0.1]}
        >>> targets = generate_tmax_training_target(t_current, 'relative', config, 'cpu')
        >>> # targets ≈ [0.35, 0.55, 0.48, 0.72] (current + random delta)
    """
    B = t_current.shape[0]
    
    if strategy == 'relative':
        # Add random delta to current value
        delta_range = config.get('relative_delta_range', [-0.15, 0.15])
        delta_min, delta_max = delta_range
        
        # Random delta per sample
        delta = torch.rand(B, device=device) * (delta_max - delta_min) + delta_min
        target = t_current + delta
        
        # Clamp to valid normalized range [0, 1]
        target = torch.clamp(target, 0.0, 1.0)
        
    elif strategy == 'sampled':
        # Sample random values from normalized range
        target = torch.rand(B, device=device)
        
    elif strategy == 'fixed':
        # Use fixed target value
        fixed_value = config.get('fixed_value', 0.5)
        if fixed_value is None:
            raise ValueError(
                "Strategy 'fixed' requires 'fixed_value' in config. "
                "Example: fixed_value: 0.5 (normalized)"
            )
        target = torch.full((B,), fixed_value, device=device, dtype=torch.float32)
        
    else:
        raise ValueError(
            f"Unknown training strategy: '{strategy}'. "
            f"Supported: 'relative', 'sampled', 'fixed'"
        )
    
    return target


def collate_fn(batch):
    """
    Custom collate function for batching dataset outputs.
    
    Handles all modes:
    - default/vae: (image, {'image': tensor|None, 'meta': dict})
    - diffusion: (latent, {'meta': dict, 'image': tensor, 'group_name': tensor, ...})
    
    Stacks first element (images/latents) and all tensor values in the dict.
    Preserves 'meta' as list of dicts, skips None values.
    """
    if isinstance(batch[0], tuple):
        # Stack first element (images or latents)
        images = torch.stack([item[0] for item in batch])
        
        # Batch the conditioning dictionary
        sample_cond = batch[0][1]
        cond_inputs = {}
        
        for key in sample_cond.keys():
            if key == 'meta':
                # Metadata: collect as list, don't stack
                cond_inputs[key] = [item[1][key] for item in batch]
            elif sample_cond[key] is None:
                # None values: skip (e.g., VAE mode where 'image' is None)
                cond_inputs[key] = None
            elif isinstance(sample_cond[key], torch.Tensor):
                # Tensors: stack along batch dimension
                stacked = torch.stack([item[1][key] for item in batch])
                
                # Special handling for scalar conditioning (e.g., tmax)
                # Ensure shape is [B] or [B, 1] for consistency
                if key == 'tmax' and stacked.dim() == 2 and stacked.shape[1] == 1:
                    stacked = stacked.squeeze(1)  # [B, 1] -> [B]
                
                cond_inputs[key] = stacked
            else:
                # Other types (lists, strings, etc.): keep first item (should be consistent)
                cond_inputs[key] = sample_cond[key]
        
        return images, cond_inputs
    else:
        # Just tensors, no conditioning
        return torch.stack(batch)
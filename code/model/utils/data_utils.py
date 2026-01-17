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

def collate_fn(batch):
    """
    Custom collate function to make sure conditioning inputs are properly batched.
    """
    if isinstance(batch[0], tuple):
        # With conditioning
        images = torch.stack([item[0] for item in batch])
        
        # Get first sample to check keys
        sample_cond = batch[0][1]
        
        cond_inputs = {}
        for key in sample_cond.keys():
            if key == 'meta':
                # Meta is a dict, just collect as list, don't stack
                cond_inputs[key] = [item[1][key] for item in batch]
            else:
                # Stack tensors
                cond_inputs[key] = torch.stack([item[1][key] for item in batch])
        
        return images, cond_inputs
    else:
        # Just images, no conditioning
        return torch.stack(batch)
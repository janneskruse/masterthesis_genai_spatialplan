###### import libraries ######
# Standard libraries
from typing import Dict, List, Optional, Union, Tuple

# Data Science/ML
import torch

# Local imports
from model.utils.data_utils import masked_quantile
from model.utils.inpainting import extract_inpainting_mask_fullres


def compute_scalar_statistic(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    statistic: str = "mean"
) -> torch.Tensor:
    """
    Compute a scalar statistic from a masked tensor.
    
    Args:
        tensor: Input tensor [C, H, W] or [H, W]
        mask: Binary mask [1, H, W] or [H, W] (1=compute, 0=ignore)
        statistic: One of ["mean", "max", "min", "p95", "p99"]
        
    Returns:
        Scalar tensor []
    """
    # Ensure mask is same spatial dims as tensor
    if tensor.ndim == 3 and mask.ndim == 3:
        # tensor [C, H, W], mask [1, H, W]
        mask = mask.expand_as(tensor)
    elif tensor.ndim == 2 and mask.ndim == 3:
        # tensor [H, W], mask [1, H, W]
        mask = mask.squeeze(0)
    
    # Apply mask
    masked_values = tensor[mask > 0.5]
    
    if len(masked_values) == 0:
        # No valid pixels - return 0
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    
    # Compute statistic
    if statistic == "mean":
        return masked_values.mean()
    elif statistic == "max":
        return masked_values.max()
    elif statistic == "min":
        return masked_values.min()
    elif statistic == "p95":
        return masked_quantile(tensor.unsqueeze(0), mask.unsqueeze(0), q=0.95).squeeze()
    elif statistic == "p99":
        return masked_quantile(tensor.unsqueeze(0), mask.unsqueeze(0), q=0.99).squeeze()
    else:
        raise ValueError(f"Unknown statistic: {statistic}. Supported: mean, max, min, p95, p99")


def extract_layer_from_unified(
    unified_image: torch.Tensor,
    channel_names: List[str],
    layer_name: str
) -> Optional[torch.Tensor]:
    """
    Extract a specific layer from unified image tensor.
    
    Args:
        unified_image: Tensor [C, H, W]
        channel_names: List of channel names matching C dimension
        layer_name: Name of layer to extract
        
    Returns:
        Layer tensor [1, H, W] or None if not found
    """
    try:
        idx = channel_names.index(layer_name)
        return unified_image[idx:idx+1, :, :]
    except ValueError:
        return None


def compute_scalar_from_layer(
    unified_image: torch.Tensor,
    channel_names: List[str],
    control_spec: Dict,
    cond: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute a scalar statistic from a layer according to control specification.
    
    Args:
        unified_image: Full unified tensor [C, H, W] with all layers
        channel_names: List of layer names for each channel
        control_spec: Control config dict with keys:
            - layer: layer name to extract
            - statistic: "mean", "max", "min", "p95", "p99"
            - region: "mask" or "full"
            - mask_by_layer: optional secondary layer to mask by
        cond: Conditioning dict (contains inpainting mask if region="mask")
        
    Returns:
        Scalar tensor [] with computed statistic
    """
    layer_name = control_spec['layer']
    statistic = control_spec.get('statistic', 'mean')
    region = control_spec.get('region', 'mask')
    mask_by_layer = control_spec.get('mask_by_layer')
    
    # Extract target layer
    layer_tensor = extract_layer_from_unified(unified_image, channel_names, layer_name)
    if layer_tensor is None:
        raise ValueError(f"Layer '{layer_name}' not found in unified_image. Available: {channel_names}")
    
    # Get full resolution shape
    fullres_shape = unified_image.shape[-2:]
    
    # Build mask
    if region == "mask":
        # Use inpainting mask
        mask = extract_inpainting_mask_fullres(cond, fullres_shape)
        if mask is None:
            # No mask found - fallback to full region
            mask = torch.ones(1, *fullres_shape, device=unified_image.device)
    else:
        # Full region
        mask = torch.ones(1, *fullres_shape, device=unified_image.device)
    
    # Apply secondary layer mask if specified
    if mask_by_layer:
        secondary_layer = extract_layer_from_unified(unified_image, channel_names, mask_by_layer)
        if secondary_layer is not None:
            # Combine masks: effective_mask = inpaint_mask * secondary_mask
            mask = mask * (secondary_layer > 0.5).float()
    
    # Compute statistic
    scalar = compute_scalar_statistic(layer_tensor.squeeze(0), mask, statistic)
    
    return scalar


def generate_training_target_scalar(
    x_current: torch.Tensor,
    training_config: Dict,
    device: torch.device
) -> torch.Tensor:
    """
    Generate training target for scalar control based on strategy.
    
    Args:
        x_current: Current scalar value (from compute_scalar_from_layer)
        training_config: Training config with keys:
            - strategy: "relative", "sampled", or "fixed"
            - relative_delta_range: [min, max] for relative strategy
            - fixed_value: value for fixed strategy
        device: Device for tensor creation
        
    Returns:
        Target scalar tensor []
    """
    strategy = training_config.get('strategy', 'relative')
    
    if strategy == 'relative':
        # Current value + random delta
        delta_range = training_config.get('relative_delta_range', [-0.15, 0.15])
        delta = torch.FloatTensor(1).uniform_(*delta_range).to(device)
        target = x_current + delta
        
        # Clamp to [0, 1] for normalized values
        target = torch.clamp(target, 0.0, 1.0)
        
    elif strategy == 'sampled':
        # Random value from [0, 1]
        target = torch.rand(1, device=device)
        
    elif strategy == 'fixed':
        # Fixed target
        fixed_val = training_config.get('fixed_value', 0.5)
        target = torch.tensor(fixed_val, device=device)
        
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")
    
    return target.squeeze()  # Return scalar []


def parse_scalar_controls_config(config: Dict) -> List[Dict]:
    """
    Parse scalar_controls config and return list of enabled control specs.
    Also supports legacy temperature_control for backwards compatibility.
    
    Args:
        config: Full global config
        
    Returns:
        List of control spec dicts, each with:
            - name: control name
            - keys: list of scalar keys (e.g., ["tmax"] or ["veg_min", "veg_max"])
            - layer: layer name
            - statistic: statistic type
            - region: "mask" or "full"
            - mask_by_layer: optional
            - training: training config dict
    """
    controls = []
    
    # Check for new scalar_controls config
    scalar_config = config.get('scalar_controls', {})
    if scalar_config.get('enabled', False):
        control_specs = scalar_config.get('controls', [])
        for spec in control_specs:
            if spec.get('enabled', True):
                # Normalize keys format
                if 'key' in spec:
                    # Single key
                    spec['keys'] = [spec['key']]
                elif 'keys' not in spec:
                    raise ValueError(f"Control spec must have 'key' or 'keys': {spec}")
                
                controls.append(spec)
    
    # Backwards compatibility: check for legacy temperature_control
    temp_config = config.get('temperature_control', {})
    if temp_config.get('enabled', False) and not any(c.get('name') == 'temperature' for c in controls):
        # Convert to new format
        temp_control = {
            'name': 'temperature',
            'keys': ['tmax'],
            'layer': temp_config.get('temperature_layer', 'lst'),
            'statistic': temp_config.get('statistic', 'p95'),
            'region': temp_config.get('region', 'mask'),
            'training': temp_config.get('training', {})
        }
        controls.append(temp_control)
    
    return controls

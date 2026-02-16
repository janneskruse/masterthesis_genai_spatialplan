"""
================================================================================
Layer configuration parsing and utilities for the inpainting model.
Handles:
- Parsing layer definitions from config (string vs dict)
- Building mappings from VAE groups to layers and vice versa
- Extracting channel information for layers
- Validating configuration consistency
================================================================================
"""

###### import libraries ######
# Standard libraries
from typing import Dict, List, Tuple, Any, Optional

# Data Science/ML
import torch
import numpy as np


def parse_layer_config(layer_def: Any) -> Tuple[str, Dict]:
    """
    Parse layer definition from config.
    
    Supports both string and dict formats:
    - String: 'buildings' -> ('buildings', {})
    - Dict: {buildings: {layer: 'buildings', type: 'binary'}} -> ('buildings', {...})
    
    Args:
        layer_def: Layer definition (string or dict)
        
    Returns:
        (layer_name, layer_config)
    """
    if isinstance(layer_def, str):
        return layer_def, {}
    elif isinstance(layer_def, dict):
        # Extract first key as layer name
        layer_name = list(layer_def.keys())[0]
        layer_config = layer_def[layer_name]
        return layer_name, layer_config
    else:
        raise ValueError(f"Invalid layer definition: {layer_def}")


def get_layer_info(layers_registry: Dict, layer_name: str) -> Dict:
    """
    Get full layer information from registry.
    
    Args:
        layers_registry: Global layers configuration
        layer_name: Name of layer to look up
        
    Returns:
        Layer configuration dict
    """
    if layer_name not in layers_registry:
        raise ValueError(f"Unknown layer: {layer_name}. Available: {list(layers_registry.keys())}")
    
    return layers_registry[layer_name]


def count_layer_channels(layer_config: Dict) -> int:
    """
    Count number of channels for a layer.
    
    Args:
        layer_config: Layer configuration from registry
        
    Returns:
        Number of channels:
        - Multi-channel layers (e.g., RGB): len(channels)
        - Categorical layers: num_classes (one-hot encoded)
        - All other layers: 1
    """
    if 'channels' in layer_config:
        return len(layer_config['channels'])
    if is_categorical_layer(layer_config):
        num_classes = layer_config.get('num_classes', None)
        if num_classes is None:
            raise ValueError("Categorical layer must define 'num_classes' in config")
        return num_classes
    return 1


def is_binary_layer(layer_config: Dict) -> bool:
    """
    Check if layer contains binary data.
    
    Args:
        layer_config: Layer configuration
        
    Returns:
        True if binary, False otherwise
    """
    return layer_config.get('type', 'continuous') == 'binary'


def is_categorical_layer(layer_config: Dict) -> bool:
    """
    Check if layer contains categorical data (multi-class, one-hot encoded).
    
    Args:
        layer_config: Layer configuration
        
    Returns:
        True if categorical, False otherwise
    """
    return layer_config.get('type', 'continuous') == 'categorical'


def get_layer_source(layer_config: Dict) -> str:
    """
    Get source data layer name.
    
    Args:
        layer_config: Layer configuration
        
    Returns:
        Name of source layer in Xarray dataset
    """
    return layer_config.get('layer', layer_config.get('name', 'unknown'))


def build_group_to_layers_mapping(vae_groups: Dict) -> Dict[str, List[str]]:
    """
    Build mapping from VAE group names to layer lists.
    
    Args:
        vae_groups: VAE groups configuration
        
    Returns:
        Dict mapping group_name -> list of layer names
    """
    mapping = {}
    
    for group_name, group_config in vae_groups.items():
        layers = group_config.get('layers', [])
        
        # Parse layer definitions
        layer_names = []
        for layer_def in layers:
            layer_name, _ = parse_layer_config(layer_def)
            layer_names.append(layer_name)
        
        mapping[group_name] = layer_names
    
    return mapping


def get_layer_dice_config(layers_registry: Dict, layer_name: str) -> Dict:
    """
    Get dice loss configuration for a layer.
    
    Args:
        layers_registry: Global layers configuration
        layer_name: Name of layer
        
    Returns:
        Dict with 'use_dice' (bool) and 'weight' (float)
    """
    layer_config = layers_registry.get(layer_name, {})
    
    # Default dice config based on layer characteristics
    default_dice_config = {
        'use_dice': False,
        'weight': 0.5
    }
    
    # Binary layers benefit from dice loss (buildings, streets, water, vegetation)
    # Buildings and other binary masks need Dice to prevent salt-and-pepper noise
    binary_layers_with_dice = ['buildings', 'streets', 'water', 'vegetation']
    if layer_name in binary_layers_with_dice:
        default_dice_config['use_dice'] = True
        if layer_name == 'buildings':
            default_dice_config['weight'] = 0.5  # Standard weight for buildings
        elif layer_name in ['streets', 'water']:
            default_dice_config['weight'] = 0.75  # Higher weight for thin structures
        else:
            default_dice_config['weight'] = 0.5  # Default for other binary layers
    
    # Override with explicit config
    if 'dice_loss' in layer_config:
        dice_config = layer_config['dice_loss']
        if isinstance(dice_config, bool):
            default_dice_config['use_dice'] = dice_config
        elif isinstance(dice_config, dict):
            default_dice_config.update(dice_config)
    
    return default_dice_config


def build_layer_to_group_mapping(vae_groups: Dict) -> Dict[str, Tuple[str, int]]:
    """
    Build mapping from layer names to (group_name, channel_index).
    
    Args:
        vae_groups: VAE groups configuration
        
    Returns:
        Dict mapping layer_name -> (group_name, channel_index_in_group)
    """
    mapping = {}
    
    for group_name, group_config in vae_groups.items():
        layers = group_config.get('layers', [])
        
        for channel_idx, layer_def in enumerate(layers):
            layer_name, _ = parse_layer_config(layer_def)
            mapping[layer_name] = (group_name, channel_idx)
    
    return mapping


def get_prediction_layers(stage_config: Dict, vae_groups: Dict) -> List[str]:
    """
    Get list of layers that should be predicted by a diffusion stage.
    
    Args:
        stage_config: Diffusion stage configuration
        vae_groups: VAE groups configuration
        
    Returns:
        List of layer names to predict
    """
    if 'prediction_layers' in stage_config:
        return stage_config['prediction_layers']
    
    # Fallback: use all layers in prediction group
    group_name = stage_config.get('prediction_group')
    if group_name and group_name in vae_groups:
        return build_group_to_layers_mapping(vae_groups)[group_name]
    
    return []


def get_conditioning_layers(stage_config: Dict) -> Dict[str, List[str]]:
    """
    Get conditioning layers grouped by VAE group.
    
    Args:
        stage_config: Diffusion stage configuration with 'conditioning' key
        
    Returns:
        Dict mapping group_name -> list of layer names
        
    Example:
        {
            'landuse': ['street_blocks'],
            'environmental': ['temperature']
        }
    """
    conditioning = stage_config.get('conditioning', {})
    latent_cond = conditioning.get('latent_space') or []
    
    grouped_layers = {}
    
    for cond_spec in latent_cond:
        group_name = cond_spec.get('group')
        layers = cond_spec.get('layers', [])
        
        if group_name not in grouped_layers:
            grouped_layers[group_name] = []
        
        grouped_layers[group_name].extend(layers)
    
    return grouped_layers


def get_channel_names(layer_name: str, layer_config: Dict) -> List[str]:
    """
    Get properly formatted channel names for a layer.
    
    - Single channel layers: returns ['layer_name']
    - Multi-channel layers: returns ['layer_name:channel1', 'layer_name:channel2', ...]
    
    Args:
        layer_name: Name of the layer
        layer_config: Layer configuration dict
        
    Returns:
        List of channel names with proper formatting
    
    Examples:
        get_channel_names('buildings', {'type': 'binary'}) -> ['buildings']
        get_channel_names('rgb', {'channels': ['red', 'green', 'blue']}) -> ['rgb:red', 'rgb:green', 'rgb:blue']
        get_channel_names('building_shapes', {'type': 'categorical', 'num_classes': 5}) -> ['building_shapes:class_0', ..., 'building_shapes:class_4']
    """
    if 'channels' in layer_config:
        channels = layer_config['channels']
        if len(channels) == 1:
            # Single channel: use layer name only
            return [layer_name]
        else:
            # Multiple channels: use layer:channel format
            return [f"{layer_name}:{ch}" for ch in channels]
    elif is_categorical_layer(layer_config):
        # Categorical layers are one-hot encoded: one channel per class
        num_classes = layer_config.get('num_classes', 1)
        return [f"{layer_name}:class_{i}" for i in range(num_classes)]
    else:
        # No channels specified: single channel layer
        return [layer_name]


def get_layer_channels_from_names(channel_names: List[str], target_layer: str) -> List[Tuple[int, str]]:
    """
    Find indices and channel names for a specific layer.
    
    Args:
        channel_names: List of all channel names in the patch
        target_layer: Layer name to find (e.g., 'rgb')
        
    Returns:
        List of (index, channel_name) tuples for the target layer
        
    Examples:
        channel_names = ['rgb:red', 'rgb:green', 'rgb:blue', 'buildings', 'streets']
        get_layer_channels_from_names(channel_names, 'rgb') -> [(0, 'rgb:red'), (1, 'rgb:green'), (2, 'rgb:blue')]
    """
    matches = []
    for idx, name in enumerate(channel_names):
        layer_name = name.split(':')[0]
        if layer_name == target_layer:
            matches.append((idx, name))
    return matches


def validate_layer_config(config: Dict) -> None:
    """
    Validate layer configuration for consistency.
    
    Args:
        config: Full configuration dict
        
    Raises:
        ValueError if configuration is invalid
    """
    layers = config.get('layers', {})
    vae_groups = config.get('vae_groups', {})
    diffusion_stages = config.get('diffusion_stages', {})
    
    # Check that all layers in VAE groups exist in registry
    for group_name, group_config in vae_groups.items():
        for layer_def in group_config.get('layers', []):
            layer_name, _ = parse_layer_config(layer_def)
            if layer_name not in layers:
                raise ValueError(f"VAE group '{group_name}' references unknown layer: {layer_name}")
    
    # Check that diffusion stages reference valid groups and layers
    for stage_name, stage_config in diffusion_stages.items():
        # Check prediction group
        pred_group = stage_config.get('prediction_group')
        if pred_group and pred_group not in vae_groups:
            raise ValueError(f"Stage '{stage_name}' references unknown prediction group: {pred_group}")
        
        # Check conditioning groups
        conditioning = stage_config.get('conditioning', {})
        for cond_spec in (conditioning.get('latent_space') or []):
            cond_group = cond_spec.get('group')
            if cond_group not in vae_groups:
                raise ValueError(f"Stage '{stage_name}' references unknown conditioning group: {cond_group}")
            
            # Check conditioning layers exist in group
            group_layers = build_group_to_layers_mapping(vae_groups)[cond_group]
            for layer_name in cond_spec.get('layers', []):
                if layer_name not in group_layers:
                    raise ValueError(
                        f"Stage '{stage_name}' conditioning references layer '{layer_name}' "
                        f"not in group '{cond_group}'"
                    )
    
    print("✓ Layer configuration validated successfully")

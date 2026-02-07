"""
==============================================================================
Color utilities for visualization of different data layers.
Provides consistent colormaps across training and validation scripts.
=
"""
###### import libraries ######
# Standard libraries
from typing import Union

# Data Science/ML libraries
import numpy as np
import torch

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns


def get_colormap_for_layer(layer_name: str):
    """
    Get appropriate colormap for a layer based on its name.
    
    Args:
        layer_name: Name of the layer (e.g., 'temperature', 'ndvi', 'buildings_heights')
        
    Returns:
        matplotlib colormap
    """
    name_lower = layer_name.lower()
    
    if 'mask' in name_lower:
        # Binary mask: black to white
        colors = [(0, 0, 0), (1, 1, 1)]
        return LinearSegmentedColormap.from_list('binary', colors, N=100)
    elif 'temp' in name_lower or 'temperature' in name_lower:
        # Temperature: rocket colormap (cool to hot)
        return sns.color_palette("rocket", as_cmap=True)
    elif 'vegetation' in name_lower or 'ndvi' in name_lower:
        # Vegetation: red-yellow-green
        rdylgn = cm.get_cmap('RdYlGn', 256)
        newcolors = rdylgn(np.linspace(0.1, 1, 256))
        newcolors[0] = [0, 0, 0, 1]  # Black for no vegetation
        return ListedColormap(newcolors)
    elif 'height' in name_lower:
        # Heights: rocket colormap (low to high)
        return sns.color_palette("rocket", as_cmap=True)
    else:
        # Default: grayscale
        return 'gray'


def apply_colormap_to_tensor(tensor: torch.Tensor, cmap: Union[str, object]) -> torch.Tensor:
    """
    Apply matplotlib colormap to a tensor.
    
    Args:
        tensor: Input tensor [B, 1, H, W] in range [0, 1]
        cmap: matplotlib colormap or colormap name
        
    Returns:
        RGB tensor [B, 3, H, W] with colormap applied
    """
    # Get colormap if string
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    # Convert to numpy and apply colormap
    tensor_np = tensor.squeeze(1).cpu().numpy()  # [B, H, W]
    
    # Apply colormap to each image in batch
    colored_images = []
    for img in tensor_np:
        # Apply colormap (returns RGBA)
        colored = cmap(img)  # [H, W, 4]
        # Convert to RGB and transpose to [3, H, W]
        rgb = torch.from_numpy(colored[:, :, :3]).permute(2, 0, 1).float()
        colored_images.append(rgb)
    
    return torch.stack(colored_images)  # [B, 3, H, W]

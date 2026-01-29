"""
Inpainting mask generation utilities.

Functions for creating various types of inpainting masks (random, center, street blocks).
"""
###### import libraries ######
# Standard libraries
from typing import Optional, Dict, Any, Tuple

# Data Handling
import numpy as np

# Data Science/ML
import torch
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from skimage.draw import polygon as draw_polygon


def _compute_patch_seed(patch_info: Dict[str, Any], base_seed: int = 42) -> int:
    """
    Compute deterministic seed from patch metadata for reproducible mask generation.
    
    Args:
        patch_info: Patch metadata (should contain 'patch_idx' or coordinate info)
        base_seed: Base seed for hashing (default: 42)
        
    Returns:
        Deterministic integer seed unique to this patch
    """
    if patch_info is None:
        return base_seed
    
    # Try to use patch_idx if available
    if 'index' in patch_info:
        return base_seed + int(patch_info['index'])
    
    # Fallback: hash coordinates if available
    if 'region' in patch_info and 'y' in patch_info and 'x' in patch_info:
        # Create unique hash from region + coordinates
        hash_str = f"{patch_info['region']}_{patch_info['y']}_{patch_info['x']}"
        return base_seed + hash(hash_str) % (2**31)
    
    # Last resort: use base seed (non-deterministic per patch)
    return base_seed


def _generate_street_blocks_mask(
    H: int,
    W: int,
    method_config: Dict[str, Any],
    street_blocks_layer: np.ndarray,
    mask_info: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Generate mask from street blocks layer.
    
    Returns:
        Mask array or None if generation failed (triggers fallback)
    """
    # Create binary mask from street blocks
    block_mask = (street_blocks_layer > 0).astype(np.float32)
    
    if block_mask.sum() == 0:
        mask_info['fallback_reason'] = 'no_street_blocks'
        return None
    
    # Find connected components
    labeled_array, num_features = label(block_mask)
    
    # Filter components by max coverage, then select largest valid one
    max_coverage_percent = method_config.get('max_coverage_percent', 25)
    total_pixels = H * W
    max_pixels = (max_coverage_percent / 100.0) * total_pixels
    
    max_area = 0
    best_mask = np.zeros_like(block_mask)
    
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.float32)
        area = component.sum()
        
        # Only consider components that satisfy coverage constraint
        if area <= max_pixels and area > max_area:
            max_area = area
            best_mask = component
    
    # Check if we found any valid block
    if max_area == 0:
        mask_info['fallback_reason'] = 'all_blocks_too_large'
        return None
    
    block_mask = best_mask
    mask_info['coverage_percent'] = (block_mask.sum() / (H * W)) * 100
    
    return block_mask


def _generate_random_polygon_mask(
    H: int,
    W: int,
    method_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate random convex polygon mask.
    
    Algorithm:
    1. Sample random points uniformly in circle
    2. Compute convex hull to get polygon vertices
    3. Scale polygon to satisfy area constraints
    4. Rasterize to binary mask
    """
    
    min_nodes = method_config.get('min_nodes', 3)
    max_nodes = method_config.get('max_nodes', 10)
    min_area_percent = method_config.get('min_area_percent', 0.15)
    max_area_percent = method_config.get('max_area_percent', 25.0)
    
    # Sample number of vertices
    num_nodes = np.random.randint(min_nodes, max_nodes + 1)
    
    # Generate random points in circle (ensures convex hull is non-degenerate)
    angles = np.random.uniform(0, 2 * np.pi, num_nodes)
    radii = np.random.uniform(0.3, 1.0, num_nodes)  # Min radius 0.3 to avoid tiny polygons
    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    
    # Compute convex hull
    if len(points) >= 3:
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
    else:
        vertices = points
    
    # Scale vertices to pixel space with area constraint
    total_pixels = H * W
    min_area_px = (min_area_percent / 100.0) * total_pixels
    max_area_px = (max_area_percent / 100.0) * total_pixels
    
    # Target area (random between min and max)
    target_area = np.random.uniform(min_area_px, max_area_px)
    
    # Scale factor based on target area (polygon area ∝ scale²)
    # Estimate current area (use bounding box approximation)
    bbox_area = (vertices[:, 0].max() - vertices[:, 0].min()) * (vertices[:, 1].max() - vertices[:, 1].min())
    if bbox_area > 0:
        scale = np.sqrt(target_area / (bbox_area * 0.5))  # 0.5 factor for polygon vs bbox
    else:
        scale = np.sqrt(target_area)
    
    # Scale vertices (currently in [-1, 1] range after normalization)
    vertices = vertices * scale
    
    # Get polygon bounds
    poly_height = vertices[:, 1].max() - vertices[:, 1].min()
    poly_width = vertices[:, 0].max() - vertices[:, 0].min()
    
    # Center polygon at origin (shift to [-poly_height/2, poly_height/2])
    vertices[:, 1] -= (vertices[:, 1].max() + vertices[:, 1].min()) / 2
    vertices[:, 0] -= (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    
    # Random center position ensuring polygon fits within image
    margin_y = max(poly_height / 2, 0)
    margin_x = max(poly_width / 2, 0)
    
    if margin_y < H / 2 and margin_x < W / 2:
        center_y = np.random.uniform(margin_y, H - margin_y)
        center_x = np.random.uniform(margin_x, W - margin_x)
    else:
        # Polygon too large, center it
        center_y = H / 2
        center_x = W / 2
    
    # Translate to center position
    vertices[:, 1] += center_y
    vertices[:, 0] += center_x
    
    # Clip to image bounds (safety)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, H - 1)
    vertices[:, 0] = np.clip(vertices[:, 0], 0, W - 1)
    
    # Rasterize polygon
    mask = np.zeros((H, W), dtype=np.float32)
    rr, cc = draw_polygon(vertices[:, 1], vertices[:, 0], shape=(H, W))
    mask[rr, cc] = 1.0
    
    return mask


def _generate_random_rectangle_mask(
    H: int,
    W: int,
    method_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate random rectangle with variable aspect ratio.
    """
    min_aspect = method_config.get('min_aspect_ratio', 0.33)  # 1:3
    max_aspect = method_config.get('max_aspect_ratio', 3.0)   # 3:1
    min_area_percent = method_config.get('min_area_percent', 10)
    max_area_percent = method_config.get('max_area_percent', 25.0)
    
    total_pixels = H * W
    min_area_px = (min_area_percent / 100.0) * total_pixels
    max_area_px = (max_area_percent / 100.0) * total_pixels
    
    # Random target area
    target_area = np.random.uniform(min_area_px, max_area_px)
    
    # Random aspect ratio (width / height)
    aspect_ratio = np.random.uniform(min_aspect, max_aspect)
    
    # Solve: w * h = target_area, w/h = aspect_ratio
    # => h = sqrt(target_area / aspect_ratio), w = aspect_ratio * h
    rect_h = int(np.sqrt(target_area / aspect_ratio))
    rect_w = int(aspect_ratio * rect_h)
    
    # Clamp to image bounds
    rect_h = min(rect_h, H - 1)
    rect_w = min(rect_w, W - 1)
    
    # Random position
    y0 = np.random.randint(0, max(1, H - rect_h))
    x0 = np.random.randint(0, max(1, W - rect_w))
    
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y0+rect_h, x0:x0+rect_w] = 1.0
    
    return mask


def _generate_random_square_mask(
    H: int,
    W: int,
    method_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate random square mask.
    """
    size_px = method_config.get('size_px', 80)
    
    y0 = np.random.randint(0, max(1, H - size_px))
    x0 = np.random.randint(0, max(1, W - size_px))
    
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y0+size_px, x0:x0+size_px] = 1.0
    
    return mask


def _generate_center_square_mask(
    H: int,
    W: int,
    method_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate center square mask.
    """
    size_px = method_config.get('size_px', 80)
    
    y0 = (H - size_px) // 2
    x0 = (W - size_px) // 2
    
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y0+size_px, x0:x0+size_px] = 1.0
    
    return mask


def create_inpainting_mask(
    H: int,
    W: int,
    hole_config: Dict[str, Any],
    street_blocks_layer: Optional[np.ndarray] = None,
    patch_info: Optional[Dict[str, Any]] = None,
    stats_list: Optional[list] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Create inpainting hole mask using specified method or mixed strategy.
    
    Args:
        H: Height of the mask
        W: Width of the mask
        hole_config: Configuration dict with 'type' and method-specific params
            - type: 'mixed' | 'street_blocks' | 'random_polygon' | 'random_rectangle' | 'random_square' | 'center_square'
            - methods: List of method configs (for 'mixed' type only)
            - fallback_method: Method name to use when street_blocks fails (default: 'random_rectangle')
        street_blocks_layer: Optional street blocks data for street-block-based masking
        patch_info: Optional metadata about the patch (for stats tracking)
        stats_list: Optional list to append mask statistics to
        
    Returns:
        Binary mask array of shape (H, W) with 1.0 for inpainting region, 0.0 elsewhere
        
    Example config for mixed strategy:
        ```yaml
        inpainting_params:
          type: 'mixed'
          methods:
            - name: 'street_blocks'
              weight: 0.60
              max_coverage_percent: 25
            - name: 'random_polygon'
              weight: 0.20
              min_nodes: 3
              max_nodes: 10
              min_area_percent: 0.15
              max_area_percent: 25.0
            - name: 'random_rectangle'
              weight: 0.15
              min_aspect_ratio: 0.33
              max_aspect_ratio: 3.0
              min_area_percent: 0.15
              max_area_percent: 25.0
            - name: 'random_square'
              weight: 0.05
              size_px: 80
          fallback_method: 'random_rectangle'
        ```
    """
    hole_type = hole_config['type']
    
    # Set deterministic seed for reproducible mask generation per patch
    if patch_info is not None:
        patch_seed = _compute_patch_seed(patch_info, base_seed=seed)
        np.random.seed(patch_seed)
    
    # Initialize mask info for stats tracking
    mask_info = {
        'requested_type': hole_type,
        'actual_type': None,
        'coverage_percent': 0.0,
        'fallback_reason': None
    }
    
    if patch_info:
        mask_info.update(patch_info)
    
    # Handle mixed strategy: randomly select method based on weights
    if hole_type == 'mixed':
        methods = hole_config.get('methods', [])
        if not methods:
            raise ValueError("Mixed strategy requires 'methods' list in hole_config")
        
        # Randomly select method (exclude center_square from random selection)
        available_methods = [m for m in methods if m['name'] != 'center_square']
        available_weights = [m.get('weight', 1.0) for m in available_methods]
        available_weights = np.array(available_weights) / np.sum(available_weights)
        
        selected_method = np.random.choice(available_methods, p=available_weights)
        hole_type = selected_method['name'] # Use selected method type
        method_config = selected_method
        
        mask_info['requested_type'] = f"mixed({hole_type})"
    else:
        # Single method mode: use hole_config as method_config
        method_config = hole_config
    
    # Generate mask using selected method
    mask = None
    
    if hole_type == 'street_blocks':
        if street_blocks_layer is not None:
            mask = _generate_street_blocks_mask(H, W, method_config, street_blocks_layer, mask_info)
        
        # Fallback if street_blocks generation failed
        if mask is None:
            fallback_method = hole_config.get('fallback_method', 'random_rectangle')
            hole_type = fallback_method
            mask_info['actual_type'] = f"street_blocks_fallback({fallback_method})"
            
            # Find fallback method config if in mixed mode
            if 'methods' in hole_config:
                fallback_configs = [m for m in hole_config['methods'] if m['name'] == fallback_method]
                if fallback_configs:
                    method_config = fallback_configs[0]
                else:
                    # Use default config for fallback
                    method_config = {'name': fallback_method}
    
    # Generate geometric masks
    if mask is None:
        if hole_type == 'random_polygon':
            mask = _generate_random_polygon_mask(H, W, method_config)
            mask_info['actual_type'] = mask_info.get('actual_type') or 'random_polygon'
        elif hole_type == 'random_rectangle':
            mask = _generate_random_rectangle_mask(H, W, method_config)
            mask_info['actual_type'] = mask_info.get('actual_type') or 'random_rectangle'
        elif hole_type == 'random_square':
            mask = _generate_random_square_mask(H, W, method_config)
            mask_info['actual_type'] = mask_info.get('actual_type') or 'random_square'
        elif hole_type == 'center_square':
            mask = _generate_center_square_mask(H, W, method_config)
            mask_info['actual_type'] = 'center_square'
        else:
            raise NotImplementedError(f"Hole type '{hole_type}' not implemented")
    
    # Update coverage for geometric masks
    if mask is not None and mask_info.get('coverage_percent', 0.0) == 0.0:
        mask_info['coverage_percent'] = (mask.sum() / (H * W)) * 100
    
    # Set actual_type if not already set (for street_blocks success case)
    if mask_info['actual_type'] is None:
        mask_info['actual_type'] = 'street_blocks'
    
    if stats_list is not None:
        stats_list.append(mask_info)
    
    return mask


def extract_inpainting_mask_fullres(
    cond: Dict[str, torch.Tensor],
    fullres_shape: Tuple[int, int]
) -> Optional[torch.Tensor]:
    """
    Extract inpainting mask from conditioning and upsample to full resolution.
    
    Args:
        cond: Conditioning dict with 'image' and 'meta'
        fullres_shape: Target (H, W) for upsampling
        
    Returns:
        Mask tensor [1, H, W] or None if no mask found
    """
    if 'image' not in cond or 'meta' not in cond:
        return None
    
    # Get pixel_space_names from meta
    meta = cond['meta']
    pixel_space_names = meta.get('pixel_space_names', [])
    
    # Find inpainting mask
    try:
        mask_idx = pixel_space_names.index('inpainting_mask')
        mask = cond['image'][mask_idx:mask_idx+1, :, :]  # [1, H_latent, W_latent]
        
        # Upsample to full resolution
        if mask.shape[-2:] != fullres_shape:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),  # [1, 1, H_latent, W_latent]
                size=fullres_shape,
                mode='nearest'
            ).squeeze(0)  # [1, H, W]
        
        return mask
    except (ValueError, IndexError):
        return None
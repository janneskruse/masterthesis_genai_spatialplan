"""
Inpainting mask generation utilities.

Functions for creating various types of inpainting masks (random, center, street blocks).
"""

import numpy as np
from typing import Optional, Dict, Any


def create_inpainting_mask(
    H: int,
    W: int,
    hole_config: Dict[str, Any],
    street_blocks_layer: Optional[np.ndarray] = None,
    patch_info: Optional[Dict[str, Any]] = None,
    stats_list: Optional[list] = None
) -> np.ndarray:
    """
    Create inpainting hole mask.
    
    Args:
        H: Height of the mask
        W: Width of the mask
        hole_config: Configuration dict with 'type', 'size_px', and 'max_coverage_percent'
        street_blocks_layer: Optional street blocks data for street-block-based masking
        patch_info: Optional metadata about the patch (for stats tracking)
        stats_list: Optional list to append mask statistics to
        
    Returns:
        Binary mask array of shape (H, W) with 1.0 for inpainting region, 0.0 elsewhere
    """
    hole_type = hole_config['type']
    hole_size = hole_config['size_px']
    
    mask_info = {
        'requested_type': hole_type,
        'actual_type': None,
        'coverage_percent': 0.0,
        'fallback_reason': None
    }
    
    if patch_info:
        mask_info.update(patch_info)
    
    if hole_type == 'street_blocks' and street_blocks_layer is not None:
        # Create binary mask from street blocks
        block_mask = (street_blocks_layer > 0).astype(np.float32)
        
        if block_mask.sum() == 0:
            # Fallback to random square if no street blocks
            hole_type = 'random_square'
            mask_info['fallback_reason'] = 'no_street_blocks'
            mask_info['actual_type'] = 'random_square'
        else:
            # Find connected pixels/street blocks
            from scipy.ndimage import label
            labeled_array, num_features = label(block_mask)
            
            # Select largest connected component
            max_area = 0
            best_mask = np.zeros_like(block_mask)
            for i in range(1, num_features + 1):
                component = (labeled_array == i).astype(np.float32)
                area = component.sum()
                if area > max_area:
                    max_area = area
                    best_mask = component
            
            block_mask = best_mask
            
            # Check if block covers more than max_coverage_percent of image
            coverage_percent = (block_mask.sum() / (H * W)) * 100
            max_coverage_percent = hole_config.get('max_coverage_percent', 25)
            mask_info['coverage_percent'] = coverage_percent
            if coverage_percent > max_coverage_percent:
                # Fallback to random square if block is too large
                hole_type = 'random_square'
                mask_info['fallback_reason'] = 'block_too_large'
                mask_info['actual_type'] = 'random_square'
            else:
                mask_info['actual_type'] = 'street_blocks'
                if stats_list is not None:
                    stats_list.append(mask_info)
                return block_mask
    
    # Generate geometric masks
    if hole_type == 'random_square':
        y0 = np.random.randint(0, max(1, H - hole_size))
        x0 = np.random.randint(0, max(1, W - hole_size))
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y0:y0+hole_size, x0:x0+hole_size] = 1.0
        mask_info['actual_type'] = 'random_square'
    elif hole_type == 'center_square':
        y0 = (H - hole_size) // 2
        x0 = (W - hole_size) // 2
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y0:y0+hole_size, x0:x0+hole_size] = 1.0
        mask_info['actual_type'] = 'center_square'
    else:
        raise NotImplementedError(f"Hole type {hole_type} not implemented")
    
    if stats_list is not None:
        stats_list.append(mask_info)
    
    return mask

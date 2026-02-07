"""
==============================================================================
Building quality metrics for evaluating inpainting realism.

Computes semantic metrics during sampling to track training progress:
- IoU and boundary consistency
- Edge smoothness and shape realism
- Size and aspect ratio distributions
- Context coherence between inpainted and surrounding regions

TO-DO: Use SSIM to assess visual quality of inpainted buildings compared to ground truth.
==============================================================================
"""

###### import libraries ######
# Standard libraries
from typing import Dict, Optional, List, Tuple

# Data handling
import numpy as np
import torch
import torch.nn.functional as F

# Image processing
from scipy.stats import wasserstein_distance
from skimage import measure, morphology
from skimage.filters import sobel


def compute_iou(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Intersection over Union for binary predictions.
    
    Args:
        pred: Predicted binary mask [B, 1, H, W] or [H, W]
        true: Ground truth binary mask [B, 1, H, W] or [H, W]
        mask: Optional region mask to compute IoU within [B, 1, H, W] or [H, W]
        
    Returns:
        IoU score [0, 1]
    """
    # Ensure binary
    pred_binary = (pred > 0.5).float()
    true_binary = (true > 0.5).float()
    
    # Apply region mask if provided
    if mask is not None:
        mask_binary = (mask > 0.5).float()
        pred_binary = pred_binary * mask_binary
        true_binary = true_binary * mask_binary
    
    # Compute intersection and union
    intersection = (pred_binary * true_binary).sum()
    union = (pred_binary + true_binary).clamp(0, 1).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def compute_dice(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Dice coefficient (F1 score) for binary predictions.
    
    Args:
        pred: Predicted binary mask [B, 1, H, W] or [H, W]
        true: Ground truth binary mask [B, 1, H, W] or [H, W]
        mask: Optional region mask
        
    Returns:
        Dice score [0, 1]
    """
    pred_binary = (pred > 0.5).float()
    true_binary = (true > 0.5).float()
    
    if mask is not None:
        mask_binary = (mask > 0.5).float()
        pred_binary = pred_binary * mask_binary
        true_binary = true_binary * mask_binary
    
    intersection = (pred_binary * true_binary).sum()
    cardinality = pred_binary.sum() + true_binary.sum()
    
    if cardinality == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection / cardinality).item()


def sobel_edge_detection(binary_map: torch.Tensor) -> np.ndarray:
    """
    Apply Sobel edge detection to extract building boundaries.
    
    Args:
        binary_map: Binary tensor [B, 1, H, W] or [H, W]
        
    Returns:
        Edge magnitude map as numpy array
    """
    # Convert to numpy and squeeze extra dims
    if isinstance(binary_map, torch.Tensor):
        binary_np = binary_map.squeeze().cpu().numpy()
    else:
        binary_np = binary_map.squeeze()
    
    # Apply Sobel filter
    edges = sobel(binary_np)
    
    return edges


def compute_edge_smoothness(edges: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute edge smoothness score based on edge gradient variance.
    Lower variance = smoother edges (less jagged).
    
    Args:
        edges: Edge magnitude map from sobel_edge_detection
        threshold: Threshold to binarize edges
        
    Returns:
        Smoothness score - higher is better (less jagged)
    """
    # Binarize edges
    edge_binary = edges > threshold
    
    if not edge_binary.any():
        return 1.0  # No edges = perfectly smooth
    
    # Compute edge curvature (second derivative)
    edge_float = edge_binary.astype(float)
    
    # Compute gradient magnitude variance along edges
    grad_y = np.gradient(edge_float, axis=0)
    grad_x = np.gradient(edge_float, axis=1)
    grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # High variance = jagged edges, low variance = smooth edges
    edge_variance = np.var(grad_magnitude[edge_binary])
    
    # Normalize to [0, 1] range (lower variance = higher score)
    smoothness = 1.0 / (1.0 + edge_variance)
    
    return float(smoothness)


def compute_building_areas(binary_map: torch.Tensor, min_size: int = 4) -> np.ndarray:
    """
    Extract individual building areas (in pixels) from binary map.
    
    Args:
        binary_map: Binary building mask [B, 1, H, W] or [H, W]
        min_size: Minimum building size to consider (filter noise)
        
    Returns:
        Array of building areas (number of pixels per building)
    """
    # Convert to numpy
    if isinstance(binary_map, torch.Tensor):
        binary_np = (binary_map.squeeze() > 0.5).cpu().numpy()
    else:
        binary_np = (binary_map.squeeze() > 0.5)
    
    # Label connected components
    labeled = measure.label(binary_np, connectivity=2)
    regions = measure.regionprops(labeled)
    
    # Extract areas, filter small noise
    areas = np.array([r.area for r in regions if r.area >= min_size])
    
    return areas if len(areas) > 0 else np.array([0])


def compute_aspect_ratios(binary_map: torch.Tensor, min_size: int = 4) -> np.ndarray:
    """
    Compute aspect ratios of individual buildings.
    Realistic buildings should have ratios roughly in [0.3, 3.0].
    
    Args:
        binary_map: Binary building mask [B, 1, H, W] or [H, W]
        min_size: Minimum building size to consider
        
    Returns:
        Array of aspect ratios (width/height) per building
    """
    if isinstance(binary_map, torch.Tensor):
        binary_np = (binary_map.squeeze() > 0.5).cpu().numpy()
    else:
        binary_np = (binary_map.squeeze() > 0.5)
    
    labeled = measure.label(binary_np, connectivity=2)
    regions = measure.regionprops(labeled)
    
    ratios = []
    for r in regions:
        if r.area >= min_size:
            # Get bounding box dimensions
            minr, minc, maxr, maxc = r.bbox
            height = maxr - minr
            width = maxc - minc
            
            if height > 0:
                ratio = width / height
                ratios.append(ratio)
    
    return np.array(ratios) if len(ratios) > 0 else np.array([1.0])


def compute_boundary_consistency(
    pred: torch.Tensor,
    mask: torch.Tensor,
    ring_width: int = 3
) -> float:
    """
    Measure density consistency at mask boundary (detect seam artifacts).
    Lower values = better blending between inpainted and context regions.
    
    Args:
        pred: Predicted binary map [B, 1, H, W] or [H, W]
        mask: Inpainting mask [B, 1, H, W] or [H, W]
        ring_width: Width of boundary ring to analyze (pixels)
        
    Returns:
        Boundary jump score - lower is better (0 = perfect match)
    """
    # Ensure same shape
    if pred.shape != mask.shape:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode='nearest')
    
    # Convert to numpy
    pred_np = pred.squeeze().cpu().numpy() if isinstance(pred, torch.Tensor) else pred.squeeze()
    mask_np = (mask.squeeze() > 0.5).cpu().numpy() if isinstance(mask, torch.Tensor) else (mask.squeeze() > 0.5)
    
    # Create boundary ring: dilate mask outward, erode mask inward
    mask_dilated = morphology.binary_dilation(mask_np, morphology.disk(ring_width))
    mask_eroded = morphology.binary_erosion(mask_np, morphology.disk(ring_width))
    
    # Outer ring (just outside mask)
    outer_ring = mask_dilated & ~mask_np
    # Inner ring (just inside mask)
    inner_ring = mask_np & ~mask_eroded
    
    if not outer_ring.any() or not inner_ring.any():
        return 0.0  # Can't compute boundary
    
    # Compute density in each ring
    outer_density = pred_np[outer_ring].mean()
    inner_density = pred_np[inner_ring].mean()
    
    # Absolute difference = boundary jump
    boundary_jump = abs(inner_density - outer_density)
    
    return float(boundary_jump)


def compute_size_distribution_mismatch(
    pred_sizes: np.ndarray,
    true_sizes: np.ndarray
) -> float:
    """
    Compute Wasserstein distance between predicted and true building size distributions.
    Measures how different the distributions are (0 = identical).
    
    Args:
        pred_sizes: Array of predicted building areas
        true_sizes: Array of true building areas
        
    Returns:
        Wasserstein distance (Earth Mover's Distance) - lower is better
    """
    # Handle empty arrays
    if len(pred_sizes) == 0 or len(true_sizes) == 0:
        if len(pred_sizes) == 0 and len(true_sizes) == 0:
            return 0.0
        return 1.0  # Max mismatch
    
    # Normalize to [0, 1] for comparable scale
    all_sizes = np.concatenate([pred_sizes, true_sizes])
    max_size = all_sizes.max()
    
    if max_size == 0:
        return 0.0
    
    pred_norm = pred_sizes / max_size
    true_norm = true_sizes / max_size
    
    # Compute Wasserstein distance
    distance = wasserstein_distance(pred_norm, true_norm)
    
    return float(distance)


def compute_building_metrics(
    pred_buildings: torch.Tensor,
    true_buildings: torch.Tensor,
    mask: torch.Tensor,
    min_building_size: int = 4,
    boundary_ring_width: int = 3
) -> Dict[str, float]:
    """
    Compute comprehensive building quality metrics for inpainting evaluation.
    
    Use during periodic sampling to track training progress:
    - IoU/Dice: Pixel-level accuracy
    - Edge smoothness: Shape quality (detect jagged artifacts)
    - Size distribution: Realism of building scales
    - Boundary consistency: Seam detection at inpaint edges
    - Aspect ratios: Shape realism (buildings should be rectangular-ish)
    
    Args:
        pred_buildings: Predicted building mask [B, 1, H, W]
        true_buildings: Ground truth building mask [B, 1, H, W]
        mask: Inpainting mask [B, 1, H, W]
        min_building_size: Minimum building area (pixels) to consider
        boundary_ring_width: Width of boundary ring for seam detection
        
    Returns:
        Dictionary of metrics:
            - iou: Intersection over Union inside mask [0, 1]
            - dice: Dice coefficient inside mask [0, 1]
            - edge_smoothness: Edge quality [0, 1] (higher = smoother)
            - size_mismatch: Distribution distance [0, ~1] (lower = better match)
            - boundary_jump: Density discontinuity at seam [0, ~1] (lower = better)
            - aspect_ratio_mean: Mean building aspect ratio
            - aspect_ratio_realistic: Boolean flag for realistic ratios
            - num_buildings_pred: Count of predicted buildings
            - num_buildings_true: Count of true buildings
    """
    # Ensure consistent shape
    if pred_buildings.shape != mask.shape:
        mask = F.interpolate(mask, size=pred_buildings.shape[-2:], mode='nearest')
    
    # 1. IoU and Dice inside inpainted region
    iou = compute_iou(pred_buildings, true_buildings, mask)
    dice = compute_dice(pred_buildings, true_buildings, mask)
    
    # 2. Edge smoothness (detect jagged edges)
    pred_edges = sobel_edge_detection(pred_buildings)
    edge_smoothness = compute_edge_smoothness(pred_edges)
    
    # 3. Building size distribution
    pred_sizes = compute_building_areas(pred_buildings, min_size=min_building_size)
    true_sizes = compute_building_areas(true_buildings, min_size=min_building_size)
    size_mismatch = compute_size_distribution_mismatch(pred_sizes, true_sizes)
    
    # 4. Boundary consistency (context matching at seam)
    boundary_jump = compute_boundary_consistency(
        pred_buildings,
        mask,
        ring_width=boundary_ring_width
    )
    
    # 5. Building aspect ratio distribution
    pred_ratios = compute_aspect_ratios(pred_buildings, min_size=min_building_size)
    aspect_mean = pred_ratios.mean()
    
    # Flag unrealistic aspect ratios (very elongated or square blobs)
    # Realistic buildings: 0.3 < ratio < 3.0 (roughly)
    realistic_count = np.sum((pred_ratios > 0.3) & (pred_ratios < 3.0))
    aspect_realistic = realistic_count / len(pred_ratios) if len(pred_ratios) > 0 else 1.0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'edge_smoothness': float(edge_smoothness),
        'size_mismatch': float(size_mismatch),
        'boundary_jump': float(boundary_jump),
        'aspect_ratio_mean': float(aspect_mean),
        'aspect_ratio_realistic': float(aspect_realistic),
        'num_buildings_pred': int(len(pred_sizes)),
        'num_buildings_true': int(len(true_sizes)),
    }


def aggregate_metrics_batch(
    pred_buildings_batch: torch.Tensor,
    true_buildings_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    min_building_size: int = 4
) -> Dict[str, float]:
    """
    Compute building metrics averaged over a batch of samples.
    
    Args:
        pred_buildings_batch: Predicted buildings [B, 1, H, W]
        true_buildings_batch: Ground truth buildings [B, 1, H, W]
        mask_batch: Inpainting masks [B, 1, H, W]
        min_building_size: Minimum building size threshold
        
    Returns:
        Dictionary of averaged metrics across batch
    """
    batch_size = pred_buildings_batch.shape[0]
    
    # Accumulate metrics
    metrics_list = []
    for i in range(batch_size):
        metrics = compute_building_metrics(
            pred_buildings_batch[i:i+1],
            true_buildings_batch[i:i+1],
            mask_batch[i:i+1],
            min_building_size=min_building_size
        )
        metrics_list.append(metrics)
    
    # Average across batch
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = float(np.mean(values))
    
    return avg_metrics


def print_metrics_summary(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Pretty print metrics summary for logging.
    
    Args:
        metrics: Metrics dictionary from compute_building_metrics
        prefix: Optional prefix for print output
    """
    print(f"\n{prefix}Building Quality Metrics:")
    print(f"  IoU:                {metrics['iou']:.4f}")
    print(f"  Dice:               {metrics['dice']:.4f}")
    print(f"  Edge Smoothness:    {metrics['edge_smoothness']:.4f} (higher = smoother)")
    print(f"  Size Mismatch:      {metrics['size_mismatch']:.4f} (lower = better)")
    print(f"  Boundary Jump:      {metrics['boundary_jump']:.4f} (lower = better seam)")
    print(f"  Aspect Ratio:       {metrics['aspect_ratio_mean']:.2f} ({metrics['aspect_ratio_realistic']*100:.1f}% realistic)")
    print(f"  Building Count:     {metrics['num_buildings_pred']} pred / {metrics['num_buildings_true']} true")

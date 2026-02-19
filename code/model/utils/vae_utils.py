"""
==============================================================================
Utility functions for VAE training and inference.
==============================================================================
"""
###### import libraries ######
# Standard libraries
import os
from typing import Dict, List

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

# Local imports
from model.utils.layer_config import is_binary_layer, is_categorical_layer, get_layer_dice_config
from model.utils.colors import get_colormap_for_layer, apply_colormap_to_tensor
from model.utils.samples import save_layerwise_comparisons, save_rgb_comparison


def save_vae_reconstruction_samples(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    channel_names: List[str],
    layer_names: List[str],
    layers_registry: Dict,
    save_dir: str,
    step: int,
    n_samples: int = 8,
    save_rgb_composite: bool = True
) -> None:
    """
    Save VAE reconstruction samples with layer-aware visualization.
    
    This function handles visualization for different layer types using
    normalization settings from the layer registry:
    - Binary layers: Applies sigmoid to logits for visualization
    - RGB/continuous layers: Uses layer-specific normalization config
      (percentile, minmax, clip, custom)
    
    Args:
        input_tensor: Input tensor [B, C, H, W]
        recon_tensor: Reconstructed tensor [B, C, H, W] (logits for binary, values for continuous)
        channel_names: List of channel names (e.g., ['rgb:red', 'rgb:green', 'rgb:blue'])
        layer_names: List of layer names for each channel (e.g., ['rgb', 'rgb', 'rgb'])
        layers_registry: Global layers configuration dict with normalization settings
        save_dir: Directory to save visualization images
        step: Training step number for filename
        n_samples: Number of samples to visualize (default: 8)
        save_rgb_composite: Whether to save RGB composite if RGB layers present (default: True)
        
    Example:
        >>> save_vae_reconstruction_samples(
        ...     input_tensor=input_tensor,
        ...     recon_tensor=recon,
        ...     channel_names=['rgb:red', 'rgb:green', 'rgb:blue'],
        ...     layer_names=['rgb', 'rgb', 'rgb'],
        ...     layers_registry=config['layers'],
        ...     save_dir='./samples',
        ...     step=1000
        ...     )
    """
    
    # Use unified comparison visualization
    save_layerwise_comparisons(
        input_tensor=input_tensor,
        recon_tensor=recon_tensor,
        channel_names=channel_names,
        layer_names=layer_names,
        layers_registry=layers_registry,
        save_dir=save_dir,
        filename_prefix=f'recon_step_{step}',
        n_samples=n_samples,
        use_colormaps=True
    )
    
    # Save RGB composite if present
    if save_rgb_composite:
        rgb_save_path = os.path.join(save_dir, f'recon_step_{step}_RGB_composite.png')
        save_rgb_comparison(
            input_tensor=input_tensor,
            recon_tensor=recon_tensor,
            layer_names=layer_names,
            save_path=rgb_save_path,
            n_samples=n_samples,
            normalize_per_channel=True
        )



def compute_reconstruction_loss(
    recon, target, channel_names, layer_names,
    layers_registry,
    binary_weight=1.0, continuous_weight=1.0, categorical_weight=1.0,
    layer_dice_config=None, posw_ema=None,
    all_channels_tensor=None,  # Full tensor with all channels for mask lookup
    layer_weights=None,  # Per-layer weight overrides
    pixel_weights=None,  # Optional [B, 1, H, W] spatial weighting (e.g., inpainting mask focus)
):
    """
    Compute reconstruction loss for any tensor using dynamic layer configuration.
    
    Handles:
    - Binary layers: BCE with logits + optional Dice loss
    - Categorical layers: Cross-entropy loss (logits vs one-hot targets)
    - Continuous layers: MSE/L1/Smooth-L1 loss (configurable per layer)
    - RGB layers: MSE/L1 loss (perceptual loss handled externally)
    - Masked layers: Loss weighted by mask layer (e.g., buildings_heights masked by buildings)
    
    Args:
        recon: Reconstructed tensor [B, C, H, W] (logits for binary/categorical, values for continuous)
        target: Target tensor [B, C, H, W] (one-hot for categorical)
        channel_names: List of channel names (e.g., ['rgb:red', 'buildings', 'building_shapes:class_0'])
        layer_names: List of layer names for each channel (e.g., ['rgb', 'buildings', 'building_shapes'])
        layers_registry: Global layers configuration dict
        binary_weight: Weight for binary channel losses (default weight)
        continuous_weight: Weight for continuous channel losses (default weight)
        categorical_weight: Weight for categorical channel losses (default weight)
        layer_dice_config: Optional dict mapping layer names to dice config overrides
        posw_ema: Optional PosWeightEMA tracker for stable class weighting (indexed by binary channel index)
        all_channels_tensor: Full input/target tensor with all channels (for mask layer lookup)
        layer_weights: Optional dict mapping layer names to custom weights (overrides defaults)
        pixel_weights: Optional [B, 1, H, W] per-pixel spatial weights for inpainting.
                       When provided, all per-pixel losses are weighted by this tensor,
                       focusing gradients on the masked (generation) region.
                       Pass None for standard VAE training (uniform weighting).
        
    Layer Config Options:
        - loss_type: 'mse' (default), 'l1', 'smooth_l1', or 'cross_entropy' (categorical)
          * 'mse': Better convergence, standard for VAE
          * 'l1': Better edge preservation, robust to outliers (recommended for RGB)
          * 'smooth_l1': Hybrid approach, robust to outliers with smooth gradients
          * 'cross_entropy': For categorical multi-class layers
        - mask_layer: Name of binary layer to use as loss mask (e.g., 'buildings' for 'buildings_heights')
        - mask_loss_weight: Weight boost for masked regions (default: 1.0, recommend 3.0-5.0)
        
    Returns:
        Dictionary with losses per channel type, total loss tensor
    """
    
    losses = {}
    binary_loss = 0.0
    continuous_loss = 0.0
    categorical_loss = 0.0
    
    binary_count = 0
    continuous_count = 0
    categorical_count = 0
    binary_ch_idx = 0  # Separate index for binary channels only
    
    # Build set of categorical layers and their channel ranges to skip individual processing
    # Categorical layers span multiple channels (num_classes) and must be processed as a group
    categorical_processed = set()
    
    # Spatial weighting for inpainting (when pixel_weights is provided)
    use_pw = pixel_weights is not None
    
    def _pw_reduce(per_pixel_loss):
        """Reduce per-pixel loss with spatial weights (inpainting mask focus)."""
        pw = pixel_weights
        # Match dimensions: pixel_weights is [B,1,H,W], CE output is [B,H,W]
        while pw.dim() > per_pixel_loss.dim():
            pw = pw.squeeze(1)
        return (per_pixel_loss * pw).sum() / pw.sum().clamp(min=1.0)
    
    idx = 0
    while idx < len(channel_names):
        channel_name = channel_names[idx]
        layer_name = layer_names[idx]
        
        # Get layer configuration
        layer_info = layers_registry.get(layer_name, {})
        is_binary = is_binary_layer(layer_info)
        is_categorical = is_categorical_layer(layer_info)
        
        # === CATEGORICAL LAYERS: process all num_classes channels at once ===
        if is_categorical and layer_name not in categorical_processed:
            categorical_processed.add(layer_name)
            num_classes = layer_info.get('num_classes', 1)
            
            # Extract all channels for this categorical layer
            recon_cat = recon[:, idx:idx+num_classes, :, :]  # [B, num_classes, H, W] logits
            target_cat = target[:, idx:idx+num_classes, :, :]  # [B, num_classes, H, W] one-hot
            
            # Convert one-hot target to class indices for cross-entropy
            target_indices = target_cat.argmax(dim=1)  # [B, H, W] long
            
            # Compute cross-entropy loss
            if use_pw:
                ce_per_px = F.cross_entropy(recon_cat, target_indices, reduction='none')  # [B, H, W]
                loss = _pw_reduce(ce_per_px)
            else:
                loss = F.cross_entropy(recon_cat, target_indices, reduction='mean')
            losses[f'{layer_name}_ce'] = loss.item()
            
            # Apply per-layer weight if specified, otherwise use default
            layer_weight = layer_weights.get(layer_name, categorical_weight) if layer_weights else categorical_weight
            categorical_loss += loss * layer_weight
            categorical_count += 1
            
            # Skip ahead past all channels of this categorical layer
            idx += num_classes
            continue
        
        # Skip channels that belong to an already-processed categorical layer
        if is_categorical and layer_name in categorical_processed:
            idx += 1
            continue
        
        recon_ch = recon[:, idx:idx+1, :, :]
        target_ch = target[:, idx:idx+1, :, :]
        
        # === BINARY LAYERS: BCE with logits + optional Dice loss ===
        if is_binary:
            # Clamp target to valid range (recon_ch is logits, no clamping)
            target_ch = target_ch.clamp(0.0, 1.0)
            
            # Compute BCE with logits and class-imbalance weighting
            if posw_ema is not None:
                pw = posw_ema.update(binary_ch_idx, target_ch)
                if use_pw:
                    bce_per_px = F.binary_cross_entropy_with_logits(
                        recon_ch, target_ch, pos_weight=pw, reduction='none'
                    )
                    bce = _pw_reduce(bce_per_px)
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        recon_ch, target_ch, pos_weight=pw, reduction='mean'
                    )
            else:
                if use_pw:
                    # Compute pos_weight manually, then do per-pixel BCE
                    pos = target_ch.mean().clamp(1e-6, 1 - 1e-6)
                    pw = ((1 - pos) / pos).detach()
                    bce_per_px = F.binary_cross_entropy_with_logits(
                        recon_ch, target_ch, pos_weight=pw, reduction='none'
                    )
                    bce = _pw_reduce(bce_per_px)
                else:
                    bce = bce_with_logits_pos_weight(recon_ch, target_ch)
            
            # Get per-layer dice configuration
            dice_config = get_layer_dice_config(layers_registry, layer_name)
            # Override with training config if provided
            if layer_dice_config and layer_name in layer_dice_config:
                dice_config.update(layer_dice_config[layer_name])
            
            use_dice = dice_config.get('use_dice', False)
            dice_weight = dice_config.get('weight', 0.5)
            
            # Compute Dice loss if enabled for this layer
            if use_dice:
                # When spatially weighted, compute dice only within the focus region
                dice_mask = (pixel_weights > 0.5) if use_pw else None
                dice = dice_loss_from_logits(recon_ch, target_ch, spatial_mask=dice_mask)
                loss = bce + dice_weight * dice
                losses[f'{channel_name}_dice'] = dice.item()
            else:
                loss = bce
            
            losses[f'{channel_name}_bce'] = bce.item()
            
            # Apply per-layer weight if specified, otherwise use default
            layer_weight = layer_weights.get(layer_name, binary_weight) if layer_weights else binary_weight
            binary_loss += loss * layer_weight
            binary_count += 1
            binary_ch_idx += 1  # Increment binary channel index
            
        # === CONTINUOUS LAYERS: MSE or L1 loss based on config ===
        else:
            # Get loss type from config (default: 'mse')
            # Options: 'mse' (L2), 'l1' (MAE), 'smooth_l1'
            loss_type = layer_info.get('loss_type', 'mse')
            
            # Check if this layer should be masked by another layer
            mask_layer_name = layer_info.get('mask_layer', None)
            mask_loss_weight = layer_info.get('mask_loss_weight', 3.0)  # Weight boost for masked regions
            
            if mask_layer_name and all_channels_tensor is not None:
                # Find mask layer index in channel_names
                try:
                    mask_idx = layer_names.index(mask_layer_name)
                    mask = all_channels_tensor[:, mask_idx:mask_idx+1, :, :]
                    
                    # Binarize mask (handle both binary logits and 0/1 values)
                    if layers_registry.get(mask_layer_name, {}).get('type') == 'binary':
                        mask = (torch.sigmoid(mask) > 0.5).float()
                    else:
                        mask = (mask > 0.5).float()
                    
                    # Compute loss with mask weighting (normalized by mask coverage)
                    if loss_type == 'l1':
                        loss_per_pixel = F.l1_loss(recon_ch, target_ch, reduction='none')
                    elif loss_type == 'smooth_l1':
                        loss_per_pixel = F.smooth_l1_loss(recon_ch, target_ch, reduction='none')
                    else:  # 'mse'
                        loss_per_pixel = F.mse_loss(recon_ch, target_ch, reduction='none')
                    
                    # Apply mask weighting (layer mask, e.g. buildings for heights)
                    layer_mask_weight = mask * mask_loss_weight + (1 - mask) * 1.0
                    # Combine layer mask with inpainting spatial weights if provided
                    if use_pw:
                        combined_weight = layer_mask_weight * pixel_weights
                    else:
                        combined_weight = layer_mask_weight
                    weighted_loss = loss_per_pixel * combined_weight
                    
                    # Normalize by mean weight to keep loss scale consistent across batches
                    # This prevents loss magnitude from varying with mask coverage percentage
                    mean_weight = combined_weight.mean() + 1e-8
                    loss = weighted_loss.sum() / (combined_weight.numel() * mean_weight)
                    
                    # Log mask coverage for debugging
                    mask_coverage = mask.mean().item()
                    losses[f'{channel_name}_mask_coverage'] = mask_coverage
                    
                except ValueError:
                    # Mask layer not found, fall back to standard loss
                    if loss_type == 'l1':
                        lpp = F.l1_loss(recon_ch, target_ch, reduction='none')
                    elif loss_type == 'smooth_l1':
                        lpp = F.smooth_l1_loss(recon_ch, target_ch, reduction='none')
                    else:
                        lpp = F.mse_loss(recon_ch, target_ch, reduction='none')
                    loss = _pw_reduce(lpp) if use_pw else lpp.mean()
            else:
                # No layer mask — standard loss (optionally with spatial weighting)
                if loss_type == 'l1':
                    lpp = F.l1_loss(recon_ch, target_ch, reduction='none')
                elif loss_type == 'smooth_l1':
                    lpp = F.smooth_l1_loss(recon_ch, target_ch, reduction='none')
                else:  # 'mse'
                    lpp = F.mse_loss(recon_ch, target_ch, reduction='none')
                loss = _pw_reduce(lpp) if use_pw else lpp.mean()
            
            # Log appropriate loss metric
            if loss_type == 'l1':
                losses[f'{channel_name}_l1'] = loss.item()
            elif loss_type == 'smooth_l1':
                losses[f'{channel_name}_smooth_l1'] = loss.item()
            else:
                losses[f'{channel_name}_mse'] = loss.item()
            
            # Apply per-layer weight if specified, otherwise use default
            layer_weight = layer_weights.get(layer_name, continuous_weight) if layer_weights else continuous_weight
            continuous_loss += loss * layer_weight
            continuous_count += 1
        
        idx += 1
    
    # Normalize by channel count
    if binary_count > 0:
        binary_loss = binary_loss / binary_count
    if continuous_count > 0:
        continuous_loss = continuous_loss / continuous_count
    if categorical_count > 0:
        categorical_loss = categorical_loss / categorical_count
    
    losses['binary_avg'] = binary_loss.item() if isinstance(binary_loss, torch.Tensor) else binary_loss
    losses['continuous_avg'] = continuous_loss.item() if isinstance(continuous_loss, torch.Tensor) else continuous_loss
    losses['categorical_avg'] = categorical_loss.item() if isinstance(categorical_loss, torch.Tensor) else categorical_loss
    
    total = binary_loss + continuous_loss + categorical_loss
    losses['total_recon'] = total.item() if isinstance(total, torch.Tensor) else 0.0
    
    return losses, total


def bce_with_logits_pos_weight(logits, targets, pos_weight=None, eps=1e-6):
    """
    Binary cross-entropy with logits and class-imbalance weighting.
    
    Args:
        logits: [B,1,H,W] raw decoder output (unbounded)
        targets: [B,1,H,W] in {0,1}
        pos_weight: Optional pre-computed positive weight (scalar or tensor)
        eps: Small epsilon for numerical stability
        
    Returns:
        BCE loss with logits and optional positive weighting
    """
    targets = targets.clamp(0.0, 1.0)
    
    if pos_weight is None:
        # Compute per-batch positive weight
        pos = targets.mean().clamp(eps, 1 - eps)
        pos_weight = ((1 - pos) / pos).detach()  # scalar
    
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='mean'
    )


def dice_loss_from_logits(logits, targets, eps=1e-6, spatial_mask=None):
    """
    Dice loss for thin structures (streets, vegetation edges).
    Applies sigmoid to logits before computing overlap.
    
    Args:
        logits: [B,1,H,W] raw decoder output
        targets: [B,1,H,W] in {0,1}
        eps: Small epsilon for numerical stability
        spatial_mask: Optional [B,1,H,W] boolean mask to restrict computation
                      (e.g., only compute dice inside inpainting mask)
        
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    targets = targets.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    
    if spatial_mask is not None:
        probs = probs * spatial_mask.float()
        targets = targets * spatial_mask.float()
    
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1 - (2 * intersection + eps) / (union + eps)
    
    return dice.mean()


class PosWeightEMA:
    """
    Exponential moving average tracker for positive weights per channel.
    Stabilizes class-imbalance weighting across small batches.
    """
    def __init__(self, num_channels, momentum=0.95, init=1.0, device='cpu'):
        self.m = momentum
        self.val = torch.full((num_channels,), float(init), device=device)
    
    def update(self, ch_idx, targets, eps=1e-6):
        """
        Update EMA for a specific channel.
        
        Args:
            ch_idx: Channel index
            targets: Target tensor for this channel [B,1,H,W]
            eps: Small epsilon for numerical stability
            
        Returns:
            Updated positive weight for this channel
        """
        with torch.no_grad():  # Prevent gradients from flowing through EMA update
            pos = targets.mean().clamp(eps, 1 - eps)
            pw = ((1 - pos) / pos)
            self.val[ch_idx] = self.m * self.val[ch_idx] + (1 - self.m) * pw
        return self.val[ch_idx].detach().clone()  # Return detached value as snapshot that won't change later

def get_kl_weight(epoch: int, kl_config: dict) -> float:
    """
    Compute KL weight with optional linear annealing.
    
    Args:
        epoch: Current epoch index (0-based)
        kl_config: Dict with keys: enabled, start_weight, end_weight, warmup_epochs
        
    Returns:
        KL weight for current epoch
    """
    if not kl_config.get('enabled', False):
        return kl_config.get('end_weight', 0.001)
    
    start_w = kl_config.get('start_weight', 0.0001)
    end_w = kl_config.get('end_weight', 0.001)
    warmup = kl_config.get('warmup_epochs', 20)
    
    if warmup <= 0 or epoch >= warmup:
        return end_w
    
    # Linear interpolation
    return start_w + (end_w - start_w) * (epoch / warmup)
"""
Utility functions for VAE training and inference.
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
from model.utils.layer_config import is_binary_layer, get_layer_dice_config
from model.utils.colors import get_colormap_for_layer, apply_colormap_to_tensor


def save_vae_reconstruction_samples(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
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
        layer_names: List of layer names corresponding to channels
        layers_registry: Global layers configuration dict with normalization settings
        save_dir: Directory to save visualization images
        step: Training step number for filename
        n_samples: Number of samples to visualize (default: 8)
        save_rgb_composite: Whether to save RGB composite if RGB layers present (default: True)
        
    Example:
        >>> save_vae_reconstruction_samples(
        ...     input_tensor=input_tensor,
        ...     recon_tensor=recon,
        ...     layer_names=['buildings', 'streets', 'lst'],
        ...     layers_registry=config['layers'],
        ...     save_dir='./samples',
        ...     step=1000
        ... )
    """
    
    n_samples = min(n_samples, input_tensor.shape[0])
    
    # Visualize each layer separately based on layer registry
    vis_grids = []
    
    for ch_idx, layer_name in enumerate(layer_names):
        input_ch = input_tensor[:n_samples, ch_idx:ch_idx+1, :, :]
        recon_ch = recon_tensor[:n_samples, ch_idx:ch_idx+1, :, :]
        
        # Get layer properties from registry
        layer_info = layers_registry.get(layer_name, {})
        layer_type = layer_info.get('type', 'continuous')
        normalize_method = layer_info.get('normalize', 'minmax')
        
        # Determine visualization method based on layer type
        if layer_type == 'binary':
            # Binary channels: input is 0/1, recon is logits
            input_vis = torch.clamp(input_ch, 0, 1)
            recon_vis = torch.sigmoid(recon_ch)  # Apply sigmoid to logits
            
        else:
            # Continuous/RGB channels: Min-max normalization based on actual data range
            # VAE latent space may have different scale than input, so normalize independently
            # Check if mask_layer is specified (e.g., buildings_heights masked by buildings)
            mask_layer = layer_info.get('mask_layer', None)
            if mask_layer:
                # For masked layers, normalize only non-zero regions
                input_nonzero = input_ch[input_ch > 0]
                recon_nonzero = recon_ch[recon_ch > 0]
                if len(input_nonzero) > 0:
                    input_min, input_max = input_nonzero.min(), input_nonzero.max()
                    input_vis = (input_ch - input_min) / (input_max - input_min + 1e-8)
                    input_vis = torch.clamp(input_vis, 0, 1)
                else:
                    input_vis = input_ch
                
                if len(recon_nonzero) > 0:
                    recon_min, recon_max = recon_nonzero.min(), recon_nonzero.max()
                    recon_vis = (recon_ch - recon_min) / (recon_max - recon_min + 1e-8)
                    recon_vis = torch.clamp(recon_vis, 0, 1)
                else:
                    recon_vis = recon_ch
            else:
                # Regular min-max normalization (works for all: percentile, clip, custom, minmax)
                input_vis = (input_ch - input_ch.min()) / (input_ch.max() - input_ch.min() + 1e-8)
                recon_vis = (recon_ch - recon_ch.min()) / (recon_ch.max() - recon_ch.min() + 1e-8)
        
        # Create comparison for this layer
        comparison_ch = torch.cat([input_vis, recon_vis], dim=0)
        
        # Apply colormap for continuous layers (but NOT for RGB channels)
        # RGB channels should be visualized as grayscale individually, composite handled separately
        if layer_type != 'binary' and 'rgb' not in layer_name.lower():
            cmap = get_colormap_for_layer(layer_name)
            comparison_ch = apply_colormap_to_tensor(comparison_ch, cmap)
        
        grid_ch = make_grid(comparison_ch, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
        vis_grids.append(grid_ch)
    
    # Save each layer separately
    for ch_idx, layer_name in enumerate(layer_names):
        save_path = os.path.join(save_dir, f'recon_step_{step}_{layer_name.replace(":", "_")}.png')
        save_image(vis_grids[ch_idx], save_path)
    
    # Also save RGB composite if RGB layers are present
    if save_rgb_composite:
        rgb_indices = [i for i, name in enumerate(layer_names) if 'rgb' in name.lower()]
        print(f"[DEBUG] Found {len(rgb_indices)} RGB layers at indices: {rgb_indices}")
        print(f"[DEBUG] Layer names: {layer_names}")
        
        if len(rgb_indices) >= 3:
            try:
                # Extract the first 3 RGB channels using their indices
                rgb_input = input_tensor[:n_samples, rgb_indices[:3], :, :]
                rgb_recon = recon_tensor[:n_samples, rgb_indices[:3], :, :]
                
                print(f"[DEBUG] RGB input shape: {rgb_input.shape}, RGB recon shape: {rgb_recon.shape}")
                
                # Normalize RGB composite based on actual data range
                # VAE may produce different scale than input, normalize independently
                # Per-channel normalization for RGB (to preserve color balance)
                rgb_input_normalized = []
                rgb_recon_normalized = []
                for ch_idx in range(3):  # Process exactly 3 channels (R, G, B)
                    input_ch = rgb_input[:, ch_idx:ch_idx+1, :, :]
                    recon_ch = rgb_recon[:, ch_idx:ch_idx+1, :, :]
                    
                    # Min-max normalize each channel independently
                    input_norm = (input_ch - input_ch.min()) / (input_ch.max() - input_ch.min() + 1e-8)
                    recon_norm = (recon_ch - recon_ch.min()) / (recon_ch.max() - recon_ch.min() + 1e-8)
                    
                    rgb_input_normalized.append(input_norm)
                    rgb_recon_normalized.append(recon_norm)
                
                rgb_input = torch.cat(rgb_input_normalized, dim=1)
                rgb_recon = torch.cat(rgb_recon_normalized, dim=1)
                
                comparison_rgb = torch.cat([rgb_input, rgb_recon], dim=0)
                grid_rgb = make_grid(comparison_rgb, nrow=n_samples, padding=2, pad_value=1.0)
                
                save_path = os.path.join(save_dir, f'recon_step_{step}_RGB_composite.png')
                save_image(grid_rgb, save_path)
                print(f"[DEBUG] ✓ Saved RGB composite to: {save_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to save RGB composite: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[WARNING] Not enough RGB layers found ({len(rgb_indices)} < 3), skipping RGB composite")



def compute_reconstruction_loss(
    recon, target, channel_names, layer_names,
    layers_registry,
    binary_weight=1.0, continuous_weight=1.0, 
    layer_dice_config=None, posw_ema=None,
    all_channels_tensor=None  # Full tensor with all channels for mask lookup
):
    """
    Compute reconstruction loss for any tensor using dynamic layer configuration.
    
    Handles:
    - Binary layers: BCE with logits + optional Dice loss
    - Continuous layers: MSE/L1/Smooth-L1 loss (configurable per layer)
    - RGB layers: MSE/L1 loss (perceptual loss handled externally)
    - Masked layers: Loss weighted by mask layer (e.g., buildings_heights masked by buildings)
    
    Args:
        recon: Reconstructed tensor [B, C, H, W] (logits for binary channels, values for continuous)
        target: Target tensor [B, C, H, W]
        channel_names: List of channel names (e.g., ['rgb:red', 'buildings', 'lst'])
        layer_names: List of layer names for each channel (e.g., ['rgb', 'buildings', 'lst'])
        layers_registry: Global layers configuration dict
        binary_weight: Weight for binary channel losses
        continuous_weight: Weight for continuous channel losses
        layer_dice_config: Optional dict mapping layer names to dice config overrides
        posw_ema: Optional PosWeightEMA tracker for stable class weighting (indexed by binary channel index)
        all_channels_tensor: Full input/target tensor with all channels (for mask layer lookup)
        
    Layer Config Options:
        - loss_type: 'mse' (default), 'l1', or 'smooth_l1'
          * 'mse': Better convergence, standard for VAE
          * 'l1': Better edge preservation, robust to outliers (recommended for RGB)
          * 'smooth_l1': Hybrid approach, robust to outliers with smooth gradients
        - mask_layer: Name of binary layer to use as loss mask (e.g., 'buildings' for 'buildings_heights')
        - mask_loss_weight: Weight boost for masked regions (default: 1.0, recommend 3.0-5.0)
        
    Returns:
        Dictionary with losses per channel type, total loss tensor
    """
    
    losses = {}
    binary_loss = 0.0
    continuous_loss = 0.0
    
    binary_count = 0
    continuous_count = 0
    binary_ch_idx = 0  # Separate index for binary channels only
    
    for idx, (channel_name, layer_name) in enumerate(zip(channel_names, layer_names)):
        recon_ch = recon[:, idx:idx+1, :, :]
        target_ch = target[:, idx:idx+1, :, :]
        
        # Get layer configuration
        layer_info = layers_registry.get(layer_name, {})
        is_binary = is_binary_layer(layer_info)
        
        # Binary channels use BCE with logits + optional Dice loss
        if is_binary:
            # Clamp target to valid range (recon_ch is logits, no clamping)
            target_ch = target_ch.clamp(0.0, 1.0)
            
            # Compute BCE with logits and class-imbalance weighting
            if posw_ema is not None:
                pw = posw_ema.update(binary_ch_idx, target_ch)
                bce = F.binary_cross_entropy_with_logits(
                    recon_ch, target_ch, pos_weight=pw, reduction='mean'
                )
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
                dice = dice_loss_from_logits(recon_ch, target_ch)
                loss = bce + dice_weight * dice
                losses[f'{channel_name}_dice'] = dice.item()
            else:
                loss = bce
            
            losses[f'{channel_name}_bce'] = bce.item()
            binary_loss += loss * binary_weight
            binary_count += 1
            binary_ch_idx += 1  # Increment binary channel index
            
        # Continuous channels use MSE or L1 loss based on config
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
                    
                    # Compute loss with mask weighting (NORMALIZED by mask coverage)
                    if loss_type == 'l1':
                        loss_per_pixel = F.l1_loss(recon_ch, target_ch, reduction='none')
                    elif loss_type == 'smooth_l1':
                        loss_per_pixel = F.smooth_l1_loss(recon_ch, target_ch, reduction='none')
                    else:  # 'mse'
                        loss_per_pixel = F.mse_loss(recon_ch, target_ch, reduction='none')
                    
                    # Apply mask weighting
                    mask_weight = mask * mask_loss_weight + (1 - mask) * 1.0
                    weighted_loss = loss_per_pixel * mask_weight
                    
                    # Normalize by mean weight to keep loss scale consistent across batches
                    # This prevents loss magnitude from varying with mask coverage percentage
                    mean_weight = mask_weight.mean() + 1e-8
                    loss = weighted_loss.sum() / (mask_weight.numel() * mean_weight)
                    
                    # Log mask coverage for debugging
                    mask_coverage = mask.mean().item()
                    losses[f'{channel_name}_mask_coverage'] = mask_coverage
                    
                except ValueError:
                    # Mask layer not found, fall back to unmasked loss
                    if loss_type == 'l1':
                        loss = F.l1_loss(recon_ch, target_ch, reduction='mean')
                    elif loss_type == 'smooth_l1':
                        loss = F.smooth_l1_loss(recon_ch, target_ch, reduction='mean')
                    else:
                        loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            else:
                # No masking - standard loss
                if loss_type == 'l1':
                    loss = F.l1_loss(recon_ch, target_ch, reduction='mean')
                elif loss_type == 'smooth_l1':
                    loss = F.smooth_l1_loss(recon_ch, target_ch, reduction='mean')
                else:  # 'mse'
                    loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            
            # Log appropriate loss metric
            if loss_type == 'l1':
                losses[f'{channel_name}_l1'] = loss.item()
            elif loss_type == 'smooth_l1':
                losses[f'{channel_name}_smooth_l1'] = loss.item()
            else:
                losses[f'{channel_name}_mse'] = loss.item()
            
            continuous_loss += loss * continuous_weight
            continuous_count += 1
    
    # Normalize by channel count
    if binary_count > 0:
        binary_loss = binary_loss / binary_count
    if continuous_count > 0:
        continuous_loss = continuous_loss / continuous_count
    
    losses['binary_avg'] = binary_loss.item() if isinstance(binary_loss, torch.Tensor) else binary_loss
    losses['continuous_avg'] = continuous_loss.item() if isinstance(continuous_loss, torch.Tensor) else continuous_loss
    losses['total_recon'] = (binary_loss + continuous_loss).item() if isinstance(binary_loss + continuous_loss, torch.Tensor) else 0.0
    
    return losses, binary_loss + continuous_loss


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


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice loss for thin structures (streets, vegetation edges).
    Applies sigmoid to logits before computing overlap.
    
    Args:
        logits: [B,1,H,W] raw decoder output
        targets: [B,1,H,W] in {0,1}
        eps: Small epsilon for numerical stability
        
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    targets = targets.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    
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


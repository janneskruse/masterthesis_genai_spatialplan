# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils
### import libraries ######
# Standard libraries
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

# Data Handling
import numpy as np

# Data Science/ML libraries
import torch
import torch.nn.functional as F


def mask_conditioning_latents(cond_dict: dict, mask_latent: torch.Tensor, mask_groups: list) -> dict:
    """
    Apply inpainting mask to specified conditioning latent groups.
    
    Args:
        cond_dict: Conditioning dictionary containing latent groups
        mask_latent: Binary mask in latent space [1, H, W] or [B, 1, H, W]
                    1 = inpaint region (zero out), 0 = keep
        mask_groups: List of group names to mask (e.g., ['environmental', 'semantic'])
        
    Returns:
        Modified conditioning dict with masked groups
    """
    for group_name in mask_groups:
        if group_name in cond_dict:
            # Zero out masked regions: multiply by (1 - mask)
            # mask_latent: 1=inpaint, 0=keep → (1-mask): 0=inpaint, 1=keep
            cond_dict[group_name] = cond_dict[group_name] * (1 - mask_latent)
    
    return cond_dict


def make_uncond_input_keep_mask(cond_input: dict) -> dict:
    """
    Create unconditional conditioning for CFG that KEEPS ONLY the inpainting mask.
    
    CRITICAL FOR INPAINTING CFG:
    - Unconditional branch must see ONLY the inpainting mask from pixel-space
    - All other pixel-space channels (LST, NDVI, etc.) are zeroed
    - All latent-space conditioning groups are zeroed
    - Otherwise CFG compares "masked denoising" vs "unmasked denoising" → invalid guidance
    
    This matches the training behavior of apply_classifier_free_guidance_dropout(keep_mask=True).
    
    Args:
        cond_input: Conditioning dictionary with 'image' (pixel-space), latent groups, and 'meta'
        
    Returns:
        Unconditional dict: keeps meta and ONLY inpainting_mask channel, zeros everything else
    """
    uncond = {}
    for k, v in cond_input.items():
        if k == 'meta':
            # Keep metadata as-is
            uncond[k] = v
        elif k == 'image':
            # CRITICAL: Keep ONLY the inpainting_mask channel, zero all other pixel-space
            if 'meta' in cond_input:
                # Handle both dict and list-of-dicts meta structures
                meta = cond_input['meta']
                if isinstance(meta, list):
                    meta = meta[0]
                
                pixel_names = meta.get('pixel_space_names', [])
                
                if 'inpainting_mask' in pixel_names:
                    # Zero out all channels except inpainting_mask
                    mask_idx = pixel_names.index('inpainting_mask')
                    uncond_image = torch.zeros_like(v)
                    uncond_image[:, mask_idx:mask_idx+1, :, :] = v[:, mask_idx:mask_idx+1, :, :]
                    uncond[k] = uncond_image
                else:
                    # No inpainting_mask found - zero everything
                    uncond[k] = torch.zeros_like(v)
            else:
                # No metadata - zero everything
                uncond[k] = torch.zeros_like(v)
        else:
            # Zero latent-space conditioning groups
            if isinstance(v, torch.Tensor):
                uncond[k] = torch.zeros_like(v)
            else:
                uncond[k] = v
    return uncond


def apply_classifier_free_guidance_dropout(
    cond_dict: dict,
    drop_prob: float,
    drop_groups: list,
    keep_mask: bool = True,
    tmax_uncond_value: float = 0.0
) -> dict:
    """
    Apply classifier-free guidance dropout to conditioning.
    Randomly zeros out specified conditioning groups for CFG training.
    
    CRITICAL: For inpainting, inpainting_mask MUST be preserved (keep_mask=True)!
    
    Args:
        cond_dict: Conditioning dictionary with 'image' (pixel-space) and latent groups
        drop_prob: Probability of dropping conditioning (e.g., 0.1 = 10% chance)
        drop_groups: List of latent-space group names to drop (e.g., ['semantic', 'environmental'])
        keep_mask: If True, preserve 'inpainting_mask' channel in pixel-space (default: True)
                  All other pixel-space channels will be dropped
        tmax_uncond_value: Unconditional value for temperature control (default: 0.0)
                          Should match temperature_control.training.unconditional_value in config
        
    Returns:
        NEW conditioning dict with randomly dropped groups (non-mutating)
        
    Note:
        Dropout is applied with a single random roll - either ALL specified groups
        are dropped together, or none are dropped. This maintains correlation
        between conditioning modalities.
        
        IMPORTANT: Returns a NEW dict to avoid mutating the original conditioning.
    """
    # Single random roll for all conditioning
    if np.random.rand() < drop_prob:
        # Create new dict to avoid mutation
        new_cond_dict = {}
        
        # Preserve metadata (never drop)
        if 'meta' in cond_dict:
            new_cond_dict['meta'] = cond_dict['meta']
        
        # Handle pixel-space conditioning: drop all except inpainting_mask
        if 'image' in cond_dict:
            if keep_mask and 'meta' in cond_dict and 'pixel_space_names' in cond_dict['meta']:
                # Selectively preserve inpainting_mask channel
                pixel_names = cond_dict['meta']['pixel_space_names']
                
                if 'inpainting_mask' in pixel_names:
                    # Zero out all channels except inpainting_mask
                    mask_idx = pixel_names.index('inpainting_mask')
                    dropped_image = torch.zeros_like(cond_dict['image'])
                    dropped_image[:, mask_idx:mask_idx+1, :, :] = cond_dict['image'][:, mask_idx:mask_idx+1, :, :]
                    new_cond_dict['image'] = dropped_image
                else:
                    # No mask found - drop everything
                    new_cond_dict['image'] = torch.zeros_like(cond_dict['image'])
            else:
                # Drop all pixel-space conditioning
                new_cond_dict['image'] = torch.zeros_like(cond_dict['image'])
        
        # Drop specified latent-space conditioning groups
        for key in cond_dict.keys():
            if key in ['image', 'meta']:
                continue  # Already handled
            
            if key == 'tmax':
                # Special handling for temperature control scalar
                # Use configured unconditional value (not always 0.0)
                new_cond_dict[key] = torch.full_like(cond_dict[key], tmax_uncond_value)
            elif key in drop_groups:
                # Zero out this group
                new_cond_dict[key] = torch.zeros_like(cond_dict[key])
            else:
                # Keep this group
                new_cond_dict[key] = cond_dict[key]
        
        return new_cond_dict
    else:
        # No dropout - return original dict unchanged
        return cond_dict


def load_latents(latent_path: str, prefix: str = None) -> List[str]:
    """
    Load pre-computed latents from disk to speed up LDM training.
    Returns list of file paths for lazy loading.
    
    Args:
        latent_path: Directory containing latent files
        prefix: Optional prefix to filter files (e.g., 'pred' for latent_pred_*.pt, 'cond' for latent_cond_*.pt)
        
    Returns:
        List of latent file paths sorted by index
    """
    latent_path = Path(latent_path)
    
    # Build pattern based on prefix
    if prefix:
        pattern = f'latent_{prefix}_*.pt'
        idx_position = 2  # e.g., latent_pred_123.pt -> split('_')[2] = '123'
    else:
        pattern = 'latent_*.pt'
        idx_position = 1  # e.g., latent_123.pt -> split('_')[1] = '123'
    
    # Try .pt files (recommended format)
    latent_files = sorted(
        latent_path.glob(pattern),
        key=lambda x: int(x.stem.split('_')[idx_position])
    )
    
    if latent_files:
        print(f"✓ Found {len(latent_files)} .pt latent files{' with prefix: ' + prefix if prefix else ''}")
        return [str(f) for f in latent_files]
    
    # Fall back to .pkl files (legacy format) - only if no prefix specified
    if not prefix:
        pkl_files = list(latent_path.glob('*.pkl'))
        if pkl_files:
            print(f"⚠ Found {len(pkl_files)} .pkl latent files (legacy format)")
            print(f"⚠ Consider converting to .pt format for better performance")
            print(f"⚠ Note: .pkl files will be fully loaded into memory by load_single_latent")
            
            # Return paths as strings, consistent with .pt behavior
            return [str(f) for f in sorted(pkl_files)]
    
    if prefix:
        print(f"❌ No latent files found with prefix '{prefix}' in {latent_path}")
    else:
        print(f"❌ No latent files found in {latent_path}")
    return []

def load_single_latent(latent_path: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """
    Load a single latent tensor from disk.
    
    Args:
        latent_path: Path to .pt or .pkl file
        device: Device to load tensor to (None = CPU)
        
    Returns:
        Loaded latent tensor, or None if file doesn't exist or loading fails
    """
    try:
        latent_path = Path(latent_path)
        
        # Check if file exists
        if not latent_path.exists():
            return None
        
        device_map = 'cpu' if device is None else device
        
        if latent_path.suffix == '.pt':
            return torch.load(latent_path, map_location=device_map, weights_only=False)
        elif latent_path.suffix == '.pkl':
            with open(latent_path, 'rb') as f:
                data = pickle.load(f)
                # Handle dict format from legacy pkl files
                if isinstance(data, dict):
                    # Get first tensor from dict values
                    tensor = next(iter(data.values()))[0]
                else:
                    tensor = data
                
                if device is not None and device != 'cpu':
                    tensor = tensor.to(device)
                return tensor
        else:
            print(f"⚠ Warning: Unsupported file format: {latent_path.suffix}")
            return None
    except Exception as e:
        print(f"⚠ Warning: Failed to load latent from {latent_path}: {e}")
        return None


def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0,
                                                                                       1) < text_drop_prob
        assert empty_text_embed is not None, ("Text Conditioning required as well as"
                                        " text dropping but empty text representation not created")
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition


# ==============================================================================
# SEAM IMPROVEMENT STRATEGIES FOR INPAINTING
# ==============================================================================

def create_boundary_ring(mask_latent: torch.Tensor, ring_width_px: int = 1) -> torch.Tensor:
    """
    Create a boundary ring around the mask for seam-aware loss weighting.
    
    This dilates the mask and subtracts the original to get a ring region
    around the hole boundary. Useful for emphasizing seam coherence during training.
    
    Args:
        mask_latent: Binary mask [B, 1, H, W] (1=inpaint, 0=keep)
        ring_width_px: Width of the ring in pixels (default: 1)
        
    Returns:
        Binary ring mask [B, 1, H, W] (1=ring region, 0=elsewhere)
        
    Example:
        ```
        mask:    [0 0 0 0]     ring:    [0 1 1 0]
                 [0 1 1 0]  →          [1 0 0 1]
                 [0 1 1 0]             [1 0 0 1]
                 [0 0 0 0]             [0 1 1 0]
        ```
    """
    kernel_size = 2 * ring_width_px + 1
    dilated = F.max_pool2d(
        mask_latent,
        kernel_size=kernel_size,
        stride=1,
        padding=ring_width_px
    )
    ring = torch.clamp(dilated - mask_latent, 0, 1)
    return ring


def feather_mask(mask: torch.Tensor, blur_radius: int = 3) -> torch.Tensor:
    """
    Blur mask edges for smooth transitions (mask feathering).
    
    Uses average pooling as a Gaussian blur approximation to create
    soft mask boundaries. Useful for seamless compositing in SD-like mode.
    
    Args:
        mask: Binary or soft mask [B, 1, H, W]
        blur_radius: Radius of blur kernel in pixels (default: 3)
        
    Returns:
        Feathered mask [B, 1, H, W] with soft boundaries
        
    Note:
        For sampling-time compositing:
        output = feathered_mask * generated + (1 - feathered_mask) * original
    """
    kernel_size = 2 * blur_radius + 1
    # Use avg_pool2d for efficient blurring
    mask_feathered = F.avg_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=blur_radius,
        count_include_pad=False
    )
    return mask_feathered


def compute_boundary_aware_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    mask_latent: torch.Tensor,
    loss_type: str,
    mask_loss_weight: float = 8.0,
    outside_weight: float = 0.0,
    use_boundary_ring: bool = False,
    ring_width_px: int = 1,
    ring_weight: float = 2.0
) -> torch.Tensor:
    """
    Enhanced loss computation with optional boundary ring emphasis.
    
    Adds extra weight to the boundary ring region to improve seam coherence
    without allowing edits outside the mask.
    
    Args:
        noise_pred: Predicted noise [B, C, H, W]
        noise: Target noise [B, C, H, W]
        mask_latent: Binary mask [B, 1, H, W] (1=inpaint, 0=keep)
        loss_type: "masked" (loss only inside mask) or "weighted" (weighted full-image)
        mask_loss_weight: Weight for masked region
        outside_weight: Weight for outside region (usually 0.0 for hard mode)
        use_boundary_ring: If True, add extra weight to boundary ring
        ring_width_px: Width of boundary ring in pixels
        ring_weight: Weight multiplier for boundary ring region
        
    Returns:
        Loss scalar
    """
    if loss_type == "masked":
        return F.mse_loss(noise_pred * mask_latent, noise * mask_latent)
    
    # Weighted full-image MSE
    per_pix = F.mse_loss(noise_pred, noise, reduction='none')
    
    if use_boundary_ring:
        # Create boundary ring for seam emphasis
        ring = create_boundary_ring(mask_latent, ring_width_px)
        
        # Weight map: outside + ring + inside
        # Ensure no overlap: outside * (1 - mask - ring) + ring * ring + mask * mask
        w = (outside_weight * (1.0 - mask_latent - ring) +
             ring_weight * ring +
             mask_loss_weight * mask_latent)
    else:
        # Standard weighting
        w = outside_weight * (1.0 - mask_latent) + mask_loss_weight * mask_latent
    
    return (per_pix * w).mean()


def apply_seam_mode(
    mode: Optional[str],
    **kwargs
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Apply seam improvement strategy for inpainting.
    
    This is the main dispatcher function that applies the specified seam mode.
    Different modes are used at different stages:
    - 'dilate': Training-time boundary ring loss (returns modified loss params)
    - 'feather': Sampling-time mask feathering (returns feathered mask)
    - 'repaint': Sampling-time reinjection (returns reinjection config)
    - None: No seam improvement (default)
    
    Args:
        mode: 'dilate', 'feather', 'repaint', or None
        **kwargs: Mode-specific parameters
        
    For 'dilate' mode (training):
        - mask_latent: Binary mask [B, 1, H, W]
        - ring_width_px: Ring width in pixels (default: 1)
        Returns: (ring_mask, {})
        
    For 'feather' mode (sampling):
        - mask: Mask to feather [B, 1, H, W]
        - blur_radius: Blur radius in pixels (default: 3)
        Returns: (feathered_mask, {})
        
    For 'repaint' mode (sampling):
        Returns: (None, repaint_config_dict)
        
    Returns:
        Tuple of (result_tensor, config_dict)
        - result_tensor: Modified mask or None
        - config_dict: Additional configuration for the mode
    """
    if mode is None or mode.lower() == 'none':
        return None, {}
    
    mode = mode.lower()
    
    if mode == 'dilate':
        # Training-time: create boundary ring for loss weighting
        mask_latent = kwargs.get('mask_latent')
        ring_width_px = kwargs.get('ring_width_px', 1)
        
        if mask_latent is None:
            raise ValueError("'dilate' mode requires 'mask_latent' argument")
        
        ring_mask = create_boundary_ring(mask_latent, ring_width_px)
        return ring_mask, {'ring_width_px': ring_width_px}
    
    elif mode == 'feather':
        # Sampling-time: feather mask edges for smooth compositing
        mask = kwargs.get('mask')
        blur_radius = kwargs.get('blur_radius', 3)
        
        if mask is None:
            raise ValueError("'feather' mode requires 'mask' argument")
        
        feathered_mask = feather_mask(mask, blur_radius)
        return feathered_mask, {'blur_radius': blur_radius}
    
    elif mode == 'repaint':
        # Sampling-time: RePaint-style reinjection config
        resample_steps = kwargs.get('resample_steps', 10)
        jump_length = kwargs.get('jump_length', 3)
        
        return None, {
            'resample_steps': resample_steps,
            'jump_length': jump_length
        }
    
    else:
        raise ValueError(
            f"Unknown seam mode: '{mode}'. "
            f"Valid modes: 'dilate', 'feather', 'repaint', None"
        )


def sample_with_repaint(
    model,
    scheduler,
    x0_latent: torch.Tensor,
    mask_latent: torch.Tensor,
    cond_input: dict,
    num_steps: int = 50,
    resample_steps: int = 10,
    jump_length: int = 3,
    device: torch.device = None
) -> torch.Tensor:
    """
    Sample with RePaint-style known-region reinjection.
    
    RePaint paper: https://arxiv.org/abs/2201.09865
    
    Key idea: At each denoising step, reinject the known region (outside mask)
    with properly noised original content. Optionally jump back in time for
    better harmonization between known and generated regions.
    
    Args:
        model: Diffusion U-Net model
        scheduler: Noise scheduler
        x0_latent: Original latent (clean) [B, C, H, W]
        mask_latent: Binary mask [B, 1, H, W] (1=inpaint, 0=keep)
        cond_input: Conditioning dictionary
        num_steps: Number of denoising steps
        resample_steps: Resample interval (jump back every N steps)
        jump_length: Number of timesteps to jump back
        device: Device for computation
        
    Returns:
        Inpainted latent [B, C, H, W]
        
    Note:
        This enforces hard constraints on the known region while allowing
        the model to generate coherent content that respects boundaries.
    """
    if device is None:
        device = x0_latent.device
    
    # Start from pure noise
    x_t = torch.randn_like(x0_latent)
    
    # Compute timestep schedule
    timestep_schedule = list(reversed(range(0, scheduler.num_timesteps, scheduler.num_timesteps // num_steps)))
    
    model.eval()
    with torch.no_grad():
        for step_idx, t in enumerate(timestep_schedule):
            # Create timestep tensor
            t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
            
            # Standard denoising step
            noise_pred = model(x_t, t_tensor, cond_input=cond_input)
            x_t_minus_1 = scheduler.sample_prev_timestep(x_t, noise_pred, torch.tensor([t], device=device))
            
            # Reinject known region (always)
            if t > 0:
                # Properly noise the known region to timestep t-1
                known_noise = torch.randn_like(x0_latent)
                t_prev = max(0, t - scheduler.num_timesteps // num_steps)
                t_prev_tensor = torch.full((x0_latent.shape[0],), t_prev, device=device, dtype=torch.long)
                known_noisy = scheduler.add_noise(x0_latent, known_noise, t_prev_tensor)
            else:
                # Last step: use clean original
                known_noisy = x0_latent
            
            # Composite: generated inside mask, known outside mask
            x_t_minus_1 = mask_latent * x_t_minus_1 + (1 - mask_latent) * known_noisy
            
            # Optional: resample/jump-back for better harmonization
            if resample_steps > 0 and step_idx % resample_steps == 0 and t > jump_length:
                # Jump back in time and resample
                jump_t = t - jump_length
                jump_noise = torch.randn_like(x_t_minus_1)
                jump_t_tensor = torch.full((x_t_minus_1.shape[0],), jump_t, device=device, dtype=torch.long)
                x_t = scheduler.add_noise(x_t_minus_1, jump_noise, jump_t_tensor)
            else:
                x_t = x_t_minus_1
    
    return x_t
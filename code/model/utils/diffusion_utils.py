# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils
### import libraries ######
# Standard libraries
import pickle
from pathlib import Path
from typing import List, Optional

# Data Handling
import numpy as np

# Data Science/ML libraries
import torch


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


def apply_classifier_free_guidance_dropout(
    cond_dict: dict,
    drop_prob: float,
    drop_groups: list,
    drop_pixel_space: bool = True
) -> dict:
    """
    Apply classifier-free guidance dropout to conditioning.
    Randomly zeros out specified conditioning groups for CFG training.
    
    Args:
        cond_dict: Conditioning dictionary with 'image' (pixel-space) and latent groups
        drop_prob: Probability of dropping conditioning (e.g., 0.1 = 10% chance)
        drop_groups: List of latent-space group names to drop (e.g., ['semantic', 'environmental'])
        drop_pixel_space: Whether to also drop pixel-space conditioning (default True)
        
    Returns:
        Modified conditioning dict with randomly dropped groups
        
    Note:
        Dropout is applied with a single random roll - either ALL specified groups
        are dropped together, or none are dropped. This maintains correlation
        between conditioning modalities.
    """
    # Single random roll for all conditioning
    if np.random.rand() < drop_prob:
        # Drop pixel-space conditioning
        if drop_pixel_space and 'image' in cond_dict:
            cond_dict['image'] = torch.zeros_like(cond_dict['image'])
        
        # Drop specified latent-space conditioning groups
        for group_name in drop_groups:
            if group_name in cond_dict:
                cond_dict[group_name] = torch.zeros_like(cond_dict[group_name])
    
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
            return torch.load(latent_path, map_location=device_map)
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
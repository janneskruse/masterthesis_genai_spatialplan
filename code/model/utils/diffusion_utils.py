# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils
### import libraries ######
# Standard libraries
import pickle
from pathlib import Path
from typing import List, Optional

# Data Science/ML libraries
import torch


def load_latents(latent_path: str) -> List[str]:
    """
    Load pre-computed latents from disk to speed up LDM training.
    Returns list of file paths for lazy loading.
    
    Args:
        latent_path: Directory containing latent files
        
    Returns:
        List of latent file paths sorted by index
    """
    latent_path = Path(latent_path)
    
    # Try .pt files (recommended format)
    latent_files = sorted(
        latent_path.glob('latent_*.pt'),
        key=lambda x: int(x.stem.split('_')[1])
    )
    
    if latent_files:
        print(f"✓ Found {len(latent_files)} .pt latent files")
        return [str(f) for f in latent_files]
    
    # Fall back to .pkl files (legacy format)
    pkl_files = list(latent_path.glob('*.pkl'))
    if pkl_files:
        print(f"⚠ Found {len(pkl_files)} .pkl latent files (legacy format)")
        print(f"⚠ Consider converting to .pt format for better performance")
        print(f"⚠ Note: .pkl files will be fully loaded into memory by load_single_latent")
        
        # Return paths as strings, consistent with .pt behavior
        return [str(f) for f in sorted(pkl_files)]
    
    print(f"❌ No latent files found in {latent_path}")
    return []

def load_single_latent(latent_path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Load a single latent tensor from disk.
    
    Args:
        latent_path: Path to .pt or .pkl file
        device: Device to load tensor to (None = CPU)
        
    Returns:
        Loaded latent tensor
    """
    latent_path = Path(latent_path)
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
        raise ValueError(f"Unsupported file format: {latent_path.suffix}. Use .pt or .pkl")


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
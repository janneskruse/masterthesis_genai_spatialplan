# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils
### import libraries ######
# Standard libraries
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
        
        import pickle
        latent_maps = {}
        for fname in pkl_files:
            with open(fname, 'rb') as f:
                s = pickle.load(f)
                for k, v in s.items():
                    latent_maps[k] = v[0]
        
        # Convert dict to sorted list
        return [latent_maps[i] for i in sorted(latent_maps.keys())]
    
    print(f"❌ No latent files found in {latent_path}")
    return []

def load_single_latent(latent_path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Load a single latent tensor from disk.
    
    Args:
        latent_path: Path to .pt file
        device: Device to load tensor to (None = CPU)
        
    Returns:
        Loaded latent tensor
    """
    if device is None:
        return torch.load(latent_path, map_location='cpu')
    else:
        return torch.load(latent_path, map_location=device)


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
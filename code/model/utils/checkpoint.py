# Utils for loading model checkpoints

###### import libraries ######
# system libraries
import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

# data science libraries
import torch


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu', is_main=False):
    """
    Load model checkpoint and optionally optimizer state.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        model: Model to load state into (can be DDP wrapped)
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
        is_main: Whether this is the main process (for logging)
        
    Returns:
        tuple: (start_epoch, checkpoint_dict) where:
            - start_epoch: Epoch number to resume from (0 if not found in checkpoint)
            - checkpoint_dict: Full checkpoint dictionary (None if file not found)
    """
    if not os.path.exists(checkpoint_path):
        if is_main:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return 0, None
    
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*50}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both dict format (with epoch info) and direct state_dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with training state
        model_state = checkpoint['model_state_dict']
        start_epoch = checkpoint.get('epoch', 0)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main:
                print("✓ Loaded optimizer state")
    else:
        # Legacy format: just model state dict
        model_state = checkpoint
        start_epoch = 0
    
    # Load into model (handle DDP wrapping)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    if is_main:
        print(f"✓ Loaded model state")
        if start_epoch > 0:
            print(f"✓ Resuming from epoch {start_epoch}")
        print(f"{'='*50}\n")
    
    return start_epoch, checkpoint


@dataclass
class ExistingPathsResult:
    """Container for resolved existing paths from config.
    
    Attributes:
        skip_training: Whether to skip training entirely (diffusion checkpoint exists)
        diffusion_checkpoint: Path to existing diffusion checkpoint (if found)
        latents_path: Path to existing latents directory (if found)
        vae_checkpoints: Dict mapping group_name -> checkpoint_path (if found)
        patches_path: Path to existing patches directory (if found)
        warnings: List of warning messages for paths specified but not found
    """
    skip_training: bool = False
    diffusion_checkpoint: Optional[str] = None
    latents_path: Optional[str] = None
    vae_checkpoints: Dict[str, str] = field(default_factory=dict)
    patches_path: Optional[str] = None
    latent_lst_predictor_checkpoint: Optional[str] = None
    warnings: list = field(default_factory=list)


def _is_valid_path_string(path_value: Any) -> bool:
    """Check if a path value is a valid non-None string."""
    if path_value is None:
        return False
    if not isinstance(path_value, str):
        return False
    if path_value.lower() == 'none' or path_value == '':
        return False
    return True


def check_existing_paths(
    train_config: Dict[str, Any],
    mode: str,
    type: str = 'default'
) -> ExistingPathsResult:
    """
    Check for existing paths specified in config and validate their existence.
    
    This function checks the `existing_paths` section of train_config for:
    - diffusion_checkpoints.<mode>: If exists, training should be skipped
    - latents.<mode>: Path to pre-computed latents
    - vae_checkpoints.<group_name>: Paths to VAE checkpoints
    - patches: Path to cached patches
    
    Args:
        train_config: The train_params section of the config
        mode: The training mode (e.g., 'semantic', 'satellite')
        type: The type of training (default: 'default'). Can be 'vae', 'diffusion', 'lst_latent', etc.
        
    Returns:
        ExistingPathsResult with resolved paths and skip_training flag
    """
    result = ExistingPathsResult()
    existing_paths = train_config.get('existing_paths', {})
    
    if type == 'diffusion':
        # Check for existing diffusion checkpoint (triggers skip_training)
        existing_diffusion_checkpoints = existing_paths.get('diffusion_checkpoints', {})
        diffusion_path_raw = existing_diffusion_checkpoints.get(mode, None)
        
        if _is_valid_path_string(diffusion_path_raw):
            if os.path.exists(diffusion_path_raw):
                result.skip_training = True
                result.diffusion_checkpoint = diffusion_path_raw
            else:
                result.warnings.append(
                    f"existing_paths.diffusion_checkpoints.{mode} specified but not found: {diffusion_path_raw}"
                )
    
    if type in ['diffusion', 'vae']:
        # Check for existing latents path
        existing_latents = existing_paths.get('latents', {})
        latents_path_raw = existing_latents.get(mode, None)
        
        if _is_valid_path_string(latents_path_raw):
            if os.path.exists(latents_path_raw):
                if type == 'vae':
                    result.skip_training = True
                    result.latents_path = latents_path_raw
                result.latents_path = latents_path_raw
            else:
                result.warnings.append(
                    f"existing_paths.latents.{mode} specified but not found: {latents_path_raw}"
                )
    
        # Check for existing VAE checkpoints
        existing_vae_checkpoints = existing_paths.get('vae_checkpoints', {})
        for group_name, vae_path_raw in existing_vae_checkpoints.items():
            if _is_valid_path_string(vae_path_raw):
                if os.path.exists(vae_path_raw):
                    if type == 'vae' and group_name == mode:
                        result.skip_training = True
                        result.vae_checkpoints[group_name] = vae_path_raw
                    
                    result.vae_checkpoints[group_name] = vae_path_raw
                else:
                    result.warnings.append(
                        f"existing_paths.vae_checkpoints.{group_name} specified but not found: {vae_path_raw}"
                    )
    
    existing_latent_lst_predictor_checkpoints = existing_paths.get('latent_lst_predictor_checkpoints', {})
    lst_latent_path_raw = existing_latent_lst_predictor_checkpoints.get(mode, None)
    if type == 'lst_latent':
        if _is_valid_path_string(lst_latent_path_raw):
            if os.path.exists(lst_latent_path_raw):
                result.skip_training = True
                result.latent_lst_predictor_checkpoint = lst_latent_path_raw
            else:
                result.warnings.append(
                    f"existing_paths.latent_lst_predictor_checkpoints.{mode} specified but not found: {lst_latent_path_raw}"
                )
    else:
        if _is_valid_path_string(lst_latent_path_raw):
            if os.path.exists(lst_latent_path_raw):
                result.latent_lst_predictor_checkpoint = lst_latent_path_raw
            else:
                result.warnings.append(
                    f"existing_paths.latent_lst_predictor_checkpoints.{mode} specified but not found: {lst_latent_path_raw}"
                )
    
    # Check for existing patches path
    patches_path_raw = existing_paths.get('patches', None)
    
    if _is_valid_path_string(patches_path_raw):
        if os.path.exists(patches_path_raw):
            result.patches_path = patches_path_raw
        else:
            result.warnings.append(
                f"existing_paths.patches specified but not found: {patches_path_raw}"
            )
    
    return result
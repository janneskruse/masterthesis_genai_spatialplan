# Utils for loading model checkpoints

###### import libraries ######
# system libraries
import os

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
        start_epoch: Epoch number to resume from (0 if not found in checkpoint)
    """
    if not os.path.exists(checkpoint_path):
        if is_main:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return 0
    
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*50}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    return start_epoch
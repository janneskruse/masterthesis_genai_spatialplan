# Unified Training script for VAE on urban data (DDP)
# Supports both semantic (Stage 1) and satellite (Stage 2) autoencoder training

###### import libraries ######
# Standard libraries
import os
import argparse
from pathlib import Path
import time
import yaml
import numpy as np
from tqdm import tqdm

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import save_image, make_grid

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.vae import VAE
from model.diffusion_blocks.discriminator import Discriminator
from model.diffusion_blocks.lpips import LPIPS
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.config_utils import get_prediction_channels, get_all_channels
from model.utils.layer_config import get_conditioning_layers, is_binary_layer, get_layer_info, get_layer_dice_config
from model.utils.vae_utils import save_vae_reconstruction_samples
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VAE for urban inpainting')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['semantic', 'satellite'],
        required=True,
        help='Training mode: semantic (Stage 1) or satellite (Stage 2)'
    )
    parser.add_argument(
        '--latent_type',
        type=str,
        choices=['prediction', 'conditioning'],
        default='prediction',
        help='Latent type to save: prediction (RGB/targets) or conditioning (OSM/env features)'
    )
    return parser.parse_args()


def save_latents_distributed(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    latent_dir: str,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    mode: str = 'satellite',
) -> int:
    """
    Save latent encodings from VAE in distributed setting.
    
    Each rank processes a disjoint subset of the dataset to avoid duplicates.
    Uses deterministic indexing to ensure consistent global indices across runs.
    
    Args:
        model: Trained VAE model (wrapped in DDP)
        dataset: Full training dataset
        latent_dir: Directory to save latent .pt files
        batch_size: Batch size for encoding
        rank: Current process rank
        world_size: Total number of processes
        device: Device for computation
        mode: VAE group name (e.g., 'satellite', 'semantic', 'environmental')
        
    Returns:
        Number of latents saved by this rank
    """
    
    # Unwrap DDP model for inference
    model_unwrapped = model.module if hasattr(model, 'module') else model
    model_unwrapped.eval()
    
    latent_dir = Path(latent_dir)
    
    # Create latent directory (only rank 0)
    if rank == 0:
        latent_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()  # Wait for directory creation
    
    # Calculate this rank's data indices
    total_samples = len(dataset)
    samples_per_rank = (total_samples + world_size - 1) // world_size  # Ceiling division
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    
    if rank == 0:
        print(f"\\n{'='*60}")
        print(f"Encoding and Saving {mode.upper()} Latents (Distributed) at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Samples per rank: {samples_per_rank}")
        print(f"World size: {world_size}")
    
    print(f"Rank {rank}: Processing indices {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    
    # Create subset of dataset for this rank
    rank_indices = list(range(start_idx, end_idx))
    rank_dataset = torch.utils.data.Subset(dataset, rank_indices)
    
    # Create dataloader for this rank's subset
    rank_loader = DataLoader(
        rank_dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain deterministic order
        num_workers=0,  # Avoid multiprocessing issues in DDP
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Encode and save latents
    latent_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(
            rank_loader,
            desc=f"Rank {rank} encoding",
            disable=(rank != 0),  # Only show progress on main rank
            unit="batch"
        )
        
        for batch_idx, data in enumerate(progress_bar):
            # Extract data
            if len(data) == 2:
                input_tensor, meta_dict = data
            else:
                input_tensor = data
            
            input_tensor = input_tensor.float().to(device)
            
            # Encode to latent space
            _, z, _, _ = model_unwrapped(input_tensor)
            
            # Save each latent with global index
            for i in range(z.shape[0]):
                # Calculate global index for this sample
                global_idx = start_idx + batch_idx * batch_size + i
                
                # Ensure we don't exceed dataset bounds
                if global_idx >= end_idx or global_idx >= total_samples:
                    break
                
                # Save latent to disk
                latent_path = latent_dir / f'latent_{global_idx}.pt'
                torch.save(z[i].cpu(), latent_path)
                latent_count += 1
    
    # Synchronize all ranks
    if world_size > 1:
        dist.barrier()
    
    # Verify completeness (rank 0 only)
    if rank == 0:
        pattern = 'latent_*.pt'
        
        saved_latents = sorted([
            int(f.stem.split('_')[1]) 
            for f in latent_dir.glob(pattern)
        ])
        
        expected_latents = list(range(total_samples))
        missing_latents = set(expected_latents) - set(saved_latents)
        duplicate_latents = len(saved_latents) - len(set(saved_latents))
        
        print(f"\\n{'='*60}")
        print(f"✓ Total latents saved: {len(saved_latents)}/{total_samples}")
        
        if missing_latents:
            print(f"⚠ Missing latents: {sorted(missing_latents)[:10]}{'...' if len(missing_latents) > 10 else ''}")
        
        if duplicate_latents > 0:
            print(f"⚠ Duplicate latents found: {duplicate_latents}")
        
        if len(saved_latents) == total_samples and not missing_latents and duplicate_latents == 0:
            print(f"✓ All {mode} latents saved successfully!")
        
        print(f"{'='*60}\n")
    
    return latent_count


def compute_reconstruction_loss(
    recon, target, channel_names, layer_names,
    layers_registry,
    binary_weight=1.0, continuous_weight=1.0, 
    layer_dice_config=None, posw_ema=None
):
    """
    Compute reconstruction loss for any tensor using dynamic layer configuration.
    
    Handles:
    - Binary layers: BCE with logits + optional Dice loss
    - Continuous layers: MSE loss
    - RGB layers: MSE loss (perceptual loss handled externally)
    
    Args:
        recon: Reconstructed tensor [B, C, H, W] (logits for binary channels, values for continuous)
        target: Target tensor [B, C, H, W]
        channel_names: List of channel names (e.g., ['rgb:red', 'buildings', 'lst'])
        layer_names: List of layer names for each channel (e.g., ['rgb', 'buildings', 'lst'])
        layers_registry: Global layers configuration dict
        binary_weight: Weight for binary channel losses
        continuous_weight: Weight for continuous channel losses
        layer_dice_config: Optional dict mapping layer names to dice config overrides
        posw_ema: Optional PosWeightEMA tracker for stable class weighting
        
    Returns:
        Dictionary with losses per channel type, total loss tensor
    """
    
    losses = {}
    binary_loss = 0.0
    continuous_loss = 0.0
    
    binary_count = 0
    continuous_count = 0
    
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
                pw = posw_ema.update(idx, target_ch)
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
            
        # Continuous channels use MSE loss
        else:
            # Check if this is a height channel that needs masking
            mask_layer = layer_info.get('mask_layer', None)
            
            if mask_layer:
                # Find mask channel index
                mask_idx = None
                for i, (ch_name, l_name) in enumerate(zip(channel_names, layer_names)):
                    if l_name == mask_layer:
                        mask_idx = i
                        break
                
                if mask_idx is not None:
                    mask = target[:, mask_idx:mask_idx+1, :, :]
                    # Apply loss only where mask is active
                    masked_recon = recon_ch * mask
                    masked_target = target_ch * mask
                    loss = F.mse_loss(masked_recon, masked_target, reduction='mean')
                else:
                    # Mask layer not found, use regular MSE
                    loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            else:
                # Regular continuous channel (e.g., LST, NDVI, RGB channels)
                loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            
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


########## Main Training Function #############
def train_vae(mode: str = 'satellite'):
    """
    Generic VAE training function supporting any VAE group defined in config.
    
    Args:
        mode: VAE group name (e.g., 'satellite', 'semantic', 'environmental')
              Must match a key in config['vae_groups']
    """
    # Record training start time
    training_start_time = time.time()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    ###### setup config variables #######
    config = load_configs()
    data_config = config['data_config']
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"VAE Training: {mode.upper()}")
        print(f"{'='*60}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        print(f"✓ VAE Group: {mode}")
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(yaml.dump(config, default_flow_style=False))
    
    dataset_config = config['dataset_params']
    train_config_global = config['train_params']
    
    # Validate VAE group exists
    vae_groups = config.get('vae_groups', {})
    if mode not in vae_groups:
        raise ValueError(
            f"VAE group '{mode}' not found in config. "
            f"Available groups: {list(vae_groups.keys())}"
        )
    
    # Get VAE group configuration
    vae_group_config = vae_groups[mode]
    layers_registry = config.get('layers', {})
    
    # Parse layers and compute input channels
    group_layers = vae_group_config.get('layers', [])
    if not group_layers:
        raise ValueError(f"VAE group '{mode}' has no layers defined")
    
    from model.utils.layer_config import count_layer_channels, get_layer_info
    
    num_input_channels = 0
    layer_names = []
    for layer_name in group_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_input_channels += num_channels
        layer_names.append(layer_name)
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"VAE Group: {mode}")
        print(f"{'='*60}")
        print(f"✓ Layers: {layer_names}")
        print(f"✓ Total input channels: {num_input_channels}")
        print(f"{'='*60}\n")
    
    # Get VAE architecture config
    autoencoder_config = {
        'z_channels': vae_group_config.get('z_channels', 4),
        'down_channels': vae_group_config.get('down_channels', [32, 64, 128, 128]),
        'mid_channels': vae_group_config.get('mid_channels', [128, 128]),
        'down_sample': vae_group_config.get('down_sample', [True, True, True]),
        'attn_down': vae_group_config.get('attn_down', [False, False, False]),
        'norm_channels': vae_group_config.get('norm_channels', 32),
        'num_heads': vae_group_config.get('num_heads', 2),
        'num_down_layers': vae_group_config.get('num_down_layers', 2),
        'num_mid_layers': vae_group_config.get('num_mid_layers', 2),
        'num_up_layers': vae_group_config.get('num_up_layers', 2),
    }
    
    # Get training configuration
    vae_training_config = train_config_global.get('vae_training', {})
    train_config = vae_training_config.get(mode, {})
    
    num_epochs = train_config.get('epochs', 50)
    batch_size = train_config.get('batch_size', 4)
    base_lr = train_config.get('lr', 0.0001)
    kl_weight = train_config.get('kl_weight', 0.000001)
    perceptual_weight = train_config.get('perceptual_weight', 1.0)
    disc_weight = train_config.get('disc_weight', 0.5)
    disc_start_steps = train_config.get('disc_start', 10000)
    use_discriminator = train_config.get('use_discriminator', True)
    use_perceptual = train_config.get('use_perceptual', True)
    penalize_out_of_range = train_config.get('penalize_out_of_range', False)
    tanh_activation = train_config.get('tanh_activation', False)
    tanh_scaling = train_config.get('tanh_scaling', 0.95)
    binary_channel_weight = train_config.get('binary_channel_weight', 1.0)
    continuous_channel_weight = train_config.get('continuous_channel_weight', 1.0)
    dice_weight = train_config.get('dice_weight', 0.5)
    img_save_steps = train_config.get('img_save_steps', 64)
    
    # Get layer-specific dice loss config
    layer_dice_config = train_config.get('layer_dice_config', {})
    
    # Directory and naming setup from VAE group config
    checkpoint_name = vae_group_config.get('checkpoint_name', f'{mode}_vae_ckpt.pth')
    latent_dir_name = vae_group_config.get('latents_dir', f'{mode}_latents')
    samples_dir_name = vae_group_config.get('samples_dir', f'{mode}_samples')
    stats_name = vae_group_config.get('stats_dir', f'{mode}_stats')
    
    
    # Create output directories
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    latent_dir = os.path.join(out_dir, latent_dir_name)
    samples_dir = os.path.join(out_dir, samples_dir_name)
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
    
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    
    cache_dir = f"{big_data_storage_path}/processed/{task_name}/vae:{mode}"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading Urban Dataset for {mode.upper()} VAE Training")
        print(f"  Cache directory: {cache_dir} (use_cached_patches={use_cached_patches})")
        print(f"  Big data storage path: {big_data_storage_path}")
        print(f"  Task name: {task_name}")
        print(f"  Output directories:")
        print(f"    - Output: {out_dir}")
        print(f"    - Latents: {latent_dir}")
        print(f"    - Samples: {samples_dir}")
        print(f"  VAE Group: {mode}")
        print(f"{'='*50}")
    
    # For VAE training, use vae:<mode> to get only the target layers
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=f'vae:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ VAE group '{mode}' with {num_input_channels} input channels")
        print(f"✓ Layers: {layer_names}")
    
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(
        urban_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    # Convert discriminator start steps to epochs
    disc_start_epoch = disc_start_steps // len(data_loader)
    
    ########## Create Models #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing Models")
        print(f"{'='*50}")
    
    # VAE model with mode-appropriate input channels
    model = VAE(
        im_channels=num_input_channels,
        model_config=autoencoder_config
    ).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if is_main:
            print("✓ Wrapped model in DistributedDataParallel")
    
    # Set model to train mode
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        param_count = sum(p.numel() for p in model_unwrapped.parameters()) / 1e6
        print(f"✓ Created VAE with {param_count:.2f}M parameters")
        print(f"  - Input channels: {num_input_channels}")
        print(f"  - Latent channels: {autoencoder_config['z_channels']}")
        print(f"  - Downsampling factor: {2 ** sum(autoencoder_config['down_sample'])}")
    
    # Discriminator for adversarial loss
    discriminator = None
    optimizer_disc = None
    
    if use_discriminator:
        discriminator = Discriminator(
            im_channels=num_input_channels
        ).to(device)
        
        # No wrap of the discriminator in DDP – each rank has its own copy
        if is_main:
            disc_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
            print(f"✓ Created Discriminator with {disc_params:.2f}M parameters")
            if world_size > 1:
                print("✓ Using per-rank discriminator (no DDP wrapper)")
    
    # LPIPS perceptual loss
    lpips_model = None
    if use_perceptual:
        lpips_model = LPIPS().eval().to(device)
        if is_main:
            print("✓ Created LPIPS perceptual loss model")
    
    ########## Training Setup #############
    
    # Initialize PosWeightEMA for binary channels (if any)
    posw_ema = None
    binary_channel_count = sum(
        1 for layer_name in layer_names 
        if layers_registry.get(layer_name, {}).get('type') == 'binary'
    )
    if binary_channel_count > 0:
        posw_ema = PosWeightEMA(
            num_channels=binary_channel_count,
            momentum=0.95,
            init=1.0,
            device=device
        )
        if is_main:
            print(f"\n✓ Initialized PosWeightEMA for {binary_channel_count} binary channels in {mode} VAE")
    
    # Scale learning rate with world size
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer_vae = Adam(model.parameters(), lr=adjusted_lr)
    
    if use_discriminator:
        optimizer_disc = Adam(discriminator.parameters(), lr=adjusted_lr)
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr} from base {base_lr}")
        print(f"✓ Batch size per GPU: {batch_size}")
        print(f"✓ Effective batch size: {batch_size * world_size}")
        print(f"✓ KL weight: {kl_weight}")
        if use_perceptual:
            print(f"✓ Perceptual weight: {perceptual_weight}")
        if use_discriminator:
            print(f"✓ Discriminator weight: {disc_weight} (starting epoch {disc_start_epoch})")
        if penalize_out_of_range:
            print(f"✓ Out-of-bounds penalization: Enabled")
        print(f"✓ Binary channel weight: {binary_channel_weight}")
        print(f"✓ Continuous channel weight: {continuous_channel_weight}")
        print(f"✓ Dice weight: {dice_weight}")
    
    ########## Training Loop #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Starting Training with {num_epochs} epochs")
        print(f"{'='*50}")
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        losses_vae = []
        losses_disc = []
        
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            # Extract data from batch
            if len(data) == 2:
                input_tensor, meta_dict = data
                # Extract metadata
                meta = meta_dict.get('meta', {})
                # meta is a list of dicts (one per batch item)
                if isinstance(meta, list) and len(meta) > 0:
                    channel_names = meta[0].get('channel_names', [])
                    layer_names_batch = meta[0].get('layer_names', [])
                else:
                    channel_names = []
                    layer_names_batch = []
            else:
                # Fallback for unexpected format
                input_tensor = data
                channel_names = []
                layer_names_batch = []
            
            # Move to device
            input_tensor = input_tensor.float().to(device)
            
            # Validate channel count matches VAE expectations
            if input_tensor.shape[1] != num_input_channels:
                raise RuntimeError(
                    f"Channel mismatch! VAE expects {num_input_channels} channels but got {input_tensor.shape[1]} channels.\n"
                    f"VAE group: {mode}\n"
                    f"Expected layers: {layer_names}\n"
                    f"Got channel_names: {channel_names}\n"
                    f"This likely means the dataset is not providing the expected channels for this VAE group."
                )
            
            # Sanity check: print channel stats on first batch each epoch (rank 0 only)
            if is_main and batch_idx == 0:
                print(f"\n[Epoch {epoch_idx + 1}, VAE group: {mode}]")
                print(f"  Channel names: {channel_names}")
                print(f"  Layer names: {layer_names_batch}")
                
                ch_means = input_tensor.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                ch_stds = input_tensor.std(dim=(0, 2, 3)).detach().cpu().numpy()
                ch_mins = input_tensor.min(dim=0)[0].min(dim=1)[0].min(dim=1)[0].detach().cpu().numpy()
                ch_maxs = input_tensor.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0].detach().cpu().numpy()
                
                ch_pos = (input_tensor > 0.5).float().mean(dim=(0, 2, 3)).detach().cpu().numpy()
                print(f"\n[Epoch {epoch_idx + 1}] Channel statistics for VAE group '{mode}':")
                for i, (ch_name, layer_name) in enumerate(zip(channel_names, layer_names_batch)):
                    # Check if binary or continuous based on layer registry
                    layer_config = layers_registry.get(layer_name, {})
                    is_binary = layer_config.get('type', 'continuous') == 'binary'
                    
                    if is_binary:
                        print(f"  {i:02d} {ch_name:30s} [binary] mean={ch_means[i]:.4f} std={ch_stds[i]:.4f} pos@0.5={ch_pos[i]:.4f}")
                    else:
                        print(f"  {i:02d} {ch_name:30s} [cont.] mean={ch_means[i]:+.4f} std={ch_stds[i]:.4f} min={ch_mins[i]:+.4f} max={ch_maxs[i]:+.4f}")
                print()
            
            ########## Train VAE ##########
            
            ############################
            # 1) VAE / Generator step  #
            ############################
            
            # Freeze discriminator params for generator step
            if use_discriminator and epoch_idx >= disc_start_epoch:
                for p in discriminator.parameters():
                    p.requires_grad = False
            
            optimizer_vae.zero_grad()
            
            # Forward pass
            recon, z, mean, logvar = model(input_tensor)
            
            # Reconstruction loss - use unified loss computation for all layer types
            loss_dict, recon_loss = compute_reconstruction_loss(
                recon, input_tensor, 
                channel_names, layer_names_batch,
                layers_registry,
                binary_weight=binary_channel_weight,
                continuous_weight=continuous_channel_weight,
                layer_dice_config=layer_dice_config,
                posw_ema=posw_ema
            )
                
            # Out-of-bounds penalization for satellite
            if penalize_out_of_range:
                range_penalty = torch.relu(torch.abs(recon) - 1.0).mean()
                recon_loss = recon_loss + 0.1 * range_penalty
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / (input_tensor.shape[0] * input_tensor.shape[2] * input_tensor.shape[3])
            
            # Perceptual loss (LPIPS)
            perceptual_loss = 0.0
            if use_perceptual and lpips_model is not None:
                if input_tensor.shape[1] == 3:
                    perceptual_loss = lpips_model(input_tensor, recon).mean()
            
            # Generator loss (fool discriminator)
            gen_loss = 0.0
            if use_discriminator and epoch_idx >= disc_start_epoch:
                disc_fake = discriminator(recon)
                gen_loss = -torch.mean(disc_fake)
            
            # Total VAE loss
            vae_loss = (recon_loss + 
                       kl_weight * kl_loss + 
                       perceptual_weight * perceptual_loss +
                       disc_weight * gen_loss)
            
            vae_loss.backward()
            optimizer_vae.step()
            
            ############################
            # 2) Discriminator step    #
            ############################
            if use_discriminator and epoch_idx >= disc_start_epoch:
                # Unfreeze discriminator params
                for p in discriminator.parameters():
                    p.requires_grad = True
                
                optimizer_disc.zero_grad()
                
                # Important: .detach() inputs so D doesn't backprop into VAE
                # Discriminator on real images
                disc_real = discriminator(input_tensor.detach())
                
                # Discriminator on fake images
                disc_fake = discriminator(recon.detach())
                
                # Discriminator loss (hinge loss)
                disc_loss = torch.mean(torch.relu(1.0 - disc_real)) + torch.mean(torch.relu(1.0 + disc_fake))
                
                disc_loss.backward()
                optimizer_disc.step()
                
                losses_disc.append(disc_loss.item())
            
            losses_vae.append(vae_loss.item())
            global_step += 1
            
            # Update progress bar (main process only)
            if is_main:
                postfix = {
                    'vae_loss': f'{vae_loss.item():.4f}',
                    'recon': f'{loss_dict["total_recon"]:.4f}',
                    'kl': f'{kl_loss.item():.6f}'
                }
                
                if use_discriminator and epoch_idx >= disc_start_epoch and len(losses_disc) > 0:
                    postfix['disc'] = f'{disc_loss.item():.4f}'
                
                progress_bar.set_postfix(postfix)
            
            # Save sample reconstructions
            if is_main and global_step % img_save_steps == 0:
                with torch.no_grad():
                    save_vae_reconstruction_samples(
                        input_tensor=input_tensor,
                        recon_tensor=recon,
                        layer_names=layer_names,
                        layers_registry=layers_registry,
                        save_dir=samples_dir,
                        step=global_step,
                        n_samples=8,
                        save_rgb_composite=True
                    )
        
        # Synchronize epoch metrics across GPUs
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary (main process only)
        if is_main:
            epoch_vae_loss = np.mean(losses_vae)
            if use_discriminator and epoch_idx >= disc_start_epoch:
                epoch_disc_loss = np.mean(losses_disc)
                print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | VAE Loss: {epoch_vae_loss:.4f} | Disc Loss: {epoch_disc_loss:.4f}')
            else:
                print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | VAE Loss: {epoch_vae_loss:.4f}')
        
        # Save checkpoint (main process only)
        if is_main:
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
            torch.save(model_to_save.state_dict(), checkpoint_path)
            
            # Save periodic checkpoint
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'{mode}_vae_ddp_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(model_to_save.state_dict(), periodic_path)
                print(f'✓ Saved checkpoint: {periodic_path}')
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
    ########## Save Latents ##########
    if train_config_global.get('save_latents', True):
        # Save latents in distributed manner
        latent_count = save_latents_distributed(
            model=model,
            dataset=urban_dataset,
            latent_dir=latent_dir,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            device=device,
            mode=mode,
        )
        
        print(f"Rank {rank}: Saved {latent_count} latents")
        
        # Save dataset statistics (main rank only)
        if is_main:
            urban_dataset.save_stats(f"{out_dir}/{stats_name}")
            print("✓ Saved dataset statistics")
    
    # Synchronize before cleanup
    if world_size > 1:
        dist.barrier()
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print(f"\n{'='*60}")
        print(f"✓ {mode.capitalize()} VAE Training Complete at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}!")
        print(f"✓ Total Training Time: {hours}h {minutes}m {seconds}s ({training_time:.2f} seconds)")
        print(f"{'='*60}")
    
    # Cleanup
    cleanup_distributed()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE DDP for Urban Inpainting')
    
    # Add config file arguments
    from helpers.load_configs import add_config_arguments
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, required=True,
                        help='VAE group to train (must match a key in config vae_groups, e.g., "satellite", "semantic", "environmental")')
    
    args = parser.parse_args()
    
    train_vae(mode=args.mode)

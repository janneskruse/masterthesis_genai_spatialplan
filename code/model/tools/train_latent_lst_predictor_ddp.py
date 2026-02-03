"""
Training script for Latent LST Predictor (DDP).

Trains a predictor to estimate LST p95 (or config-specified statistic) from VAE latent representations.
Can be used for Phase 2 (soft guidance) and Phase 3 (hard check) in diffusion sampling.

"""

###### import libraries ######
# Standard libraries
import os
import argparse
import time
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Data Science/ML libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.checkpoint import load_checkpoint, check_existing_paths
from model.lst_predictor.latent_predictor import LatentLSTPredictor
from helpers.load_configs import load_configs, add_config_arguments

# Load CUDA
load_cuda()


def compute_lst_statistic(
    lst_tensor: torch.Tensor,
    statistic: str = 'p95',
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute LST statistic from full-resolution tensor.
    
    Args:
        lst_tensor: LST tensor [B, 1, H, W] in [0, 1] range
        statistic: 'p95', 'mean', 'max', 'p99'
        mask: Optional mask [B, 1, H, W] where 1 = inside region
        
    Returns:
        Statistic tensor [B, 1]
    """
    B = lst_tensor.shape[0]
    results = []
    
    for b in range(B):
        lst_flat = lst_tensor[b].flatten()  # [H*W]
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask[b].flatten().bool()
            lst_flat = lst_flat[mask_flat]
        
        # Remove NaN/invalid values
        lst_flat = lst_flat[~torch.isnan(lst_flat)]
        lst_flat = lst_flat[lst_flat > 0]  # LST should be positive
        
        if len(lst_flat) == 0:
            # Fallback to 0 if no valid pixels
            results.append(torch.tensor(0.0, device=lst_tensor.device))
            continue
        
        if statistic == 'p95':
            val = torch.quantile(lst_flat, 0.95)
        elif statistic == 'p99':
            val = torch.quantile(lst_flat, 0.99)
        elif statistic == 'mean':
            val = lst_flat.mean()
        elif statistic == 'max':
            val = lst_flat.max()
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        results.append(val)
    
    return torch.stack(results).unsqueeze(1)  # [B, 1]


def train_latent_lst_predictor(mode: str = 'semantic', load_checkpoint_path: str = None):
    """
    Train latent LST predictor for a specific VAE group.
    
    Args:
        mode: VAE group name ('semantic' or 'satellite')
        load_checkpoint_path: Optional path to checkpoint file to resume training from
    """
    # Record training start time
    training_start_time = time.time()
    
    # ========= load config files ==========
    config = load_configs()
    data_config = config['data_config']
    train_config_global = config['train_params']
    
    # ========== Check for existing paths (skip training if artifacts already exist) ==========
    existing_paths_result = check_existing_paths(
        train_config=train_config_global,
        mode=mode,
        type='lst_latent'
    )
    
    # Early exit if LST latent predictor checkpoint already exists (before DDP setup)
    if existing_paths_result.skip_training:
        print(f"\n{'='*60}")
        print(f"SKIPPING LST LATENT PREDICTOR TRAINING: Using existing checkpoint")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Existing path: {existing_paths_result.latent_lst_predictor_checkpoint or 'N/A'}")
        print(f"{'='*60}\n")
        return
    
    existing_patches_path = existing_paths_result.patches_path
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Latent LST Predictor DDP Training")
        print(f"{'='*60}")
        print(f"✓ Mode: {mode}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
    
    # Validate mode
    vae_groups = config.get('vae_groups', {})
    if mode not in vae_groups:
        raise ValueError(
            f"Mode '{mode}' not found in VAE groups. "
            f"Available: {list(vae_groups.keys())}"
        )
    
    # Get predictor config
    predictor_config = config.get('latent_lst_predictor', {})
    mode_config = predictor_config.get('modes', {}).get(mode, {})
    
    # Architecture params
    z_channels = mode_config.get('z_channels', vae_groups[mode].get('z_channels', 3))
    latent_size = mode_config.get('latent_size', 64)
    hidden_dims = predictor_config.get('hidden_dims', [64, 128, 256])
    
    # Training params
    num_epochs = predictor_config.get('epochs', 100)
    batch_size = predictor_config.get('batch_size', 32)
    base_lr = predictor_config.get('lr', 0.0001)
    loss_type = predictor_config.get('loss', 'mse')
    
    # Target computation params
    statistic = predictor_config.get('statistic', 'p95')
    region = predictor_config.get('region', 'full')  # 'full' or 'mask'
    
    # Create output directory
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = Path(big_data_storage_path) / "results" / task_name
    
    # Checkpoint name
    checkpoint_name = mode_config.get('checkpoint_name', f'latent_lst_predictor_{mode}.pth')
    
    # checkpoint path
    if load_checkpoint_path is not None:
        load_checkpoint_path = os.path.join(out_dir, load_checkpoint_path)
    
    # Get LST normalization range for interpretability
    layers_registry = config.get('layers', {})
    lst_config = layers_registry.get('lst', {})
    lst_max = lst_config.get('normalize_params', {}).get('max', 80)
    
    if is_main:
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(f"✓ z_channels: {z_channels}")
        print(f"✓ latent_size: {latent_size}")
        print(f"✓ hidden_dims: {hidden_dims}")
        print(f"✓ epochs: {num_epochs}")
        print(f"✓ batch_size: {batch_size}")
        print(f"✓ lr: {base_lr}")
        print(f"✓ loss: {loss_type}")
        print(f"✓ statistic: {statistic}")
        print(f"✓ region: {region}")
        print(f"✓ LST max (Celsius): {lst_max}")
    
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading Dataset for LST Predictor (mode: lst:{mode})")
        print(f"{'='*50}")
    
    # check for existing cached patches
    if existing_patches_path is not None:
        cache_dir = existing_patches_path
    else:
        cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    # Create dataset
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=f'lst:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=str(cache_dir)
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training samples")
        print(f"✓ Latent size: {latent_size}x{latent_size}")
        print(f"✓ Latent channels: {z_channels}")
    
    # Distributed sampler
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
    
    ########## Create Model #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing Latent LST Predictor")
        print(f"{'='*50}")
    
    model = LatentLSTPredictor(
        z_channels=z_channels,
        latent_size=latent_size,
        hidden_dims=hidden_dims,
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
    
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        param_count = sum(p.numel() for p in model_unwrapped.parameters()) / 1e6
        print(f"✓ Created Latent LST Predictor with {param_count:.2f}M parameters")
    
    ########## Training Setup #############
    adjusted_lr = base_lr * world_size
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    # Loss function
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_type == 'huber':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    
    # Load checkpoint if provided
    start_epoch = 0
    if load_checkpoint_path:
        start_epoch, _ = load_checkpoint(
            checkpoint_path=load_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            is_main=is_main
        )
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        print(f"✓ Effective batch size: {batch_size * world_size}")
    
    ########## Training Loop #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Starting Training")
        print(f"{'='*50}")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch_idx in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        losses = []
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}', disable=(rank != 0))
        
        for batch_idx, data in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Extract data: (latent, cond_dict)
            if len(data) != 2:
                continue
            
            latent, cond_dict = data
            
            # Get LST full-res from conditioning
            if 'image' not in cond_dict or cond_dict['image'] is None:
                continue
            
            cond_image = cond_dict['image']
            meta = cond_dict.get('meta', {})
            
            if isinstance(meta, list) and len(meta) > 0:
                pixel_space_names = meta[0].get('pixel_space_names', [])
            else:
                pixel_space_names = meta.get('pixel_space_names', [])
            
            # Find LST channel in conditioning
            lst_idx = None
            mask_idx = None
            
            for i, name in enumerate(pixel_space_names):
                if name == 'lst':
                    lst_idx = i
                elif name == 'inpainting_mask':
                    mask_idx = i
            
            if lst_idx is None:
                if is_main and global_step == 0:
                    print(f"⚠ LST not found in pixel_space_names: {pixel_space_names}")
                continue
            
            # Extract LST and mask
            lst_fullres = cond_image[:, lst_idx:lst_idx+1, :, :].float().to(device)
            
            mask = None
            if region == 'mask' and mask_idx is not None:
                mask = cond_image[:, mask_idx:mask_idx+1, :, :].float().to(device)
            
            # Compute target statistic
            target = compute_lst_statistic(lst_fullres, statistic=statistic, mask=mask)
            target = target.to(device)
            
            # Forward pass
            latent = latent.float().to(device)
            pred = model(latent)
            
            # Compute loss
            loss = loss_fn(pred, target)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            global_step += 1
            
            if rank == 0:
                # Convert loss to Celsius for interpretability
                loss_celsius = loss.item() * lst_max
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    '~°C': f'{loss_celsius:.2f}'
                })
        
        # Synchronize
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary
        epoch_loss = np.mean(losses)
        
        if is_main:
            epoch_loss_celsius = epoch_loss * lst_max
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f} (~{epoch_loss_celsius:.2f}°C)')
        
        # Save best checkpoint
        if is_main and epoch_loss < best_loss:
            best_loss = epoch_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = out_dir / checkpoint_name
            
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'z_channels': z_channels,
                    'latent_size': latent_size,
                    'hidden_dims': hidden_dims,
                    'mode': mode,
                    'statistic': statistic,
                    'region': region,
                }
            }, checkpoint_path)
            
            best_loss_celsius = best_loss * lst_max
            print(f"  ✓ Saved best model (loss: {best_loss:.4f} ~{best_loss_celsius:.2f}°C)")
        
        # Periodic checkpoint
        if is_main and (epoch_idx + 1) % 20 == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            periodic_path = out_dir / f'latent_lst_predictor_{mode}_epoch_{epoch_idx + 1}.pth'
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model_to_save.state_dict(),
                'loss': epoch_loss,
            }, periodic_path)
            print(f'✓ Saved checkpoint: {periodic_path}')
        
        if world_size > 1:
            dist.barrier()
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        best_loss_celsius = best_loss * lst_max
        
        print(f"\n{'='*60}")
        print(f"✓ Latent LST Predictor Training Complete!")
        print(f"✓ Mode: {mode}")
        print(f"✓ Best loss: {best_loss:.4f} (~{best_loss_celsius:.2f}°C)")
        print(f"✓ Total Training Time: {hours}h {minutes}m {seconds}s")
        print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Latent LST Predictor DDP')
    
    add_config_arguments(parser)
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['semantic', 'satellite'],
        help='VAE group to train predictor for (semantic or satellite)'
    )
    
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    args = parser.parse_args()
    
    try:
        train_latent_lst_predictor(mode=args.mode, load_checkpoint_path=args.load_checkpoint)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Cleanup all distributed processes
        cleanup_distributed()
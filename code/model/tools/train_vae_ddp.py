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
from model.utils.config_utils import get_semantic_channels
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()


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
    semantic_channels: list = None,
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
        mode: 'semantic' or 'satellite' - determines input building logic
        semantic_channels: List of semantic channel names (required for semantic mode)
        
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
        mode_name = "Semantic" if mode == 'semantic' else "Satellite"
        print(f"\n{'='*60}")
        print(f"Encoding and Saving {mode_name} Latents (Distributed) at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Samples per rank: {samples_per_rank}")
        print(f"World size: {world_size}")
    
    print(f"Rank {rank}: Processing indices {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    
    # Create subset of dataset for this rank
    rank_indices = list(range(start_idx, end_idx))
    rank_dataset = torch.utils.data.Subset(dataset, rank_indices)
    
    # Create dataloader for this rank's subset
    from model.utils.data_utils import collate_fn
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
            # Handle different data formats
            if len(data) == 2:
                im, cond_input = data
            else:
                im = data
                cond_input = {}
            
            # Build input based on mode
            if mode == 'semantic':
                # Build semantic tensor
                if 'image' in cond_input and 'meta' in cond_input:
                    semantic_tensor = []
                    meta = cond_input['meta']
                    # meta is a list of dicts (one per batch item), get spatial_names from first item
                    spatial_names = meta[0].get('spatial_names', []) if isinstance(meta, list) and len(meta) > 0 else []
                    
                    for sem_ch in semantic_channels:
                        found = False
                        for idx, name in enumerate(spatial_names):
                            if sem_ch in name or name in sem_ch:
                                semantic_tensor.append(cond_input['image'][:, idx:idx+1, :, :])
                                found = True
                                break
                        
                        if not found:
                            B, _, H, W = cond_input['image'].shape
                            semantic_tensor.append(torch.zeros(B, 1, H, W, device=cond_input['image'].device))
                    
                    input_tensor = torch.cat(semantic_tensor, dim=1)
                else:
                    input_tensor = im
            else:
                # Satellite mode: use RGB image directly
                input_tensor = im
            
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
        saved_latents = sorted([
            int(f.stem.split('_')[1]) 
            for f in latent_dir.glob('latent_*.pt')
        ])
        
        expected_latents = list(range(total_samples))
        missing_latents = set(expected_latents) - set(saved_latents)
        duplicate_latents = len(saved_latents) - len(set(saved_latents))
        
        print(f"\n{'='*60}")
        print(f"✓ Total latents saved: {len(saved_latents)}/{total_samples}")
        
        if missing_latents:
            print(f"⚠ Missing latents: {sorted(missing_latents)[:10]}{'...' if len(missing_latents) > 10 else ''}")
        
        if duplicate_latents > 0:
            print(f"⚠ Duplicate latents found: {duplicate_latents}")
        
        if len(saved_latents) == total_samples and not missing_latents and duplicate_latents == 0:
            print(f"✓ All latents saved successfully!")
        
        print(f"{'='*60}\n")
    
    return latent_count


def compute_semantic_reconstruction_loss(recon, target, semantic_channels):
    """
    Compute reconstruction loss for semantic tensor.
    
    Args:
        recon: Reconstructed semantic tensor [B, C, H, W]
        target: Target semantic tensor [B, C, H, W]
        semantic_channels: List of channel names
        
    Returns:
        Dictionary with losses per channel type
    """
    losses = {}
    binary_loss = 0.0
    continuous_loss = 0.0
    
    binary_count = 0
    continuous_count = 0
    
    for idx, channel_name in enumerate(semantic_channels):
        recon_ch = recon[:, idx:idx+1, :, :]
        target_ch = target[:, idx:idx+1, :, :]
        
        # Binary channels use BCE loss
        if 'buildings' in channel_name or 'streets' in channel_name or 'vegetation' in channel_name or 'water' in channel_name:
            # Clamp to valid probability range
            recon_ch = torch.clamp(recon_ch, 0.0, 1.0)
            target_ch = torch.clamp(target_ch, 0.0, 1.0)
            
            loss = F.binary_cross_entropy(recon_ch, target_ch, reduction='mean')
            losses[f'{channel_name}_bce'] = loss.item()
            binary_loss += loss
            binary_count += 1
            
        # Continuous channels (height) use MSE loss, gated by building mask
        elif 'height' in channel_name:
            # Find building mask channel
            building_idx = None
            for i, name in enumerate(semantic_channels):
                if 'buildings' in name and 'height' not in name:
                    building_idx = i
                    break
            
            if building_idx is not None:
                building_mask = target[:, building_idx:building_idx+1, :, :]
                # Apply loss only where buildings exist
                masked_recon = recon_ch * building_mask
                masked_target = target_ch * building_mask
                loss = F.mse_loss(masked_recon, masked_target, reduction='mean')
            else:
                # No building mask found, use regular MSE
                loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            
            losses[f'{channel_name}_mse'] = loss.item()
            continuous_loss += loss
            continuous_count += 1
            
        else:
            # Default to MSE for unknown channels
            loss = F.mse_loss(recon_ch, target_ch, reduction='mean')
            losses[f'{channel_name}_mse'] = loss.item()
            continuous_loss += loss
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
    Unified VAE training function supporting both semantic and satellite modes.
    
    Args:
        mode: 'semantic' or 'satellite' - determines which VAE to train
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
        print(f"{'Semantic' if mode == 'semantic' else 'Satellite'} VAE DDP Training")
        print(f"{'='*60}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        print(f"✓ Mode: {mode}")
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(yaml.dump(config, default_flow_style=False))
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    # Mode-specific configuration
    autoencoder_config = config['autoencoder_params'][mode]
    ldm_config = config.get('ldm_params', {})
    train_config = train_config.get(mode, train_config)
    num_epochs = train_config.get('autoencoder_epochs', 50)
    batch_size = train_config.get('autoencoder_batch_size', 4)
    base_lr = train_config.get('autoencoder_lr', 0.0001)
    kl_weight = train_config.get('kl_weight', 0.000001)
    perceptual_weight = train_config.get('perceptual_weight', 1.0)
    disc_weight = train_config.get('disc_weight', 0.5)
    disc_start_steps = train_config.get('disc_start', 10000)
    use_discriminator = train_config.get('use_discriminator', True)
    use_perceptual = train_config.get('use_perceptual', True)
    penalize_out_of_range = train_config.get('penalize_out_of_range', False)
    
    # directory and naming setup
    latent_dir_name = train_config.get('latents_dir_name', f'{mode}_vae_ddp_latents')
    samples_dir_name = train_config.get('autoencoder_samples_dir_name', f'{mode}_vae_ddp_samples')
    stats_name = train_config.get('stats_dir_name', f'{mode}_vae_ddp_stats')
    checkpoint_name = train_config.get('autoencoder_ckpt_name', f'{mode}_vae_ddp_ckpt.pth')
    
    if mode == 'semantic':
        # Get semantic channels from condition config
        semantic_ldm_config = ldm_config.get('semantic', ldm_config)
        condition_config = semantic_ldm_config.get('condition_config', {})
        semantic_channels = get_semantic_channels(condition_config)
        
        if not semantic_channels:
            # Fallback to default
            semantic_channels = [
                'osm:buildings',
                'osm:streets', 
                'env:vegetation',
                'osm:buildings_heights'
            ]
        
        num_input_channels = len(semantic_channels)
        
    else: # satellite mode
        num_input_channels = dataset_config['im_channels']
        semantic_channels = None
    
    # Create output directories
    task_name = train_config.get('task_name', 'urban_inpainting')
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
    
    cache_dir = f"{big_data_storage_path}/processed/{task_name}/{mode}"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading Urban Dataset for {mode.capitalize()} Training")
        print(f"  Cache directory: {cache_dir} (use_cached_patches={use_cached_patches})")
        print(f"  Big data storage path: {big_data_storage_path}")
        print(f"  Task name: {task_name}")
        print(f"  Output directories:")
        print(f"    - Output: {out_dir}")
        print(f"    - Latents: {latent_dir}")
        print(f"    - Samples: {samples_dir}")
        print(f"  Mode: {mode}")
        print(f"{'='*50}")
    
    # For VAE training, we don't use latents
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=mode,
        use_latents=False,
        latent_path=None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        if mode == 'semantic':
            print(f"✓ Semantic channels ({num_input_channels}): {semantic_channels}")
        else:
            print(f"✓ Image channels: {num_input_channels}")
    
    
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
                im, cond_input = data
            else:
                im = data
                cond_input = {}
            
            # Build input based on mode
            if mode == 'semantic':
                # Build semantic tensor from conditioning input
                if 'image' in cond_input and 'meta' in cond_input:
                    semantic_tensor = []
                    meta = cond_input['meta']
                    # meta is a list of dicts (one per batch item), get spatial_names from first item
                    spatial_names = meta[0].get('spatial_names', []) if isinstance(meta, list) and len(meta) > 0 else []
                    
                    # Extract semantic channels based on configuration
                    for sem_ch in semantic_channels:
                        found = False
                        for idx, name in enumerate(spatial_names):
                            if sem_ch in name or name in sem_ch:
                                semantic_tensor.append(cond_input['image'][:, idx:idx+1, :, :])
                                found = True
                                break
                        
                        if not found:
                            # Channel not found, create zeros
                            B, _, H, W = cond_input['image'].shape
                            semantic_tensor.append(torch.zeros(B, 1, H, W, device=cond_input['image'].device))
                    
                    input_tensor = torch.cat(semantic_tensor, dim=1)
                else:
                    # Fallback: use RGB image channels if available
                    input_tensor = im
            else:
                # Satellite mode: use RGB image directly
                input_tensor = im
            
            input_tensor = input_tensor.float().to(device)
            
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
            
            # Reconstruction loss
            if mode == 'semantic':
                # Semantic reconstruction loss (channel-specific)
                loss_dict, recon_loss = compute_semantic_reconstruction_loss(
                    recon, input_tensor, semantic_channels
                )
            else:
                # Satellite reconstruction loss (L1)
                recon_loss = torch.abs(input_tensor - recon).mean()
                
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
                if mode == 'semantic':
                    if use_discriminator and epoch_idx >= disc_start_epoch:
                        progress_bar.set_postfix({
                            'vae_loss': f'{vae_loss.item():.4f}',
                            'recon': f'{loss_dict["total_recon"]:.4f}',
                            'kl': f'{kl_loss.item():.6f}',
                            'disc': f'{disc_loss.item():.4f}'
                        })
                    else:
                        progress_bar.set_postfix({
                            'vae_loss': f'{vae_loss.item():.4f}',
                            'recon': f'{loss_dict["total_recon"]:.4f}',
                            'kl': f'{kl_loss.item():.6f}'
                        })
                else:
                    if epoch_idx >= disc_start_epoch:
                        progress_bar.set_postfix({
                            'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}',
                            'disc_loss': f'{np.mean(losses_disc[-100:]):.4f}'
                        })
                    else:
                        progress_bar.set_postfix({'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}'})
            
            # Save sample reconstructions
            if is_main and global_step % train_config.get('autoencoder_img_save_steps', 500) == 0:
                with torch.no_grad():
                    n_samples = min(8, input_tensor.shape[0])
                    
                    if mode == 'semantic':
                        # Semantic visualization: normalize each channel appropriately
                        vis_input = []
                        vis_recon = []
                        
                        for ch_idx, ch_name in enumerate(semantic_channels):
                            input_ch = input_tensor[:n_samples, ch_idx:ch_idx+1, :, :]
                            recon_ch = recon[:n_samples, ch_idx:ch_idx+1, :, :]
                            
                            if 'height' in ch_name:
                                # Normalize height to [0, 1] for visualization
                                max_height = 100.0  # Assume max height of 100m
                                input_ch = torch.clamp(input_ch / max_height, 0, 1)
                                recon_ch = torch.clamp(recon_ch / max_height, 0, 1)
                            else:
                                # Binary channels
                                input_ch = torch.clamp(input_ch, 0, 1)
                                recon_ch = torch.clamp(recon_ch, 0, 1)
                            
                            vis_input.append(input_ch)
                            vis_recon.append(recon_ch)
                        
                        # Stack channels for visualization
                        vis_input_tensor = torch.cat(vis_input, dim=1)
                        vis_recon_tensor = torch.cat(vis_recon, dim=1)
                        
                        # Create comparison grid
                        comparison = torch.cat([vis_input_tensor, vis_recon_tensor], dim=0)
                        grid = make_grid(comparison, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
                    else:
                        # Satellite visualization: normalize to [0, 1]
                        sample_im = input_tensor[:n_samples]
                        sample_recon = recon[:n_samples]
                        
                        sample_im = torch.clamp(sample_im, -1., 1.)
                        sample_im = (sample_im + 1) / 2
                        sample_recon = torch.clamp(sample_recon, -1., 1.)
                        sample_recon = (sample_recon + 1) / 2
                        
                        # Create comparison grid
                        comparison = torch.cat([sample_im, sample_recon], dim=0)
                        grid = make_grid(comparison, nrow=8, padding=2, pad_value=1.0)
                    
                    save_path = os.path.join(samples_dir, f'recon_step_{global_step}.png')
                    save_image(grid, save_path)
        
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
    if train_config.get('save_latents', True):
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
            semantic_channels=semantic_channels,
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
    
    parser.add_argument('--mode', type=str, default='satellite', choices=['semantic', 'satellite'],
                        help='Mode of VAE to train: "semantic" or "satellite"')
    
    args = parser.parse_args()
    
    train_vae(mode=args.mode)

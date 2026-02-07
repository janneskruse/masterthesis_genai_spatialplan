"""
==============================================================================
Training script for a Latent Temperature predictor (DDP).

Trains a predictor to estimate Temperature p95 (or config-specified statistic) 
from VAE latent representations.
Can be used for Phase 2 (soft guidance) and Phase 3 (hard check)
in diffusion sampling.
==============================================================================
"""

###### import libraries ######
# Standard libraries
import os
import argparse
import time
import math
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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.checkpoint import load_checkpoint, check_existing_paths
from model.utils.statistics import compute_temperature_statistic
from model.blocks.vae_registry import VAERegistry
from model.temperature_predictor.latent_predictor import LatentTemperaturePredictor
from helpers.load_configs import load_configs, add_config_arguments

# Load CUDA
load_cuda()

def train_latent_temperature_predictor(mode: str = 'semantic', load_checkpoint_path: str = None):
    """
    Train latent Temperature predictor for a specific VAE group.
    
    Args:
        mode: VAE group name ('semantic' or 'satellite')
        load_checkpoint_path: Optional path to checkpoint file to resume training from
    """
    # Record training start time
    training_start_time = time.time()
    
    # //////////////////////////////////////////////////
    # ============= load config files =================
    # /////////////////////////////////////////////////
    config = load_configs()
    data_config = config['data_config']
    train_config_global = config['train_params']
    
    # /////////////////////////////////////////////////////////////////////////
    # == Check for existing paths (skip training if artifacts already exist) ==
    # /////////////////////////////////////////////////////////////////////////
    existing_paths_result = check_existing_paths(
        train_config=train_config_global,
        mode=mode,
        type='temperature_latent'
    )
    
    # Early exit if Temperature latent predictor checkpoint already exists
    if existing_paths_result.skip_training:
        print(f"\n{'='*60}")
        print(f"SKIPPING TEMPERATURE LATENT PREDICTOR TRAINING: Using existing checkpoint")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Existing path: {existing_paths_result.latent_temperature_predictor_checkpoint or 'N/A'}")
        print(f"{'='*60}\n")
        return
    
    existing_patches_path = existing_paths_result.patches_path
    existing_vae_paths = existing_paths_result.vae_checkpoints
    
    
    # //////////////////////////////////////////////////////////////////
    # == Setup distributed training with all training configurations ==
    # /////////////////////////////////////////////////////////////////
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Latent Temperature predictor DDP Training")
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
    predictor_config = config.get('latent_temperature_predictor', {})
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
    
    # Regularization params
    weight_decay = predictor_config.get('weight_decay', 0.0)  # L2 regularization
    dropout = predictor_config.get('dropout', 0.1)  # Dropout rate
    
    # Learning rate scheduler params
    lr_scheduler_type = predictor_config.get('lr_scheduler', None)  # 'cosine', 'step', 'plateau', or None
    lr_warmup_epochs = predictor_config.get('lr_warmup_epochs', 0)
    lr_min = predictor_config.get('lr_min', 1e-6)
    
    # Early stopping params
    early_stopping_enabled = predictor_config.get('early_stopping', True)
    patience = predictor_config.get('patience', 10)  # Epochs to wait for improvement
    min_delta = predictor_config.get('min_delta', 1e-4)  # Minimum improvement threshold
    
    # Target computation params
    statistic = predictor_config.get('statistic', 'p95')
    region = predictor_config.get('region', 'full')  # 'full' or 'mask'
    
    # Create output directory
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = Path(big_data_storage_path) / "results" / task_name
    
    # Checkpoint name
    checkpoint_name = mode_config.get('checkpoint_name', f'latent_temperature_predictor_{mode}.pth')
    
    # checkpoint path
    if load_checkpoint_path is not None:
        load_checkpoint_path = os.path.join(out_dir, load_checkpoint_path)
    
    # Get Temperature normalization range for interpretability
    layers_registry = config.get('layers', {})
    temperature_config = layers_registry.get('temperature', {})
    temp_max = temperature_config.get('normalize_params', {}).get('max', 80)
    
    if is_main:
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(f"✓ z_channels: {z_channels}")
        print(f"✓ latent_size: {latent_size}")
        print(f"✓ hidden_dims: {hidden_dims}")
        print(f"✓ dropout: {dropout}")
        print(f"✓ epochs: {num_epochs}")
        print(f"✓ batch_size: {batch_size}")
        print(f"✓ lr: {base_lr}")
        print(f"✓ weight_decay: {weight_decay}")
        print(f"✓ lr_scheduler: {lr_scheduler_type or 'none'}")
        if lr_scheduler_type:
            print(f"✓ lr_warmup: {lr_warmup_epochs} epochs")
            print(f"✓ lr_min: {lr_min}")
        print(f"✓ loss: {loss_type}")
        print(f"✓ statistic: {statistic}")
        print(f"✓ region: {region}")
        print(f"✓ early_stopping: {early_stopping_enabled} (patience={patience})")
        print(f"✓ Temperature max (Celsius): {temp_max}")
    
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading Dataset for Temperature predictor (mode: temperature:{mode})")
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
        mode=f'temperature:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=str(cache_dir)
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training samples")
        print(f"✓ Latent size: {latent_size}x{latent_size}")
        print(f"✓ Latent channels: {z_channels}")
    
    ########## Check if training dataset needs encoding #############
    # This happens when pre-computed latents don't exist
    vae = None
    needs_encoding_global = False
    
    # Check first sample to see if encoding is needed
    sample_data = urban_dataset[0]
    if len(sample_data) == 2:
        _, sample_cond = sample_data
        sample_meta = sample_cond.get('meta', {})
        if isinstance(sample_meta, list) and len(sample_meta) > 0:
            needs_encoding_global = sample_meta[0].get('needs_encoding', False)
        else:
            needs_encoding_global = sample_meta.get('needs_encoding', False)
    
    if is_main and needs_encoding_global:
        print(f"✓ Training dataset requires on-the-fly encoding")
    
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
        sampler=sampler,
        drop_last=True
    )
    
    ########## Load Validation Dataset (for early stopping) #############
    val_loader = None
    val_needs_encoding = False
    if early_stopping_enabled:
        if is_main:
            print(f"\n{'='*50}")
            print(f"Loading Validation Dataset for Early Stopping")
            print(f"{'='*50}")
        
        val_dataset = UrbanInpaintingDataset(
            split='val',
            mode=f'temperature:{mode}',
            use_cached_patches=use_cached_patches,
            cache_dir=str(cache_dir)
        )
        
        if is_main:
            print(f"✓ Loaded {len(val_dataset)} validation samples")
        
        # Check if validation dataset needs encoding
        val_sample_data = val_dataset[0]
        if len(val_sample_data) == 2:
            _, val_sample_cond = val_sample_data
            val_sample_meta = val_sample_cond.get('meta', {})
            if isinstance(val_sample_meta, list) and len(val_sample_meta) > 0:
                val_needs_encoding = val_sample_meta[0].get('needs_encoding', False)
            else:
                val_needs_encoding = val_sample_meta.get('needs_encoding', False)
        
        if is_main and val_needs_encoding:
            print(f"✓ Validation dataset requires on-the-fly encoding")
        
        # Validation sampler (no shuffle, include all samples)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        ) if world_size > 1 else None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=val_sampler
        )
    
    ########## Load VAE for on-the-fly encoding (if needed by train OR val) #############
    # Check if either training or validation needs encoding
    if (needs_encoding_global or val_needs_encoding) and vae is None:
        if is_main:
            print(f"\n{'='*50}")
            print(f"Loading VAE for on-the-fly encoding")
            print(f"{'='*50}")
        
        # Use VAERegistry for cleaner management
        vae_registry = VAERegistry(config, device)
        
        # Determine VAE checkpoint path
        vae_config = vae_groups.get(mode, {})
        if mode in existing_vae_paths:
            vae_ckpt_path = existing_vae_paths[mode]
        else:
            default_ckpt_name = vae_config.get('checkpoint_name', f'{mode}_vae_ckpt.pth')
            vae_ckpt_path = os.path.join(out_dir, default_ckpt_name)
        
        # Load VAE
        if is_main:
            print(f"  - {mode.upper()} VAE for encoding")
        vae_registry.load_vae(
            group_name=mode,
            checkpoint_path=vae_ckpt_path,
            autoencoder_config=vae_config
        )
        vae = vae_registry.get_vae(mode)
        vae_registry.freeze(mode)
        
        if vae is None:
            raise RuntimeError(
                f"Dataset returned full-res images but could not load VAE for mode '{mode}'. "
                f"Either provide pre-computed latents or ensure VAE checkpoint exists at {vae_ckpt_path}."
            )
        
        if is_main:
            print(f"✓ Loaded and froze {mode} VAE for encoding samples")
    
    ########## Create Model #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing Latent Temperature predictor")
        print(f"{'='*50}")
    
    model = LatentTemperaturePredictor(
        z_channels=z_channels,
        latent_size=latent_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
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
        print(f"✓ Created Latent Temperature predictor with {param_count:.2f}M parameters")
    
    ########## Training Setup #############
    adjusted_lr = base_lr * world_size
    optimizer = Adam(model.parameters(), lr=adjusted_lr, weight_decay=weight_decay)
    
    # Loss function
    # Note: RMSE loss gives values directly in normalized units (multiply by temp_max for Celsius)
    #       MSE loss requires sqrt conversion for interpretable error
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
        loss_is_squared = True  # Need sqrt for RMSE display
    elif loss_type == 'rmse':
        # RMSE: sqrt(mean((pred - target)^2)) - directly interpretable
        loss_fn = lambda pred, target: torch.sqrt(nn.MSELoss()(pred, target) + 1e-8)
        loss_is_squared = False  # Already in RMSE form
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss()
        loss_is_squared = False  # MAE is already linear
    elif loss_type == 'huber':
        loss_fn = nn.SmoothL1Loss()
        loss_is_squared = False  # Huber is ~linear for large errors
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Options: 'mse', 'rmse', 'l1', 'huber'")
    
    # Learning rate scheduler
    scheduler = None
    if lr_scheduler_type == 'cosine':
        # Cosine annealing from adjusted_lr to lr_min over (epochs - warmup) epochs
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs - lr_warmup_epochs,
            eta_min=lr_min
        )
        if is_main:
            print(f"✓ Using CosineAnnealingLR scheduler (T_max={num_epochs - lr_warmup_epochs}, eta_min={lr_min})")
    elif lr_scheduler_type == 'step':
        # Step decay every 30 epochs
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        if is_main:
            print(f"✓ Using StepLR scheduler (step=30, gamma=0.5)")
    elif lr_scheduler_type == 'plateau':
        # Reduce on plateau (will be stepped with val_loss)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=lr_min)
        if is_main:
            print(f"✓ Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")
    elif lr_scheduler_type is not None:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_type}. Options: 'cosine', 'step', 'plateau', or null")
    
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
    
    
    
    
    # /////////////////////////////////////////////////
    # =============== Training Loop ===================
    # /////////////////////////////////////////////////
    if is_main:
        print(f"\n{'='*50}")
        print(f"Starting Training")
        print(f"{'='*50}")
    
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch_idx in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        # ========== Training Phase ==========
        model.train()
        train_losses = []
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs} [Train]', disable=(rank != 0))
        
        for batch_idx, data in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Extract data: (latent_or_image, cond_dict)
            if len(data) != 2:
                continue
            
            latent_or_image, cond_dict = data
            
            # Get Temperature full-res from conditioning
            if 'image' not in cond_dict or cond_dict['image'] is None:
                continue
            
            cond_image = cond_dict['image']
            meta = cond_dict.get('meta', {})
            
            if isinstance(meta, list) and len(meta) > 0:
                pixel_space_names = meta[0].get('pixel_space_names', [])
                needs_encoding = meta[0].get('needs_encoding', False)
            else:
                pixel_space_names = meta.get('pixel_space_names', [])
                needs_encoding = meta.get('needs_encoding', False)
            
            # Encode full-res image to latent if needed
            if needs_encoding:
                with torch.no_grad():
                    full_res_image = latent_or_image.float().to(device)
                    latent, _, _ = vae.encode(full_res_image)
            else:
                latent = latent_or_image
            
            # Find Temperature channel in conditioning
            temperature_idx = None
            mask_idx = None
            
            for i, name in enumerate(pixel_space_names):
                if name == 'temperature':
                    temperature_idx = i
                elif name == 'inpainting_mask':
                    mask_idx = i
            
            if temperature_idx is None:
                if is_main and global_step == 0:
                    print(f"⚠ Temperature not found in pixel_space_names: {pixel_space_names}")
                continue
            
            # Extract Temperature and mask
            temperature_fullres = cond_image[:, temperature_idx:temperature_idx+1, :, :].float().to(device)
            
            mask = None
            if region == 'mask' and mask_idx is not None:
                mask = cond_image[:, mask_idx:mask_idx+1, :, :].float().to(device)
            
            # Compute target statistic
            target = compute_temperature_statistic(temperature_fullres, statistic=statistic, mask=mask)
            target = target.to(device)
            
            # Forward pass (latent already on device if encoded, otherwise move now)
            if not needs_encoding:
                latent = latent.float().to(device)
            pred = model(latent)
            
            # Compute loss
            loss = loss_fn(pred, target)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            if rank == 0:
                # Convert loss to error in Celsius for interpretability
                loss_val = loss.item()
                error_celsius = (math.sqrt(loss_val) if loss_is_squared else loss_val) * temp_max
                progress_bar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'err°C': f'{error_celsius:.2f}'
                })
        
        # Synchronize
        if world_size > 1:
            dist.barrier()
        
        # Training epoch summary
        train_loss = np.mean(train_losses)
        
        # ========== Validation Phase ==========
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs} [Val]', disable=(rank != 0)):
                    if len(data) != 2:
                        continue
                    
                    latent_or_image, cond_dict = data
                    
                    if 'image' not in cond_dict or cond_dict['image'] is None:
                        continue
                    
                    cond_image = cond_dict['image']
                    meta = cond_dict.get('meta', {})
                    
                    if isinstance(meta, list) and len(meta) > 0:
                        pixel_space_names = meta[0].get('pixel_space_names', [])
                        needs_encoding = meta[0].get('needs_encoding', False)
                    else:
                        pixel_space_names = meta.get('pixel_space_names', [])
                        needs_encoding = meta.get('needs_encoding', False)
                    
                    # Encode full-res image to latent if needed
                    if needs_encoding:
                        full_res_image = latent_or_image.float().to(device)
                        latent, _, _ = vae.encode(full_res_image)
                    else:
                        latent = latent_or_image
                    
                    # Find Temperature channel
                    temperature_idx = None
                    mask_idx = None
                    for i, name in enumerate(pixel_space_names):
                        if name == 'temperature':
                            temperature_idx = i
                        elif name == 'inpainting_mask':
                            mask_idx = i
                    
                    if temperature_idx is None:
                        continue
                    
                    temperature_fullres = cond_image[:, temperature_idx:temperature_idx+1, :, :].float().to(device)
                    
                    mask = None
                    if region == 'mask' and mask_idx is not None:
                        mask = cond_image[:, mask_idx:mask_idx+1, :, :].float().to(device)
                    
                    target = compute_temperature_statistic(temperature_fullres, statistic=statistic, mask=mask)
                    target = target.to(device)
                    
                    if not needs_encoding:
                        latent = latent.float().to(device)
                    pred = model(latent)
                    
                    val_loss_batch = loss_fn(pred, target)
                    val_losses.append(val_loss_batch.item())
            
            # Gather validation losses across all ranks
            val_loss = np.mean(val_losses)
            
            if world_size > 1:
                # Reduce val_loss across all ranks
                val_loss_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()
        
        # Epoch summary
        if is_main:
            # Convert loss to error in Celsius
            train_err_celsius = (math.sqrt(train_loss) if loss_is_squared else train_loss) * temp_max
            err_label = 'RMSE' if loss_is_squared else 'err'
            summary = f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Train: {train_loss:.4f} ({err_label} ~{train_err_celsius:.2f}°C)'
            if val_loss is not None:
                val_err_celsius = (math.sqrt(val_loss) if loss_is_squared else val_loss) * temp_max
                summary += f' | Val: {val_loss:.4f} ({err_label} ~{val_err_celsius:.2f}°C)'
            print(summary)
        
        # Determine which loss to use for checkpointing
        checkpoint_loss = val_loss if val_loss is not None else train_loss
        
        # Check if there's improvement (all ranks need to agree on this!)
        is_improvement = checkpoint_loss < best_val_loss - min_delta
        
        # Save best checkpoint (based on validation loss if available)
        if is_improvement:
            best_val_loss = checkpoint_loss
            patience_counter = 0  # Reset patience on ALL ranks
            
            # Only main rank saves the checkpoint
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_path = out_dir / checkpoint_name
                
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': {
                        'z_channels': z_channels,
                        'latent_size': latent_size,
                        'hidden_dims': hidden_dims,
                        'mode': mode,
                        'statistic': statistic,
                        'region': region,
                        'dropout': dropout,
                    },
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                
                best_err_celsius = (math.sqrt(best_val_loss) if loss_is_squared else best_val_loss) * temp_max
                loss_type_str = 'val' if val_loss is not None else 'train'
                err_label = 'RMSE' if loss_is_squared else 'err'
                print(f"  ✓ Saved best model ({loss_type_str} loss: {best_val_loss:.4f}, {err_label} ~{best_err_celsius:.2f}°C)")
        else:
            # No improvement - increment patience counter on ALL ranks
            patience_counter += 1
            if is_main and early_stopping_enabled:
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if early_stopping_enabled and patience_counter >= patience:
            if is_main:
                print(f"\n⚠ Early stopping triggered! No improvement for {patience} epochs.")
            break
        
        # Learning rate scheduler step
        if scheduler is not None:
            if epoch_idx >= lr_warmup_epochs:
                # After warmup, step the scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(checkpoint_loss)
                else:
                    scheduler.step()
            else:
                # During warmup: linear warmup from lr_min to adjusted_lr
                warmup_lr = lr_min + (adjusted_lr - lr_min) * (epoch_idx + 1) / lr_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            if is_main:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  LR: {current_lr:.6f}")
        
        # Periodic checkpoint
        if is_main and (epoch_idx + 1) % 20 == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            periodic_path = out_dir / f'latent_temperature_predictor_{mode}_epoch_{epoch_idx + 1}.pth'
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model_to_save.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
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
        best_rmse_celsius = math.sqrt(best_val_loss) * temp_max
        
        print(f"\n{'='*60}")
        print(f"✓ Latent Temperature predictor Training Complete!")
        print(f"✓ Mode: {mode}")
        print(f"✓ Best {'val' if val_loader else 'train'} loss: {best_val_loss:.4f} (RMSE ~{best_rmse_celsius:.2f}°C)")
        print(f"✓ Stopped at epoch: {epoch_idx + 1}/{num_epochs}")
        print(f"✓ Total Training Time: {hours}h {minutes}m {seconds}s")
        print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Latent Temperature predictor DDP')
    
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
        train_latent_temperature_predictor(mode=args.mode, load_checkpoint_path=args.load_checkpoint)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Cleanup all distributed processes
        cleanup_distributed()
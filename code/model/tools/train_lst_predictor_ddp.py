# Training script for LST predictor from semantic features
# Predicts Land Surface Temperature from semantic layout (buildings/roads/vegetation/height)

###### import libraries ######
# Standard libraries
import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

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
from model.utils.config_utils import get_prediction_channels
from helpers.load_configs import load_configs
from model.lst_predictor.predictor import LSTPredictor

# Load CUDA
load_cuda()


def train_lst_predictor():
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
        print("LST Predictor DDP Training")
        print(f"{'='*60}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(yaml.dump(config, default_flow_style=False))
    
    dataset_config = config['dataset_params']
    ldm_config = config.get('ldm_params', {})
    train_config = config['train_params']
    
    # Get semantic-specific config
    semantic_ldm_config = ldm_config.get('semantic', ldm_config)
    condition_config = semantic_ldm_config.get('condition_config', {})
    semantic_train_config = train_config.get('semantic', train_config)
    
    # Get semantic channels from condition config
    semantic_channels = get_prediction_channels(condition_config)
    
    if not semantic_channels:
        # Fallback to default
        semantic_channels = [
            'osm:buildings',
            'osm:streets',
            'env:vegetation',
            'osm:buildings_heights'
        ]
    
    num_semantic_channels = len(semantic_channels)
    
    # Optionally include NDVI as additional input
    include_ndvi = train_config.get('lst_predictor_use_ndvi', True)
    if include_ndvi:
        num_input_channels = num_semantic_channels + 1  # +1 for NDVI
    else:
        num_input_channels = num_semantic_channels
    
    # Create output directories
    task_name = train_config.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    
    cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print("Loading Urban Dataset for LST Predictor Training")
        print(f"{'='*50}")
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode='semantic',
        use_latents=False,
        latent_path=None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Semantic channels ({num_semantic_channels}): {semantic_channels}")
        print(f"✓ Include NDVI: {include_ndvi}")
    
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
        batch_size=train_config.get('lst_predictor_batch_size', 16),
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    ########## Create Model #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing LST Predictor")
        print(f"{'='*50}")
    
    # LST Predictor model
    hidden_dims = train_config.get('lst_predictor_hidden_dims', [64, 128, 256])
    model = LSTPredictor(
        in_channels=num_input_channels,
        hidden_dims=hidden_dims,
        out_channels=1
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
        print(f"✓ Created LST Predictor with {param_count:.2f}M parameters")
        print(f"  - Input channels: {num_input_channels}")
        print(f"  - Hidden dims: {hidden_dims}")
    
    ########## Training Setup #############
    num_epochs = train_config.get('lst_predictor_epochs', 50)
    base_lr = train_config.get('lst_predictor_lr', 1e-4)
    batch_size = train_config.get('lst_predictor_batch_size', 16)
    
    # Scale learning rate with world size
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    # Loss function
    loss_fn_type = train_config.get('lst_predictor_loss', 'huber')  # 'l1', 'l2', 'huber'
    if loss_fn_type == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_fn_type == 'l2':
        loss_fn = nn.MSELoss()
    else:  # huber
        loss_fn = nn.SmoothL1Loss()
    
    # Optionally weight loss inside mask
    use_mask_weighting = train_config.get('lst_predictor_mask_weighting', True)
    mask_weight = train_config.get('lst_predictor_mask_weight', 3.0)
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr} from base {base_lr}")
        print(f"✓ Batch size per GPU: {batch_size}")
        print(f"✓ Effective batch size: {batch_size * world_size}")
        print(f"✓ Loss function: {loss_fn_type}")
        print(f"✓ Mask weighting: {use_mask_weighting} (weight: {mask_weight})")
    
    ########## Training Loop #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Starting Training with {num_epochs} epochs")
        print(f"{'='*50}")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch_idx in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        losses = []
        
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Unpack data
            if len(data) == 2:
                im, cond_input = data
            else:
                im = data
                cond_input = {}
            
            # Build semantic tensor and extract LST target
            if 'image' in cond_input and 'meta' in cond_input:
                semantic_tensor = []
                lst_target = None
                ndvi_channel = None
                mask = None
                
                meta = cond_input['meta']
                # meta is a list of dicts (one per batch item), get spatial_names from first item
                spatial_names = meta[0].get('spatial_names', []) if isinstance(meta, list) and len(meta) > 0 else []
                
                # Extract semantic channels
                for sem_ch in semantic_channels:
                    found = False
                    for idx, name in enumerate(spatial_names):
                        if sem_ch == name:
                            semantic_tensor.append(cond_input['image'][:, idx:idx+1, :, :])
                            found = True
                            break
                    
                    if not found:
                        B, _, H, W = cond_input['image'].shape
                        semantic_tensor.append(torch.zeros(B, 1, H, W, device=cond_input['image'].device))
                
                # Extract LST target
                for idx, name in enumerate(spatial_names):
                    if 'landsat_surface_temp' in name or 'LST' in name or 'lst' in name:
                        lst_target = cond_input['image'][:, idx:idx+1, :, :]
                        break
                
                # Extract NDVI if needed
                if include_ndvi:
                    for idx, name in enumerate(spatial_names):
                        if 'ndvi' in name.lower() and '_context' not in name:
                            ndvi_channel = cond_input['image'][:, idx:idx+1, :, :]
                            break
                
                # Extract inpainting mask from image channels
                try:
                    mask_idx = spatial_names.index('inpaint_mask')
                    mask = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                except (ValueError, IndexError):
                    mask = None
                
                # Build input tensor
                semantic_input = torch.cat(semantic_tensor, dim=1)
                
                if include_ndvi and ndvi_channel is not None:
                    semantic_input = torch.cat([semantic_input, ndvi_channel], dim=1)
                elif include_ndvi:
                    # Create zero NDVI channel if not found
                    B, _, H, W = semantic_input.shape
                    semantic_input = torch.cat([semantic_input, torch.zeros(B, 1, H, W, device=semantic_input.device)], dim=1)
                
            else:
                # No conditioning, skip this batch
                continue
            
            if lst_target is None:
                # No LST target found, skip
                continue
            
            semantic_input = semantic_input.float().to(device)
            lst_target = lst_target.float().to(device)
            
            # Forward pass
            lst_pred = model(semantic_input)
            
            # Compute loss
            if use_mask_weighting and mask is not None:
                mask = mask.float().to(device)
                # Weighted loss: higher weight inside mask
                per_pixel_loss = F.mse_loss(lst_pred, lst_target, reduction='none') if loss_fn_type == 'l2' else \
                                 F.l1_loss(lst_pred, lst_target, reduction='none') if loss_fn_type == 'l1' else \
                                 F.smooth_l1_loss(lst_pred, lst_target, reduction='none')
                
                # Weight: higher inside mask
                weight = mask * mask_weight + (1 - mask) * 1.0
                loss = (per_pixel_loss * weight).mean()
            else:
                loss = loss_fn(lst_pred, lst_target)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            global_step += 1
            
            # Update progress bar (main process only)
            if is_main:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Synchronize epoch metrics across GPUs
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary (main process only)
        epoch_loss = np.mean(losses)
        if is_main:
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
        
        # Save best checkpoint (main process only)
        if is_main and epoch_loss < best_loss:
            best_loss = epoch_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = os.path.join(
                out_dir,
                train_config.get('lst_predictor_ckpt_name', 'lst_predictor_best.pth')
            )
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'in_channels': num_input_channels,
                    'hidden_dims': hidden_dims,
                    'semantic_channels': semantic_channels,
                    'include_ndvi': include_ndvi
                }
            }, checkpoint_path)
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
        
        # Save periodic checkpoint (main process only)
        if is_main and (epoch_idx + 1) % 10 == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            periodic_path = os.path.join(
                out_dir,
                f'lst_predictor_epoch_{epoch_idx + 1}.pth'
            )
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'config': {
                    'in_channels': num_input_channels,
                    'hidden_dims': hidden_dims,
                    'semantic_channels': semantic_channels,
                    'include_ndvi': include_ndvi
                }
            }, periodic_path)
            print(f'✓ Saved checkpoint: {periodic_path}')
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
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
        print(f"✓ LST Predictor Training Complete at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}!")
        print(f"✓ Best loss: {best_loss:.4f}")
        print(f"✓ Total Training Time: {hours}h {minutes}m {seconds}s ({training_time:.2f} seconds)")
        print(f"{'='*60}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    train_lst_predictor()

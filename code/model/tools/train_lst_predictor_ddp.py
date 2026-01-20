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
from model.utils.layer_config import get_layer_channels_from_names
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
    train_config = config['train_params']
    layers_registry = config.get('layers', {})
    vae_groups = config.get('vae_groups', {})
    
    # Get semantic VAE group configuration
    if 'semantic' not in vae_groups:
        raise ValueError("Config must define 'semantic' VAE group for LST predictor training")
    
    semantic_vae_config = vae_groups['semantic']
    semantic_layers = semantic_vae_config.get('layers', [])
    
    if not semantic_layers:
        raise ValueError("Semantic VAE group has no layers defined")
    
    # Get LST normalization range from config for loss interpretation
    lst_layer_config = layers_registry.get('lst', {})
    lst_normalize_params = lst_layer_config.get('normalize_params', {})
    lst_max_celsius = lst_normalize_params.get('max', 80)  # Default to 80°C if not specified
    
    # Count channels in semantic layers
    num_semantic_channels = 0
    for layer_name in semantic_layers:
        if layer_name not in layers_registry:
            raise ValueError(f"Layer '{layer_name}' not found in layers registry")
        layer_config = layers_registry[layer_name]
        channels = layer_config.get('channels', None)
        if channels:
            num_semantic_channels += len(channels)
        else:
            num_semantic_channels += 1  # Binary or single-channel layer
    
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
    
    # Use default mode to get access to ALL layers (RGB as prediction, rest as conditioning)
    # This gives us semantic layers + LST + NDVI all in the conditioning dict
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode='default',  # Get all layers: RGB as image, rest as conditioning
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Semantic layers: {semantic_layers}")
        print(f"✓ Semantic channels: {num_semantic_channels}")
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
            
            # Extract data from batch (default mode returns: rgb_image, cond_dict)
            # cond_dict contains: {'image': [B, C_cond, H, W], 'meta': [list of dicts]}
            if len(data) == 2:
                rgb_image, cond_dict = data
                
                # Extract conditioning image (contains all non-RGB layers)
                if 'image' not in cond_dict or cond_dict['image'] is None:
                    # No conditioning channels, skip
                    continue
                
                cond_image = cond_dict['image']
                
                # Extract metadata
                meta = cond_dict.get('meta', {})
                # meta is a list of dicts (one per batch item)
                if isinstance(meta, list) and len(meta) > 0:
                    channel_names = meta[0].get('channel_names', [])
                    layer_names = meta[0].get('layer_names', [])
                else:
                    channel_names = []
                    layer_names = []
            else:
                # Unexpected format, skip
                continue
            
            if not channel_names:
                # No metadata, skip
                continue
            
            # cond_image is [B, C_cond, H, W] containing all non-RGB layers
            # This includes: semantic layers (buildings, streets, etc.) + LST + NDVI + mask
            
            # Build semantic input tensor from semantic layers only
            semantic_tensor_list = []
            for layer_name in semantic_layers:
                layer_matches = get_layer_channels_from_names(channel_names, layer_name)
                if not layer_matches:
                    # Layer not found - create zero channel
                    if len(semantic_tensor_list) > 0:
                        B, _, H, W = semantic_tensor_list[0].shape
                    else:
                        B, _, H, W = cond_image.shape[0], 1, cond_image.shape[2], cond_image.shape[3]
                    semantic_tensor_list.append(torch.zeros(B, 1, H, W, device=cond_image.device))
                    if is_main and global_step == 0:
                        print(f"⚠ Warning: Semantic layer '{layer_name}' not found - using zeros")
                    continue
                
                # Add all channels for this layer
                for idx, ch_name in layer_matches:
                    semantic_tensor_list.append(cond_image[:, idx:idx+1, :, :])
            
            if not semantic_tensor_list:
                # No semantic channels found, skip
                if is_main and global_step == 0:
                    print(f"⚠ Warning: No semantic layers found in batch")
                continue
            
            # Extract LST target from conditioning
            lst_target = None
            lst_matches = get_layer_channels_from_names(channel_names, 'lst')
            if lst_matches:
                idx, _ = lst_matches[0]
                lst_target = cond_image[:, idx:idx+1, :, :]
            
            if lst_target is None:
                # Try alternative names
                for idx, ch_name in enumerate(channel_names):
                    if 'landsat_surface_temp' in ch_name.lower() or 'surface_temp' in ch_name.lower():
                        lst_target = cond_image[:, idx:idx+1, :, :]
                        break
            
            if lst_target is None:
                if is_main and global_step == 0:
                    print(f"⚠ Warning: LST target not found in batch")
                    print(f"  Available layers: {set(layer_names)}")
                    print(f"  Sample channels: {channel_names[:10]}")  # Show first 10
                continue
            
            # Extract NDVI if needed
            ndvi_channel = None
            if include_ndvi:
                ndvi_matches = get_layer_channels_from_names(channel_names, 'ndvi')
                if ndvi_matches:
                    idx, _ = ndvi_matches[0]
                    ndvi_channel = cond_image[:, idx:idx+1, :, :]
            
            # Extract mask if available
            mask = None
            mask_matches = get_layer_channels_from_names(channel_names, 'inpainting_mask')
            if mask_matches:
                idx, _ = mask_matches[0]
                mask = cond_image[:, idx:idx+1, :, :]
            
            # Build semantic input
            semantic_input = torch.cat(semantic_tensor_list, dim=1)
            
            # Add NDVI if needed
            if include_ndvi:
                if ndvi_channel is not None:
                    semantic_input = torch.cat([semantic_input, ndvi_channel], dim=1)
                else:
                    # Create zero NDVI channel
                    B, _, H, W = semantic_input.shape
                    semantic_input = torch.cat([semantic_input, torch.zeros(B, 1, H, W, device=semantic_input.device)], dim=1)
                    if is_main and global_step == 0:
                        print(f"⚠ Warning: NDVI not found - using zeros")
            
            semantic_input = semantic_input.float().to(device)
            lst_target = lst_target.float().to(device)
            
            # Forward pass
            lst_pred = model(semantic_input)
            
            # Compute loss (LST is normalized to [0, 1] by dataset, so loss is in normalized units)
            # To interpret: multiply loss by 80 to get approximate error in °C
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
            # Convert loss to Celsius for interpretability (loss is in normalized [0, 1] range)
            epoch_loss_celsius = epoch_loss * lst_max_celsius
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f} (~{epoch_loss_celsius:.2f}°C)')
        
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
                    'semantic_layers': semantic_layers,
                    'include_ndvi': include_ndvi
                }
            }, checkpoint_path)
            best_loss_celsius = best_loss * lst_max_celsius
            print(f"  ✓ Saved best model (loss: {best_loss:.4f} ~{best_loss_celsius:.2f}°C)")
        
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
                    'semantic_layers': semantic_layers,
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
        best_loss_celsius = best_loss * lst_max_celsius
        print(f"\n{'='*60}")
        print(f"✓ LST Predictor Training Complete at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}!")
        print(f"✓ Best loss: {best_loss:.4f} (~{best_loss_celsius:.2f}°C)")
        print(f"✓ Total Training Time: {hours}h {minutes}m {seconds}s ({training_time:.2f} seconds)")
        print(f"{'='*60}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    train_lst_predictor()

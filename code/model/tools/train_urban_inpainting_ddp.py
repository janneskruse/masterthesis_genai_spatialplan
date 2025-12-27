# Training script for urban inpainting latent diffusion model with DDP

###### import libraries ######
# system libraries
import sys
import os
import time
import yaml
from tqdm import tqdm

# data science libraries
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as torchF

# local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.utils.config_utils import get_config_value
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()

def apply_cond_dropout(cond_input, spatial_names, drop_prob, drop_groups=("osm", "env")):
    if 'image' not in cond_input:
        return cond_input
    if np.random.rand() >= drop_prob:
        return cond_input

    x = cond_input['image']
    keep = torch.ones((x.shape[1],), device=x.device, dtype=x.dtype)

    for i, name in enumerate(spatial_names):
        # 'inpaint_mask', 'masked_image:blue' (or whatever in _append_spatial names),
        # 'osm:buildings', 'env:...'
        if any(name.startswith(g + ":") for g in drop_groups):
            keep[i] = 0.0

    cond_input['image'] = x * keep.view(1, -1, 1, 1)
    return cond_input


def compute_noise_loss(noise_pred, noise, mask_latent, loss_type, mask_loss_weight=8.0, outside_weight=0.0):
    """
    loss_type: "masked" or "weighted"
    outside_weight is important in hard mode (usually 0.0).
    """
    if loss_type == "masked":
        return torchF.mse_loss(noise_pred * mask_latent, noise * mask_latent)

    # weighted full-image MSE
    per_pix = torchF.mse_loss(noise_pred, noise, reduction='none')
    w = outside_weight * (1.0 - mask_latent) + mask_loss_weight * mask_latent
    return (per_pix * w).mean()


def train():
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
        print("="*50)
        print("Urban Inpainting DDP Training Configuration")
        print("="*50)
        print(yaml.dump(config, default_flow_style=False))
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    latent_path = f'{big_data_storage_path}/results/{train_config["task_name"]}/{train_config.get("latents_dir_name", "vae_ddp_latents")}'
    use_latents = os.path.exists(latent_path) and len(os.listdir(latent_path)) > 0
    cache_dir = f"{big_data_storage_path}/processed/{train_config.get('task_name', 'urban_inpainting')}"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    # Create output directory
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)
    
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    if is_main:
        print(f"\n✓ Created noise scheduler with {diffusion_config['num_timesteps']} timesteps")
    
    ########## Load Dataset #############
    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    assert condition_config is not None, "Condition config required for urban inpainting"
    
    if is_main:
        print("\n" + "="*50)
        print("Loading Urban Dataset")
        print("="*50)
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        use_latents=use_latents,
        latent_path=latent_path if use_latents else None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Using latents: {use_latents}")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Conditioning types: {condition_config['condition_types']}")
    
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
        batch_size=train_config['ldm_batch_size'],
        shuffle=(sampler is None),
        num_workers=0,  # Set to 0 for DDP to avoid issues
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    ########## Create Model #############
    if is_main:
        print("\n" + "="*50)
        print("Initializing Models")
        print("="*50)
    
    # Instantiate the U-Net model
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
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
            print(f"✓ Wrapped U-Net in DistributedDataParallel")
    
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        print(f"✓ Created U-Net with {sum(p.numel() for p in model_unwrapped.parameters())/1e6:.2f}M parameters")
    
    # Load VAE if not using latents
    vae = None
    if not use_latents:
        if is_main:
            print('\n⚠ Loading VAE model as latents not present')
        
        vae = VAE(
            im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config
        ).to(device)
        vae.eval()
        
        # Load VAE checkpoint if exists
        vae_path = os.path.join(
            train_config['task_name'],
            train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth')
        )
        if os.path.exists(vae_path):
            if is_main:
                print(f'✓ Loaded VAE checkpoint from {vae_path}')
            vae.load_state_dict(torch.load(vae_path, map_location=device))
        else:
            raise Exception(f'VAE checkpoint not found at {vae_path}. Please train VAE first.')
        
        # Freeze VAE
        for param in vae.parameters():
            param.requires_grad = False
    
    ########## Training Setup #############
    num_epochs = train_config['ldm_epochs']
    
    # Scale learning rate with world size (linear scaling rule)
    base_lr = train_config['ldm_lr']
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    # Loss weights with warmup
    mask_loss_weight = train_config.get('mask_loss_weight', 2.0)
    seg_loss_weight_initial = train_config.get('seg_loss_weight_initial', 0.1)
    seg_loss_weight_final = train_config.get('seg_loss_weight_final', 0.5)
    env_loss_weight_initial = train_config.get('env_loss_weight_initial', 0.1)
    env_loss_weight_final = train_config.get('env_loss_weight_final', 0.3)
    
    # Check if model has auxiliary prediction heads
    model_unwrapped = model.module if hasattr(model, 'module') else model
    has_seg_head = hasattr(model_unwrapped, 'segmentation_head') and model_unwrapped.segmentation_head is not None
    has_env_head = hasattr(model_unwrapped, 'environmental_head') and model_unwrapped.environmental_head is not None
    
    # Conditioning dropout probability
    inpainting_cfg = train_config.get('inpainting', {})
    cond_cfg = inpainting_cfg.get('cfg', {})
    cond_drop_prob = cond_cfg.get('drop_prob', 0.1)
    drop_groups = tuple(cond_cfg.get('drop_groups', ["osm", "env"]))
    
    mode = inpainting_cfg.get('mode', 'hard')         # "hard" | "sdlike"
    loss_type = inpainting_cfg.get('loss', 'masked')  # "masked" | "weighted"
    mask_loss_weight = inpainting_cfg.get('mask_loss_weight', 8.0)
    if mode == "hard" and loss_type == "weighted":
        outside_weight = inpainting_cfg.get('outside_weight', 0.0)  # default 0.0, good
    elif mode == "sdlike" and loss_type == "weighted":
        outside_weight = inpainting_cfg.get('outside_weight', 1.0)  # default 1.0 makes sense
    
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        print(f"✓ Batch size per GPU: {train_config['ldm_batch_size']}")
        print(f"✓ Effective batch size: {train_config['ldm_batch_size'] * world_size}")
        print(f"✓ Mask loss weight: {mask_loss_weight}")
        print(f"✓ OSM loss weight warmup: {seg_loss_weight_initial:.2f} → {seg_loss_weight_final:.2f}")
        print(f"✓ Env loss weight warmup: {env_loss_weight_initial:.2f} → {env_loss_weight_final:.2f}")
        print(f"✓ OSM segmentation head enabled: {has_seg_head}")
        print(f"✓ Environmental head enabled: {has_env_head}")
        print(f"✓ Conditioning dropout: {cond_drop_prob}")
    
    ########## Training Loop #############
    if is_main:
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
        # Calculate loss weights with warmup for this epoch
        epoch_progress = epoch_idx / num_epochs
        seg_loss_weight = seg_loss_weight_initial + epoch_progress * (seg_loss_weight_final - seg_loss_weight_initial)
        env_loss_weight = env_loss_weight_initial + epoch_progress * (env_loss_weight_final - env_loss_weight_initial)
        
        
        # Set epoch for distributed sampler
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        losses = []
        
        # Only show progress bar on main process
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            # Unpack data
            if len(data) == 2:
                im, cond_input = data
            else:
                im = data
                cond_input = None
            
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Encode to latent space if not using pre-computed latents
            if not use_latents and vae is not None:
                with torch.no_grad():
                    im, _ = vae.encode(im)
            
            ########## Handle Conditional Input ##########
            if cond_input is not None:
                # Move all tensors in cond_input to device
                for key in cond_input:
                    if key == 'meta':
                        continue  # Skip metadata
                    if isinstance(cond_input[key], torch.Tensor):
                        cond_input[key] = cond_input[key].to(device)
                
                # Get mask for loss weighting from spatial image
                if 'image' in cond_input and 'meta' in cond_input:
                    if isinstance(cond_input['meta'], list):
                        if len(cond_input['meta']) == 0:
                            spatial_names = []
                        else:
                            # Use first sample's metadata to get spatial names structure
                            # all images in the batch would have the same index for 'inpaint_mask'
                            spatial_names = cond_input['meta'][0].get('spatial_names', [])
                    else:
                        spatial_names = cond_input['meta'].get('spatial_names', [])
                    try:
                        # get index for inpaint_mask
                        mask_idx = spatial_names.index('inpaint_mask')
                        mask = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                        # Downsample mask to latent resolution
                        mask_latent = torchF.interpolate(
                            mask,
                            size=im.shape[-2:],
                            mode='nearest'
                        )
                    except (ValueError, KeyError):
                        print("⚠ 'inpaint_mask' not found in spatial conditioning names.")
                        # No mask found, use uniform weighting
                        mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
                else:
                    # No conditioning or mask, use uniform weighting
                    mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            else:
                mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            
            cond_input = apply_cond_dropout(cond_input, spatial_names, cond_drop_prob, drop_groups)
            
            if is_main and global_step == 0:
                print(f"\n{'='*50}")
                print("Mask Validation (First Batch)")
                print(f"{'='*50}")
                print(f"Mask stats: min={mask_latent.min().item():.4f}, max={mask_latent.max().item():.4f}, mean={mask_latent.mean().item():.4f}")
                print(f"Mask shape: {mask_latent.shape}, Latent shape: {im.shape}")
                print(f"Mask unique values: {torch.unique(mask_latent)}")
                if 'image' in cond_input and 'meta' in cond_input:
                    print(f"Spatial conditioning channels: {spatial_names}")
                    print(f"Spatial conditioning shape: {cond_input['image'].shape}")
                print(f"{'='*50}\n")
            
            ########## Diffusion Forward Process ##########
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample random timestep
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (im.shape[0],)
            ).to(device)
            
            # Add noise to images according to timestep
            noisy_full = scheduler.add_noise(im, noise, t)
            
            if mode == "hard":
                # Keep known region (unmasked) fixed --> only noise the masked region
                noisy_im = mask_latent * noisy_full + (1 - mask_latent) * im
            else:  # "sdlike"
                # whole image is noised, and the inpainting constraint comes via conditioning + sampling clamp
                noisy_im = noisy_full
            
            # Predict noise (and optionally auxiliary outputs)
            if has_seg_head or has_env_head:
                outputs = model(noisy_im, t, cond_input=cond_input, return_segmentation=True)
                noise_pred = outputs[0]
                seg_pred = outputs[1] if len(outputs) > 1 else None
                env_pred = outputs[2] if len(outputs) > 2 else None
            else:
                noise_pred = model(noisy_im, t, cond_input=cond_input)
                seg_pred = None
                env_pred = None
            
            # Compute noise prediction loss with inpainting weighting
            loss = compute_noise_loss(noise_pred, noise, mask_latent, loss_type,
                                    mask_loss_weight=mask_loss_weight,
                                    outside_weight=outside_weight)

            
            # Add OSM segmentation loss if enabled
            if has_seg_head and seg_pred is not None and 'image' in cond_input and 'meta' in cond_input:
                osm_layers = condition_config.get('osm_layers', [])
                if len(osm_layers) > 0:
                    # Find OSM channels in spatial conditioning
                    osm_indices = []
                    for layer_name in osm_layers:
                        channel_name = f'osm:{layer_name}'
                        try:
                            idx = spatial_names.index(channel_name)
                            osm_indices.append(idx)
                        except ValueError:
                            pass
                    
                    if len(osm_indices) > 0:
                        # Extract ground truth OSM masks
                        osm_gt = cond_input['image'][:, osm_indices, :, :]
                        
                        # Downsample to latent resolution if needed
                        if seg_pred.shape[-2:] != osm_gt.shape[-2:]:
                            osm_gt = torchF.interpolate(osm_gt, size=seg_pred.shape[-2:], mode='nearest')
                        
                        # Binary cross-entropy loss for masks (only in masked region)
                        seg_loss = torchF.binary_cross_entropy_with_logits(
                            seg_pred * mask_latent,
                            osm_gt * mask_latent
                        )
                        
                        loss = loss + seg_loss_weight * seg_loss
                    else:
                        if is_main:
                            print(f"⚠ No OSM indices found for segmentation loss on idx {batch_idx} in epoch {epoch_idx}.")
                            print(f"   Spatial names: {spatial_names}")
                            print(f"   Meta: {cond_input['meta']}")
                            print(f"   OSM layers: {osm_layers}")
                        # Dummy loss to ensure gradients flow through seg head
                        seg_loss = (seg_pred * 0.0).sum()
                        loss = loss + seg_loss
                else:
                    if is_main:
                        print(f"⚠ No OSM layers specified in config for segmentation loss on idx {batch_idx} in epoch {epoch_idx}.")
                        print(f"   Spatial names: {spatial_names}")
                        print(f"   Meta: {cond_input['meta']}")
                        print(f"   OSM layers: {osm_layers}")
                    # Dummy loss to ensure gradients flow through seg head
                    seg_loss = (seg_pred * 0.0).sum()
                    loss = loss + seg_loss
            
            # Add environmental prediction loss if enabled
            if has_env_head and env_pred is not None and 'image' in cond_input and 'meta' in cond_input:
                # Use prediction layers (subset of conditioning layers)
                env_pred_layers = condition_config.get('environmental_prediction_layers',
                                                      condition_config.get('environmental_layers', []))
                if len(env_pred_layers) > 0:
                    # Find environmental channels in spatial conditioning (only for prediction layers)
                    env_indices = []
                    for layer_name in env_pred_layers:
                        channel_name = f'env:{layer_name}'
                        try:
                            idx = spatial_names.index(channel_name)
                            env_indices.append(idx)
                        except ValueError:
                            pass
                    
                    if len(env_indices) > 0:
                        # Extract ground truth environmental data
                        env_gt = cond_input['image'][:, env_indices, :, :]
                        
                        # Downsample to latent resolution if needed
                        if env_pred.shape[-2:] != env_gt.shape[-2:]:
                            env_gt = torchF.interpolate(env_gt, size=env_pred.shape[-2:], mode='bilinear', align_corners=False)
                        
                        # MSE loss for continuous environmental values (only in masked region)
                        env_loss = torchF.mse_loss(
                            env_pred * mask_latent,
                            env_gt * mask_latent
                        )
                        
                        loss = loss + env_loss_weight * env_loss
                    else:
                        if is_main:
                            print(f"⚠ No environmental indices found for environmental loss on idx {batch_idx} in epoch {epoch_idx}.")
                            print(f"   Spatial names: {spatial_names}")
                            print(f"   Meta: {cond_input['meta']}")
                            print(f"   Environmental layers: {env_pred_layers}")
                        # Dummy loss to ensure gradients flow through env head
                        env_loss = (env_pred * 0.0).sum()
                        loss = loss + env_loss
                else:
                    if is_main:
                        print(f"⚠ No environmental layers specified in config for environmental loss on idx {batch_idx} in epoch {epoch_idx}.")
                        print(f"   Spatial names: {spatial_names}")
                        print(f"   Meta: {cond_input['meta']}")
                        print(f"   Environmental layers: {env_pred_layers}")
                    # Dummy loss to ensure gradients flow through env head
                    env_loss = (env_pred * 0.0).sum()
                    loss = loss + env_loss
            
            losses.append(loss.item())
            loss.backward()
            
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            global_step += 1
            
            # Update progress bar (only on main process)
            if is_main:
                progress_bar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})
        
        # Synchronize losses across all ranks
        if world_size > 1:
            epoch_loss_tensor = torch.tensor(np.mean(losses)).to(device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
            epoch_loss = epoch_loss_tensor.item()
        else:
            epoch_loss = np.mean(losses)
        
        # Epoch summary (only on main process)
        if is_main:
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
            
            # Save checkpoint (only main process saves)
            model_unwrapped = model.module if hasattr(model, 'module') else model
            
            checkpoint_path = os.path.join(
                out_dir,
                train_config.get('ldm_ckpt_name', 'ddpm_urban_inpainting_ddp_ckpt.pth')
            )
            torch.save(model_unwrapped.state_dict(), checkpoint_path)
            
            # Save periodic checkpoint
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'ddpm_urban_inpainting_ddp_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(model_unwrapped.state_dict(), periodic_path)
                print(f'✓ Saved checkpoint: {periodic_path}')
        
        # Synchronize all ranks after checkpoint saving
        if world_size > 1:
            dist.barrier()
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        print('\n' + "="*50)
        print('✓ Training Complete!')
        print(f'✓ Total training time: {training_time/3600:.2f} hours')
        print("="*50)
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    train()

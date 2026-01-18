# Training script for semantic diffusion inpainting with DDP
# Stage 1: Generate semantic layouts (buildings/roads/vegetation/height) with temperature control

###### import libraries ######
# system libraries
import os
import time
import yaml
import random
import numpy as np
from tqdm import tqdm
import argparse

# data science libraries
import torch
import torch.nn.functional as torchF
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

# local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.utils.vae_registry import VAERegistry
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.utils.data_utils import collate_fn
from model.utils.diffusion_utils import apply_classifier_free_guidance_dropout
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.config_utils import build_unet_condition_config
from model.utils.layer_config import count_layer_channels, get_layer_info
from helpers.load_configs import load_configs, add_config_arguments

# Load CUDA
load_cuda()


def compute_noise_loss(noise_pred, noise, mask_latent, loss_type, mask_loss_weight=8.0, outside_weight=0.0):
    """
    Compute noise prediction loss for inpainting.
    
    Args:
        noise_pred: Predicted noise
        noise: Target noise
        mask_latent: Binary mask in latent space (1=regenerate, 0=keep)
        loss_type: "masked" or "weighted"
        mask_loss_weight: Weight for masked region
        outside_weight: Weight for outside region (usually 0.0 for hard mode)
        
    Returns:
        Loss scalar
    """
    if loss_type == "masked":
        return torchF.mse_loss(noise_pred * mask_latent, noise * mask_latent)

    # weighted full-image MSE
    per_pix = torchF.mse_loss(noise_pred, noise, reduction='none')
    w = outside_weight * (1.0 - mask_latent) + mask_loss_weight * mask_latent
    return (per_pix * w).mean()


def train(mode: str = 'semantic'):
    """
    Generic diffusion training function supporting any diffusion stage defined in config.
    
    Args:
        mode: Diffusion stage name (e.g., 'semantic', 'satellite')
              Must match a key in config['diffusion_stages']
    """
    # Record training start time
    training_start_time = time.time()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    ###### setup config variables #######
    config = load_configs()
    data_config = config['dataset_params']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Diffusion Training: {mode.upper()}")
        print(f"{'='*60}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        print(f"✓ Diffusion Stage: {mode}")
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(yaml.dump(config, default_flow_style=False))
    
    diffusion_config = config['diffusion_params']
    train_config_global = config['train_params']
    
    # Validate diffusion stage exists
    diffusion_stages = config.get('diffusion_stages', {})
    if mode not in diffusion_stages:
        raise ValueError(
            f"Diffusion stage '{mode}' not found in config. "
            f"Available stages: {list(diffusion_stages.keys())}"
        )
    
    # Get diffusion stage configuration
    stage_config = diffusion_stages[mode]
    prediction_group = stage_config.get('prediction_group')
    unet_config = stage_config.get('unet_config', {})
    conditioning_config = stage_config.get('conditioning', {})
    inpainting_config = stage_config.get('inpainting', {})
    
    if not prediction_group:
        raise ValueError(f"Diffusion stage '{mode}' has no prediction_group defined")
    
    # Get prediction VAE group configuration
    vae_groups = config.get('vae_groups', {})
    if prediction_group not in vae_groups:
        raise ValueError(
            f"Prediction group '{prediction_group}' not found in vae_groups. "
            f"Available groups: {list(vae_groups.keys())}"
        )
    
    prediction_vae_config = vae_groups[prediction_group]
    layers_registry = config.get('layers', {})
    
    # Parse prediction layers and compute channels
    prediction_layers = prediction_vae_config.get('layers', [])
    if not prediction_layers:
        raise ValueError(f"Prediction group '{prediction_group}' has no layers defined")
    
    num_prediction_channels = 0
    layer_names = []
    for layer_name in prediction_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_prediction_channels += num_channels
        layer_names.append(layer_name)
    
    # Get VAE architecture config
    vae_arch_config = {
        'z_channels': prediction_vae_config.get('z_channels', 4),
        'down_channels': prediction_vae_config.get('down_channels', [32, 64, 128, 128]),
        'mid_channels': prediction_vae_config.get('mid_channels', [128, 128]),
        'down_sample': prediction_vae_config.get('down_sample', [True, True, True]),
        'attn_down': prediction_vae_config.get('attn_down', [False, False, False]),
        'norm_channels': prediction_vae_config.get('norm_channels', 32),
        'num_heads': prediction_vae_config.get('num_heads', 2),
        'num_down_layers': prediction_vae_config.get('num_down_layers', 2),
        'num_mid_layers': prediction_vae_config.get('num_mid_layers', 2),
        'num_up_layers': prediction_vae_config.get('num_up_layers', 2),
    }
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Diffusion Stage: {mode}")
        print(f"{'='*60}")
        print(f"✓ Prediction group: {prediction_group}")
        print(f"✓ Prediction layers: {layer_names}")
        print(f"✓ Total prediction channels: {num_prediction_channels}")
        print(f"✓ Latent channels: {vae_arch_config['z_channels']}")
        print(f"{'='*60}\n")
    
    # Get training configuration
    diffusion_training_config = train_config_global.get('diffusion_training', {})
    train_config = diffusion_training_config.get(mode, {})
    
    # Path to prediction VAE latents
    latent_dir_name = prediction_vae_config.get('latents_dir', f'{prediction_group}_latents')
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    latent_path = f'{big_data_storage_path}/results/{task_name}/{latent_dir_name}'
    use_existing_latents = os.path.exists(latent_path) and len(os.listdir(latent_path)) > 0
    
    cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    # Create output directory
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    samples_dir_name = f'{mode}_diffusion_samples'
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, samples_dir_name), exist_ok=True)
    
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
    if is_main:
        print("\n" + "="*50)
        print(f"Loading Urban Dataset for {mode.upper()} Diffusion Training")
        print("="*50)
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=f'diffusion:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Using existing latents: {use_existing_latents}")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Prediction layers: {layer_names}")
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(
        urban_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    batch_size = train_config.get('batch_size', 4)
    
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
        print("\n" + "="*50)
        print("Initializing Models")
        print("="*50)
    
    # Build condition_config for U-Net from stage conditioning configuration
    condition_config = build_unet_condition_config(stage_config, vae_groups)
    
    # Add condition_config to unet_config
    unet_config_with_cond = unet_config.copy()
    unet_config_with_cond['condition_config'] = condition_config
    
    # Instantiate the U-Net model for diffusion
    model = Unet(
        im_channels=vae_arch_config['z_channels'],
        model_config=unet_config_with_cond,
        mode=mode
    ).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if is_main:
            print("✓ Wrapped model in DistributedDataParallel")
    
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        print(f"✓ Created U-Net with {sum(p.numel() for p in model_unwrapped.parameters())/1e6:.2f}M parameters")
    
    # Load prediction VAE if no existing latents
    vae = None
    vae_registry = None
    if not use_existing_latents:
        if is_main:
            print(f"\nLoading {prediction_group.upper()} VAE...")
        
        # Use VAERegistry for cleaner management
        vae_registry = VAERegistry(vae_arch_config, device)
        vae = vae_registry.load_vae(
            group_name=prediction_group,
            checkpoint_path=os.path.join(out_dir, prediction_vae_config.get('checkpoint_name', f'{prediction_group}_vae_ckpt.pth')),
            num_channels=num_prediction_channels,
            is_main=is_main
        )
        
        # Freeze VAE to prevent gradient updates during diffusion training
        vae_registry.freeze(prediction_group)
        vae.eval()
    
    ########## Training Setup #############
    num_epochs = train_config.get('epochs', 300)
    
    # Scale learning rate with world size
    base_lr = train_config.get('lr', 0.00001)
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    # Inpainting configuration
    inpainting_mode = inpainting_config.get('mode', 'hard')         # "hard" | "sdlike"
    loss_type = inpainting_config.get('loss', 'masked')  # "masked" | "weighted"
    mask_loss_weight = inpainting_config.get('mask_loss_weight', 8.0)
    outside_weight = 1.0
    if inpainting_mode == "hard" and loss_type == "weighted":
        outside_weight = inpainting_config.get('outside_weight', 0.0)
    elif inpainting_mode == "sdlike" and loss_type == "weighted":
        outside_weight = inpainting_config.get('outside_weight', 1.0)
    
    # Classifier-free guidance dropout
    cfg_config = inpainting_config.get('cfg', {})
    cond_drop_prob = cfg_config.get('drop_prob', 0.1)
    drop_groups = cfg_config.get('drop_groups', [])  # Which latent groups to drop
    
    # Image save frequency
    img_save_steps = train_config_global.get('img_save_steps', 1000)
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        print(f"✓ Batch size per GPU: {batch_size}")
        print(f"✓ Effective batch size: {batch_size * world_size}")
        print(f"✓ Inpainting mode: {inpainting_mode}")
        print(f"✓ Loss type: {loss_type}")
        print(f"✓ Mask loss weight: {mask_loss_weight}")
        print(f"✓ Conditioning dropout: {cond_drop_prob}")
    
    ########## Training Loop #############
    if is_main:
        print("\n" + "="*50)
        print(f"Starting Training")
        print("="*50)
    
    global_step = 0
    
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
                prediction_data, cond_input = data
            else:
                prediction_data = data
                cond_input = {}
            
            prediction_data = prediction_data.float().to(device)
            
            # Move ALL conditioning tensors to device (pixel-space and latent-space groups)
            if 'image' in cond_input:
                cond_input['image'] = cond_input['image'].float().to(device)
            
            # Move latent-space conditioning groups to device
            latent_group_keys = [k for k in cond_input.keys() if k not in ['image', 'meta']]
            for group_key in latent_group_keys:
                cond_input[group_key] = cond_input[group_key].float().to(device)
            
            # Encode prediction to latent space
            if use_existing_latents:
                # Precomputed latents
                im_latent = prediction_data
            else:
                # Encode on-the-fly
                with torch.no_grad():
                    im_latent, _, _ = vae.encode(prediction_data)
            
            # Extract inpainting mask from conditioning (already in latent space)
            mask_latent = None
            if 'image' in cond_input and 'meta' in cond_input:
                # Access pixel_space_names from metadata (first item, same across batch)
                pixel_space_names = cond_input['meta'][0].get('pixel_space_names', [])
                
                if pixel_space_names:
                    try:
                        mask_idx = pixel_space_names.index('inpainting_mask')
                        mask_latent = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                    except (ValueError, IndexError):
                        pass
            
            # Fallback: create full mask if not provided
            if mask_latent is None:
                mask_latent = torch.ones_like(im_latent[:, :1, :, :])
            
            # Sample timestep
            t = torch.randint(0, scheduler.num_timesteps, (im_latent.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(im_latent)
            
            # Apply noise according to inpainting mode
            if inpainting_mode == "hard":
                # Hard inpainting: only add noise inside mask
                noisy_im = im_latent.clone()
                noisy_region = scheduler.add_noise(im_latent, noise, t)
                noisy_im = mask_latent * noisy_region + (1 - mask_latent) * im_latent
            else:
                # SD-like: add noise everywhere
                noisy_im = scheduler.add_noise(im_latent, noise, t)
            
            # Apply classifier-free guidance dropout to ALL conditioning
            if cond_drop_prob > 0:
                cond_input = apply_classifier_free_guidance_dropout(
                    cond_input,
                    drop_prob=cond_drop_prob,
                    drop_groups=drop_groups,
                    drop_pixel_space=True
                )
            
            # First batch validation
            if is_main and global_step == 0:
                print(f"\n{'='*50}")
                print("First Batch Validation")
                print(f"{'='*50}")
                print(f"Prediction latent shape: {im_latent.shape}")
                print(f"Mask latent shape: {mask_latent.shape}")
                print(f"Mask stats: min={mask_latent.min().item():.4f}, max={mask_latent.max().item():.4f}, mean={mask_latent.mean().item():.4f}")
                
                # Pixel-space conditioning
                if 'image' in cond_input:
                    print(f"\nPixel-space conditioning:")
                    print(f"  Shape: {cond_input['image'].shape}")
                    print(f"  Device: {cond_input['image'].device}")
                    print(f"  Dtype: {cond_input['image'].dtype}")
                    # Access pixel_space_names from metadata
                    pixel_space_names = cond_input['meta'][0].get('pixel_space_names', [])
                    if pixel_space_names:
                        print(f"  Channels: {pixel_space_names}")
                
                # Latent-space conditioning groups
                latent_cond_groups = [k for k in cond_input.keys() if k not in ['image', 'meta']]
                if latent_cond_groups:
                    print(f"\nLatent-space conditioning groups: {latent_cond_groups}")
                    for group_name in latent_cond_groups:
                        group_tensor = cond_input[group_name]
                        print(f"  {group_name}:")
                        print(f"    Shape: {group_tensor.shape}")
                        print(f"    Device: {group_tensor.device}")
                        print(f"    Dtype: {group_tensor.dtype}")
                        print(f"    Stats: min={group_tensor.min().item():.4f}, max={group_tensor.max().item():.4f}, mean={group_tensor.mean().item():.4f}")
                
                # CFG dropout config
                print(f"\nCFG Dropout:")
                print(f"  Drop probability: {cond_drop_prob}")
                print(f"  Drop groups: {drop_groups}")
                
                print(f"{'='*50}\n")
            
            # Predict noise
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            
            # Compute loss
            loss = compute_noise_loss(
                noise_pred, noise, mask_latent, 
                loss_type, mask_loss_weight, outside_weight
            )
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            global_step += 1
            
            # Update progress bar
            if is_main:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save sample predictions periodically
            if is_main and global_step % img_save_steps == 0:
                with torch.no_grad():
                    model.eval()
                    
                    # Generate a few samples
                    num_samples = min(4, im_latent.shape[0])
                    
                    # Start from pure noise in masked region
                    x_sample = im_latent[:num_samples].clone()
                    if mode == "hard":
                        x_sample = mask_latent[:num_samples] * torch.randn_like(x_sample) + (1 - mask_latent[:num_samples]) * x_sample
                    else:
                        x_sample = torch.randn_like(x_sample)
                    
                    # Quick sampling (fewer steps for speed)
                    sample_steps = min(50, scheduler.num_timesteps)
                    step_size = scheduler.num_timesteps // sample_steps
                    
                    # Prepare conditioning for sampling (slice all keys to num_samples)
                    sample_cond = {}
                    for key in cond_input:
                        sample_cond[key] = cond_input[key][:num_samples]
                    
                    for i in reversed(range(0, scheduler.num_timesteps, step_size)):
                        t_sample = torch.full((num_samples,), i, device=device, dtype=torch.long)
                        noise_pred = model(x_sample, t_sample, cond_input=sample_cond)
                        
                        if inpainting_mode == "hard":
                            # Use inpainting scheduler
                            x_sample, _ = scheduler.sample_prev_timestep_inpainting(
                                x_sample, noise_pred, i,
                                im_latent[:num_samples],
                                mask_latent[:num_samples]
                            )
                        else:
                            x_sample, _ = scheduler.sample_prev_timestep(x_sample, noise_pred, i)
                    
                    # Decode to pixel space
                    if vae is not None:
                        sample_decoded = vae.decode(x_sample)
                    else:
                        sample_decoded = x_sample
                    
                    # Save visualization (simple normalization per channel)
                    vis_samples = []
                    for ch_idx in range(min(sample_decoded.shape[1], 4)):  # Show up to 4 channels
                        ch = sample_decoded[:, ch_idx:ch_idx+1, :, :]
                        # Normalize to [0, 1] range
                        ch_min = ch.min()
                        ch_max = ch.max()
                        if ch_max > ch_min:
                            ch = (ch - ch_min) / (ch_max - ch_min)
                        vis_samples.append(ch)
                    
                    if vis_samples:
                        vis_tensor = torch.cat(vis_samples, dim=1)
                        grid = make_grid(vis_tensor, nrow=num_samples, normalize=False, padding=2)
                        save_path = os.path.join(out_dir, samples_dir_name, f'sample_step_{global_step}.png')
                        save_image(grid, save_path)
                    
                    model.train()
        
        # Synchronize epoch metrics
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary
        if is_main:
            epoch_loss = np.mean(losses)
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        if is_main:
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_name = train_config.get('checkpoint_name', f'{mode}_diffusion_ckpt.pth')
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
            torch.save(model_to_save.state_dict(), checkpoint_path)
            
            # Periodic checkpoint
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'{mode}_diffusion_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(model_to_save.state_dict(), periodic_path)
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        print('\n' + "="*50)
        print(f'✓ {mode.upper()} Diffusion Training Complete!')
        print(f'✓ Total training time: {training_time/3600:.2f} hours')
        print("="*50)
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Diffusion Model DDP for Urban Inpainting')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, default='semantic',
                        help='Diffusion stage to train (must match a key in config diffusion_stages, e.g., "semantic", "satellite")')
    
    args = parser.parse_args()
    
    train(mode=args.mode)

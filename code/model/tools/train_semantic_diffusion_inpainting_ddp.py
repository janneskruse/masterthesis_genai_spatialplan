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
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.utils.config_utils import get_prediction_channels
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()


def apply_cond_dropout(cond_input, spatial_names, drop_prob, drop_groups=("osm", "env")):
    """
    Apply conditioning dropout for classifier-free guidance.
    
    Args:
        cond_input: Conditioning dictionary with 'image' key
        spatial_names: List of channel names
        drop_prob: Probability of dropping conditioning
        drop_groups: Tuple of prefix groups to drop (e.g., ('osm', 'env'))
        
    Returns:
        Modified cond_input with dropped channels
    """
    if 'image' not in cond_input:
        return cond_input
    if np.random.rand() >= drop_prob:
        return cond_input

    x = cond_input['image']
    keep = torch.ones((x.shape[1],), device=x.device, dtype=x.dtype)

    for i, name in enumerate(spatial_names):
        # Drop channels that start with specified group prefixes
        # Keep: 'inpaint_mask', 'masked_image', 'LST_target'
        # Drop: 'osm:*_context', 'env:*_context' based on drop_groups
        if any(name.endswith("_context") and name.startswith(g + ":") for g in drop_groups):
            keep[i] = 0.0

    cond_input['image'] = x * keep.view(1, -1, 1, 1)
    return cond_input


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
        print("Semantic Diffusion Inpainting DDP Training Configuration")
        print("="*50)
        print(yaml.dump(config, default_flow_style=False))
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    # Get semantic-specific configs
    ldm_config = config.get('ldm_params', {})
    semantic_ldm_config = ldm_config.get('semantic', ldm_config)
    autoencoder_config = config['autoencoder_params']
    semantic_autoencoder_config = autoencoder_config.get('semantic', autoencoder_config)
    
    # Get semantic training config
    semantic_train_config = train_config.get('semantic', {})
    
    # Extract semantic channels from condition config
    condition_config = semantic_ldm_config.get('condition_config', {})
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
    
    # Path to semantic VAE latents
    latent_dir_name = semantic_train_config.get('latents_dir_name', 'semantic_vae_ddp_latents')
    latent_path = f'{big_data_storage_path}/results/{train_config["task_name"]}/{latent_dir_name}'
    use_latents = os.path.exists(latent_path) and len(os.listdir(latent_path)) > 0
    
    cache_dir = f"{big_data_storage_path}/processed/{train_config.get('task_name', 'urban_inpainting')}/semantic"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    # Create output directory
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'semantic_samples'), exist_ok=True)
    
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
    if condition_config is None or not condition_config:
        raise ValueError("Condition config required for semantic inpainting")
    
    if is_main:
        print("\n" + "="*50)
        print("Loading Urban Dataset for Semantic Training")
        print("="*50)
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode='semantic',
        use_latents=use_latents,
        latent_path=latent_path if use_latents else None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Using latents: {use_latents}")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Semantic channels: {semantic_channels}")
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
        batch_size=semantic_train_config.get('ldm_batch_size', 4),
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
    
    # Instantiate the U-Net model for semantic diffusion
    model = Unet(
        im_channels=semantic_autoencoder_config['z_channels'],
        model_config=semantic_ldm_config
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
    
    # Load Semantic VAE if not using latents
    vae = None
    if not use_latents:
        if is_main:
            print("\nLoading Semantic VAE...")
        
        vae = VAE(
            im_channels=num_semantic_channels,
            model_config=semantic_autoencoder_config
        ).to(device)
        vae.eval()
        
        # Load VAE checkpoint
        vae_ckpt_name = semantic_train_config.get('autoencoder_ckpt_name', 'semantic_vae_ddp_ckpt.pth')
        vae_path = os.path.join(out_dir, vae_ckpt_name)
        if os.path.exists(vae_path):
            vae.load_state_dict(torch.load(vae_path, map_location=device))
            if is_main:
                print(f"✓ Loaded Semantic VAE from {vae_path}")
        else:
            if is_main:
                print(f"⚠ Semantic VAE checkpoint not found at {vae_path}")
        
        # Freeze VAE
        for param in vae.parameters():
            param.requires_grad = False
    
    ########## Training Setup #############
    num_epochs = semantic_train_config.get('ldm_epochs', 300)
    
    # Scale learning rate with world size
    base_lr = semantic_train_config.get('ldm_lr', 0.00001)
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    # Conditioning dropout probability for CFG
    inpainting_cfg = semantic_train_config.get('inpainting', {})
    cond_cfg = inpainting_cfg.get('cfg', {})
    cond_drop_prob = cond_cfg.get('drop_prob', 0.1)
    drop_groups = tuple(cond_cfg.get('drop_groups', ["osm", "env"]))
    
    mode = inpainting_cfg.get('mode', 'hard')         # "hard" | "sdlike"
    loss_type = inpainting_cfg.get('loss', 'masked')  # "masked" | "weighted"
    mask_loss_weight = inpainting_cfg.get('mask_loss_weight', 8.0)
    outside_weight = 1.0
    if mode == "hard" and loss_type == "weighted":
        outside_weight = inpainting_cfg.get('outside_weight', 0.0)
    elif mode == "sdlike" and loss_type == "weighted":
        outside_weight = inpainting_cfg.get('outside_weight', 1.0)
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        batch_size = semantic_train_config.get('ldm_batch_size', 4)
        print(f"✓ Batch size per GPU: {batch_size}")
        print(f"✓ Effective batch size: {batch_size * world_size}")
        print(f"✓ Inpainting mode: {mode}")
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
                im, cond_input = data
            else:
                im = data
                cond_input = {}
            
            # Build semantic tensor from conditioning
            if 'image' in cond_input and 'meta' in cond_input:
                semantic_tensor = []
                meta = cond_input['meta']
                # meta is a list of dicts (one per batch item), get spatial_names from first item
                spatial_names = meta[0].get('spatial_names', []) if isinstance(meta, list) and len(meta) > 0 else []
                
                # Extract semantic channels (non-context versions for target)
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
                
                semantic_input = torch.cat(semantic_tensor, dim=1)
            else:
                semantic_input = im
            
            semantic_input = semantic_input.float().to(device)
            
            # Get inpainting mask from image channels
            mask_full = None
            if 'image' in cond_input and 'meta' in cond_input:
                try:
                    mask_idx = spatial_names.index('inpaint_mask')
                    mask_full = cond_input['image'][:, mask_idx:mask_idx+1, :, :].to(device)
                except (ValueError, IndexError):
                    mask_full = None
            
            # Encode semantics to latent space
            if use_latents:
                # precomputed latents
                im_latent = im.float().to(device)
            else:
                with torch.no_grad():
                    _, im_latent, _, _ = vae.encoder(semantic_input)
            
            # Downsample mask to latent resolution
            if mask_full is not None:
                latent_size = im_latent.shape[-1]
                mask_latent = torchF.interpolate(
                    mask_full.float(),
                    size=(latent_size, latent_size),
                    mode='nearest'
                )
            else:
                mask_latent = torch.ones_like(im_latent[:, :1, :, :])
            
            # Sample timestep
            t = torch.randint(0, scheduler.num_timesteps, (im_latent.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(im_latent)
            
            # Apply noise according to mode
            if mode == "hard":
                # Hard inpainting: only add noise inside mask
                noisy_im = im_latent.clone()
                noisy_region = scheduler.add_noise(im_latent, noise, t)
                noisy_im = mask_latent * noisy_region + (1 - mask_latent) * im_latent
            else:
                # SD-like: add noise everywhere
                noisy_im = scheduler.add_noise(im_latent, noise, t)
            
            # Apply conditioning dropout for CFG
            if cond_drop_prob > 0 and 'meta' in cond_input:
                cond_input = apply_cond_dropout(
                    cond_input, 
                    spatial_names if 'meta' in cond_input else [],
                    cond_drop_prob,
                    drop_groups
                )
            
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
            if is_main and global_step % train_config.get('ldm_img_save_steps', 1000) == 0:
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
                    
                    # Prepare conditioning for sampling
                    sample_cond = {}
                    for key in cond_input:
                        if key == 'image':
                            sample_cond[key] = cond_input[key][:num_samples]
                        elif key == 'meta':
                            sample_cond[key] = {
                                k: v[:num_samples] if isinstance(v, torch.Tensor) else v 
                                for k, v in cond_input[key].items()
                            }
                    
                    for i in reversed(range(0, scheduler.num_timesteps, step_size)):
                        t_sample = torch.full((num_samples,), i, device=device, dtype=torch.long)
                        noise_pred = model(x_sample, t_sample, cond_input=sample_cond)
                        
                        if mode == "hard":
                            # Use inpainting scheduler
                            x_sample, _ = scheduler.sample_prev_timestep_inpainting(
                                x_sample, noise_pred, i,
                                im_latent[:num_samples],
                                mask_latent[:num_samples]
                            )
                        else:
                            x_sample, _ = scheduler.sample_prev_timestep(x_sample, noise_pred, i)
                    
                    # Decode to semantic space
                    if vae is not None:
                        semantic_sample = vae.decoder(x_sample)
                    else:
                        semantic_sample = x_sample
                    
                    # Save visualization
                    # Normalize semantic channels for visualization
                    vis_samples = []
                    for ch_idx in range(min(num_semantic_channels, semantic_sample.shape[1])):
                        ch = semantic_sample[:, ch_idx:ch_idx+1, :, :]
                        if ch_idx < len(semantic_channels) and 'height' in semantic_channels[ch_idx]:
                            ch = torch.clamp(ch / 100.0, 0, 1)  # Normalize height
                        else:
                            ch = torch.clamp(ch, 0, 1)  # Binary masks
                        vis_samples.append(ch)
                    
                    if vis_samples:
                        vis_tensor = torch.cat(vis_samples, dim=1)
                        grid = make_grid(vis_tensor, nrow=num_samples, normalize=False, padding=2)
                        save_path = os.path.join(out_dir, 'semantic_samples', f'sample_step_{global_step}.png')
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
            checkpoint_name = semantic_train_config.get('ldm_ckpt_name', 'semantic_ldm_ddp_ckpt.pth')
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
            torch.save(model_to_save.state_dict(), checkpoint_path)
            
            # Periodic checkpoint
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'semantic_ldm_ddp_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(model_to_save.state_dict(), periodic_path)
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        print('\n' + "="*50)
        print('✓ Semantic Diffusion Training Complete!')
        print(f'✓ Total training time: {training_time/3600:.2f} hours')
        print("="*50)
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    train()

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

def compute_inpainting_loss(noise_pred, noise, mask_latent, mask_loss_weight=2.0):
    """
    Compute loss with higher weight in masked region.
    
    Args:
        noise_pred: Predicted noise [B, C, H, W]
        noise: Ground truth noise [B, C, H, W]
        mask_latent: Mask in latent space [B, 1, H, W], 1=regenerate, 0=keep
        mask_loss_weight: Weight multiplier for masked region
    
    Returns:
        Weighted MSE loss
    """
    # Basic MSE loss
    loss = torchF.mse_loss(noise_pred, noise, reduction='none')
    
    # Apply mask weighting: higher weight where mask==1
    mask_weight = 1.0 + (mask_loss_weight - 1.0) * mask_latent
    weighted_loss = loss * mask_weight
    
    return weighted_loss.mean()


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
    
    # Loss weights
    mask_loss_weight = train_config.get('mask_loss_weight', 2.0)
    
    # Conditioning dropout probability
    cond_drop_prob = get_config_value(
        condition_config.get('image_condition_config', {}),
        'cond_drop_prob',
        0.1
    )
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        print(f"✓ Batch size per GPU: {train_config['ldm_batch_size']}")
        print(f"✓ Effective batch size: {train_config['ldm_batch_size'] * world_size}")
        print(f"✓ Mask loss weight: {mask_loss_weight}")
        print(f"✓ Conditioning dropout: {cond_drop_prob}")
    
    ########## Training Loop #############
    if is_main:
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
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
                
                # Apply conditioning dropout for classifier-free guidance
                if 'image' in cond_input and np.random.rand() < cond_drop_prob:
                    # Replace with zeros (unconditional)
                    cond_input['image'] = torch.zeros_like(cond_input['image'])
                
                # Get mask for loss weighting from spatial image
                if 'image' in cond_input and 'meta' in cond_input:
                    spatial_names = cond_input['meta']['spatial_names']
                    try:
                        mask_idx = spatial_names.index('inpaint_mask')
                        mask = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                        # Downsample mask to latent resolution
                        mask_latent = torchF.interpolate(
                            mask,
                            size=im.shape[-2:],
                            mode='nearest'
                        )
                    except (ValueError, KeyError):
                        # No mask found, use uniform weighting
                        mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
                else:
                    # No conditioning or mask, use uniform weighting
                    mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            else:
                mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            
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
            noisy_im = scheduler.add_noise(im, noise, t)
            
            # Predict noise
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            
            # Compute weighted loss
            loss = compute_inpainting_loss(
                noise_pred,
                noise,
                mask_latent,
                mask_loss_weight
            )
            
            losses.append(loss.item())
            loss.backward()
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

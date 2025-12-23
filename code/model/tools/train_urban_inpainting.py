# Training script for urban inpainting latent diffusion model

###### import libraries ######
# system libraries
import sys
import os
import yaml
from tqdm import tqdm

# data science libraries
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as torchF

# local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.utils.config_utils import get_config_value
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    
    ###### setup config variables #######
    config = load_configs()
    # repo_dir = config['repo_dir']
    data_config = config['data_config']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
        
    
    print("="*50)
    print("Urban Inpainting Training Configuration")
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
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    print(f"\n✓ Created noise scheduler with {diffusion_config['num_timesteps']} timesteps")
    
    ########## Load Dataset #############
    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    assert condition_config is not None, "Condition config required for urban inpainting"
    
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
    
    print(f"✓ Loaded {len(urban_dataset)} training patches")
    print(f"✓ Using latents: {use_latents}")
    print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
    print(f"✓ Conditioning types: {condition_config['condition_types']}")
    
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=train_config['ldm_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    ########## Create Model #############
    print("\n" + "="*50)
    print("Initializing Models")
    print("="*50)
    
    # Instantiate the U-Net model
    # Input channels: 4 (noisy latent) for now
    # We'll handle spatial conditioning via image_cond
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    model.train()
    
    print(f"✓ Created U-Net with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load VAE if not using latents
    vae = None
    if not use_latents:
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
            print(f'✓ Loaded VAE checkpoint from {vae_path}')
            vae.load_state_dict(torch.load(vae_path, map_location=device))
        else:
            raise Exception(f'VAE checkpoint not found at {vae_path}. Please train VAE first.')
        
        # Freeze VAE
        for param in vae.parameters():
            param.requires_grad = False
    
    ########## Training Setup #############
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    
    # Loss weights
    mask_loss_weight = train_config.get('mask_loss_weight', 2.0)
    seg_loss_weight = train_config.get('seg_loss_weight', 0.1)
    env_loss_weight = train_config.get('env_loss_weight', 0.1)
    
    # Check if model has auxiliary prediction heads
    has_seg_head = hasattr(model, 'segmentation_head') and model.segmentation_head is not None
    has_env_head = hasattr(model, 'environmental_head') and model.environmental_head is not None
    
    # Conditioning dropout probability
    cond_drop_prob = get_config_value(
        condition_config.get('image_condition_config', {}),
        'cond_drop_prob',
        0.1
    )
    
    print(f"\n✓ Training for {num_epochs} epochs")
    print(f"✓ Learning rate: {train_config['ldm_lr']}")
    print(f"✓ Batch size: {train_config['ldm_batch_size']}")
    print(f"✓ Mask loss weight: {mask_loss_weight}")
    print(f"✓ OSM segmentation loss weight: {seg_loss_weight}")
    print(f"✓ Environmental loss weight: {env_loss_weight}")
    print(f"✓ OSM segmentation head enabled: {has_seg_head}")
    print(f"✓ Environmental head enabled: {has_env_head}")
    print(f"✓ Conditioning dropout: {cond_drop_prob}")
    
    ########## Training Loop #############
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
        losses = []
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        
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
            
            if global_step == 0:
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
            noisy_im = scheduler.add_noise(im, noise, t)
            
            # Keep known region (unmasked) fixed --> only noise the masked region
            noisy_im = mask_latent * noisy_im + (1 - mask_latent) * im
            
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
            
            # Compute noise prediction loss (masked)
            noise_masked = mask_latent * noise
            noise_pred_masked = mask_latent * noise_pred
            loss = torchF.mse_loss(noise_pred_masked, noise_masked)
            
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
                        print(f"⚠ No OSM indices found for segmentation loss on idx {batch_idx} in epoch {epoch_idx}.")
                        print(f"   Spatial names: {spatial_names}")
                        print(f"   Meta: {cond_input['meta']}")
                        print(f"   OSM layers: {osm_layers}")
                        # Dummy loss to ensure gradients flow through seg head
                        seg_loss = (seg_pred * 0.0).sum()
                        loss = loss + seg_loss
                else:
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
                        print(f"⚠ No environmental indices found for environmental loss on idx {batch_idx} in epoch {epoch_idx}.")
                        print(f"   Spatial names: {spatial_names}")
                        print(f"   Meta: {cond_input['meta']}")
                        print(f"   Environmental layers: {env_pred_layers}")
                        # Dummy loss to ensure gradients flow through env head
                        env_loss = (env_pred * 0.0).sum()
                        loss = loss + env_loss
                else:
                    print(f"⚠ No environmental layers specified in config for environmental loss on idx {batch_idx} in epoch {epoch_idx}.")
                    print(f"   Spatial names: {spatial_names}")
                    print(f"   Meta: {cond_input['meta']}")
                    print(f"   Environmental layers: {env_pred_layers}")
                    # Dummy loss to ensure gradients flow through env head
                    env_loss = (env_pred * 0.0).sum()
                    loss = loss + env_loss
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})
        
        # Epoch summary
        epoch_loss = np.mean(losses)
        print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            train_config['task_name'],
            train_config.get('ldm_ckpt_name', 'ddpm_urban_inpainting_ckpt.pth')
        )
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch_idx + 1) % 10 == 0:
            periodic_path = os.path.join(
                train_config['task_name'],
                f'ddpm_urban_inpainting_epoch_{epoch_idx + 1}.pth'
            )
            torch.save(model.state_dict(), periodic_path)
            print(f'✓ Saved checkpoint: {periodic_path}')
    
    print('\n' + "="*50)
    print('✓ Training Complete!')
    print("="*50)


if __name__ == '__main__':
    train()

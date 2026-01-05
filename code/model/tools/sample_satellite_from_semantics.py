# Stage 2: Render satellite imagery from generated semantics
# Uses existing satellite diffusion model to generate realistic imagery conditioned on semantic layout

###### import libraries ######
# Standard libraries
import os
import sys
import argparse
import random
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob

# Visualization
from PIL import Image

# Data handling
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# Local libraries
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.config_utils import get_config_value, compute_patch_and_latent_sizes
from helpers.load_configs import load_configs
from helpers.indexed_outputs import get_next_run_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_semantic_samples(semantic_dir):
    """
    Load semantic samples from directory.
    
    Args:
        semantic_dir: Directory containing saved semantic samples
        
    Returns:
        List of dictionaries with semantic data
    """
    sample_files = sorted(glob.glob(os.path.join(semantic_dir, 'sample_*.pt')))
    
    if not sample_files:
        print(f"✗ No semantic samples found in {semantic_dir}")
        return []
    
    samples = []
    for sample_file in sample_files:
        sample_data = torch.load(sample_file, map_location='cpu')
        samples.append(sample_data)
    
    print(f"✓ Loaded {len(samples)} semantic samples from {semantic_dir}")
    return samples


def build_conditioning_from_semantics(
    semantic_tensor,
    semantic_channels,
    original_conditioning,
    mask,
    include_masked_rgb=True,
    include_lst_target=True
):
    """
    Build conditioning input for Stage 2 satellite rendering.
    
    Combines:
    - Generated semantic layout from Stage 1
    - Original context (masked RGB, LST target, NDVI) from outside mask
    - Inpainting mask
    
    Args:
        semantic_tensor: Generated semantic tensor [C, H, W]
        semantic_channels: List of semantic channel names
        original_conditioning: Original conditioning dict from dataset
        mask: Inpainting mask [1, H, W]
        include_masked_rgb: Whether to include masked RGB context
        include_lst_target: Whether to include LST target
        
    Returns:
        Conditioning dictionary for Stage 2 model
    """
    cond_input = {'image': [], 'meta': {'spatial_names': []}}
    
    # Add inpainting mask
    cond_input['image'].append(mask)
    cond_input['meta']['spatial_names'].append('inpaint_mask')
    
    # Add masked RGB if available and requested
    if include_masked_rgb and 'image' in original_conditioning and 'meta' in original_conditioning:
        spatial_names = original_conditioning['meta'].get('spatial_names', [])
        for idx, name in enumerate(spatial_names):
            if 'masked_image' in name:
                cond_input['image'].append(original_conditioning['image'][idx:idx+1, :, :])
                cond_input['meta']['spatial_names'].append(name)
    
    # Add generated semantics
    for ch_idx, ch_name in enumerate(semantic_channels):
        if ch_idx < semantic_tensor.shape[0]:
            cond_input['image'].append(semantic_tensor[ch_idx:ch_idx+1, :, :])
            cond_input['meta']['spatial_names'].append(ch_name)
    
    # Add continuous NDVI from original (outside mask only)
    if 'image' in original_conditioning and 'meta' in original_conditioning:
        spatial_names = original_conditioning['meta'].get('spatial_names', [])
        for idx, name in enumerate(spatial_names):
            if 'ndvi' in name.lower() and 'vegetation' not in name.lower():
                # Use original NDVI outside mask, generated semantics inside
                ndvi_orig = original_conditioning['image'][idx:idx+1, :, :]
                # Optionally blend or keep original outside
                cond_input['image'].append(ndvi_orig)
                cond_input['meta']['spatial_names'].append(name)
                break
    
    # Add LST target if requested
    if include_lst_target and 'image' in original_conditioning and 'meta' in original_conditioning:
        spatial_names = original_conditioning['meta'].get('spatial_names', [])
        for idx, name in enumerate(spatial_names):
            if 'landsat_surface_temp' in name or 'LST' in name or 'lst' in name:
                cond_input['image'].append(original_conditioning['image'][idx:idx+1, :, :])
                cond_input['meta']['spatial_names'].append(name)
                break
    
    # Stack image channels
    if cond_input['image']:
        cond_input['image'] = torch.cat(cond_input['image'], dim=0)
    
    return cond_input


def render_satellite_from_semantics(
    semantic_samples,
    model,
    scheduler,
    vae,
    train_config,
    diffusion_model_config,
    autoencoder_model_config,
    diffusion_config,
    dataset_config,
    big_data_storage_path,
    guidance_scale=7.5,
    include_masked_rgb=True,
    include_lst_target=True,
    overwrite_samples=False
):
    """
    Render satellite imagery from semantic layouts.
    
    Args:
        semantic_samples: List of semantic sample dicts from Stage 1
        model: Trained satellite diffusion U-Net
        scheduler: Noise scheduler
        vae: Trained satellite VAE
        train_config: Training configuration
        diffusion_model_config: Diffusion model config
        autoencoder_model_config: VAE config
        diffusion_config: Diffusion process config
        dataset_config: Dataset config
        big_data_storage_path: Data storage path
        guidance_scale: Classifier-free guidance scale
        include_masked_rgb: Include masked RGB context
        include_lst_target: Include LST target in conditioning
        overwrite_samples: Whether to overwrite existing samples
        
    Returns:
        Generated satellite images
    """
    model.eval()
    
    # Get image and latent sizes using utility function
    im_size, latent_size, vae_factor, unet_factor, total_divisor = compute_patch_and_latent_sizes(
        dataset_config,
        autoencoder_model_config,
        diffusion_model_config,
        use_latents=False
    )
    
    print("\n" + "="*50)
    print("Satellite Rendering Configuration")
    print("="*50)
    print(f"Image size: {im_size}x{im_size} ({im_size * dataset_config['res']}m)")
    print(f"Latent size: {latent_size}x{latent_size}")
    print(f"VAE downsample: {vae_factor}x, U-Net downsample: {unet_factor}x")
    print(f"Number of samples: {len(semantic_samples)}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Include masked RGB: {include_masked_rgb}")
    print(f"Include LST target: {include_lst_target}")
    
    # Get inpainting mode from config
    inpainting_cfg = train_config.get('inpainting', {})
    mode = inpainting_cfg.get('mode', 'hard')
    
    print(f"Inpainting mode: {mode}")
    
    ################# Rendering Loop ########################
    print("\n" + "="*50)
    print("Starting Satellite Rendering")
    print("="*50)
    
    all_renders = []
    
    for sample_idx, sample_data in enumerate(semantic_samples):
        print(f"\nRendering sample {sample_idx + 1}/{len(semantic_samples)}")
        
        # Extract data
        semantic_tensor = sample_data['semantic_tensor'].to(device)
        semantic_channels = sample_data['semantic_channels']
        original_conditioning = sample_data.get('conditioning', {})
        mask = sample_data.get('mask')
        
        if mask is None:
            # Create full mask if not provided
            mask = torch.ones(1, semantic_tensor.shape[1], semantic_tensor.shape[2])
        
        mask = mask.to(device)
        
        # Build conditioning for Stage 2
        cond_input = build_conditioning_from_semantics(
            semantic_tensor,
            semantic_channels,
            original_conditioning,
            mask,
            include_masked_rgb=include_masked_rgb,
            include_lst_target=include_lst_target
        )
        
        # Add batch dimension
        cond_input['image'] = cond_input['image'].unsqueeze(0)
        
        # Create unconditional input
        uncond_input = {
            'image': torch.zeros_like(cond_input['image']),
            'meta': cond_input['meta']
        }
        
        # Get context RGB for inpainting
        x_context_rgb = None
        if mode == "hard" and include_masked_rgb:
            # Extract masked RGB from conditioning
            spatial_names = cond_input['meta']['spatial_names']
            rgb_channels = []
            for idx, name in enumerate(spatial_names):
                if 'masked_image' in name:
                    rgb_channels.append(cond_input['image'][:, idx:idx+1, :, :])
            
            if rgb_channels and len(rgb_channels) == 3:
                # Encode to latent
                rgb_context = torch.cat(rgb_channels, dim=1)
                with torch.no_grad():
                    x_context_rgb = vae.encoder(rgb_context)
            else:
                x_context_rgb = torch.zeros(1, autoencoder_model_config['z_channels'], 
                                           latent_size, latent_size, device=device)
        
        # Downsample mask to latent resolution
        mask_latent = F.interpolate(
            mask.unsqueeze(0).float(),
            size=(latent_size, latent_size),
            mode='nearest'
        )
        
        # Initialize latent
        x = torch.randn(1, autoencoder_model_config['z_channels'], 
                       latent_size, latent_size, device=device)
        
        if mode == "hard" and x_context_rgb is not None:
            # Start with context in unmasked region
            x = mask_latent * x + (1 - mask_latent) * x_context_rgb
        
        # Sampling loop with inpainting
        for i in tqdm(reversed(range(scheduler.num_timesteps)), desc="Denoising"):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if guidance_scale > 0:
                noise_pred_cond = model(x, t, cond_input=cond_input)
                noise_pred_uncond = model(x, t, cond_input=uncond_input)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, t, cond_input=cond_input)
            
            # Denoise with inpainting
            if mode == "hard" and x_context_rgb is not None:
                x, x0 = scheduler.sample_prev_timestep_inpainting(
                    x, noise_pred, i, x_context_rgb, mask_latent
                )
            else:
                x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
        
        all_renders.append(x)
    
    # Stack renders
    all_renders = torch.cat(all_renders, dim=0)
    
    # Decode to RGB
    with torch.no_grad():
        rgb_renders = vae.decoder(all_renders)
    
    # Normalize to [0, 1]
    rgb_renders = torch.clamp(rgb_renders, -1., 1.)
    rgb_renders = (rgb_renders + 1) / 2
    rgb_renders = torch.clamp(rgb_renders, 0., 1.)
    
    # Save results
    # Note: task_name should be from base train_config, not satellite_train_config
    task_name = train_config.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}/satellite_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get run index
    base_name = f'satellite_cfg{guidance_scale}'
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*50}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*50}")
    
    # Save grid
    grid = make_grid(rgb_renders, nrow=int(np.sqrt(len(semantic_samples))) + 1, 
                    padding=4, pad_value=1.0)
    output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}.png')
    save_image(grid, output_path)
    print(f"\n✓ Saved visualization to {output_path}")
    
    # Save individual renders
    for idx in range(len(semantic_samples)):
        sample_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_sample_{idx}.png')
        save_image(rgb_renders[idx], sample_path)
    
    print(f"✓ Saved {len(semantic_samples)} satellite renders")
    
    return rgb_renders


def infer(args, config):
    ###### setup config variables #######
    data_config = config['data_config']
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Get satellite-specific configs
    satellite_ldm_config = ldm_config.get('satellite', ldm_config)
    satellite_autoencoder_config = autoencoder_config.get('satellite', autoencoder_config)
    satellite_train_config = train_config.get('satellite', train_config)
    
    ########## Create Scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    ########## Load Models #############
    print("\n" + "="*50)
    print("Loading Models")
    print("="*50)
    
    # Load Satellite VAE
    vae = VAE(
        im_channels=dataset_config['im_channels'],
        model_config=satellite_autoencoder_config
    ).to(device)
    vae.eval()
    
    out_dir = f"{big_data_storage_path}/results/{train_config['task_name']}"
    vae_path = os.path.join(out_dir, satellite_train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth'))
    
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print(f"✓ Loaded Satellite VAE from {vae_path}")
    else:
        print(f"✗ Satellite VAE not found at {vae_path}")
        return
    
    # Load Satellite Diffusion Model
    model = Unet(
        im_channels=satellite_autoencoder_config['z_channels'],
        model_config=satellite_ldm_config,
        mode='satellite'
    ).to(device)
    model.eval()
    
    ldm_path = os.path.join(out_dir, satellite_train_config.get('ldm_ckpt_name', 'ddpm_urban_inpainting_ckpt.pth'))
    
    if os.path.exists(ldm_path):
        model.load_state_dict(torch.load(ldm_path, map_location=device))
        print(f"✓ Loaded Satellite Diffusion Model from {ldm_path}")
    else:
        print(f"✗ Satellite Diffusion Model not found at {ldm_path}")
        return
    
    ########## Load Semantic Samples #############
    if not args.semantic_dir:
        print("✗ Please provide --semantic_dir with path to Stage 1 semantic samples")
        return
    
    semantic_samples = load_semantic_samples(args.semantic_dir)
    
    if not semantic_samples:
        return
    
    ########## Render Satellite Images #############
    renders = render_satellite_from_semantics(
        semantic_samples=semantic_samples,
        model=model,
        scheduler=scheduler,
        vae=vae,
        train_config=satellite_train_config,
        diffusion_model_config=satellite_ldm_config,
        autoencoder_model_config=satellite_autoencoder_config,
        diffusion_config=diffusion_config,
        dataset_config=dataset_config,
        big_data_storage_path=big_data_storage_path,
        guidance_scale=args.guidance_scale,
        include_masked_rgb=args.include_masked_rgb,
        include_lst_target=args.include_lst_target,
        overwrite_samples=args.overwrite_samples
    )
    
    return renders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render satellite imagery from semantic layouts')
    parser.add_argument('--semantic_dir', type=str, required=True, 
                       help='Directory containing Stage 1 semantic samples')
    parser.add_argument('--guidance_scale', type=float, default=7.5, 
                       help='Classifier-free guidance scale')
    parser.add_argument('--include_masked_rgb', action='store_true', default=True,
                       help='Include masked RGB context')
    parser.add_argument('--include_lst_target', action='store_true', default=True,
                       help='Include LST target in conditioning')
    parser.add_argument('--overwrite_samples', action='store_true', 
                       help='Overwrite existing samples')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        os.environ['CONFIG_PATH'] = args.config
    
    config = load_configs()
    
    infer(args, config)

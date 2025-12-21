# Sampling script for urban inpainting
#### import libraries ######
# Standard libraries
import sys
import os
import random
import argparse
import yaml
from pathlib import Path

# Visualization
from tqdm import tqdm

# Data handling
import numpy as np
from PIL import Image

# Data Science/ML libraries
import torch
import torchvision
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

# Local libraries
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.config_utils import get_config_value
from helpers.load_configs import load_configs
from helpers.indexed_outputs import get_next_run_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_inpainting(model, scheduler, train_config, diffusion_model_config,
                     autoencoder_model_config, diffusion_config, dataset_config,
                     big_data_storage_path, vae,
                     num_samples=4, guidance_scale=7.5,
                     overwrite_samples=False,
                     clamp_sampling=True):
    """
    Sample urban layouts using inpainting diffusion model.
    
    Args:
        model: Trained U-Net diffusion model
        scheduler: Noise scheduler
        train_config: Training configuration
        diffusion_model_config: Diffusion model config
        autoencoder_model_config: VAE config
        diffusion_config: Diffusion process config
        dataset_config: Dataset config
        vae: Trained VAE model
        num_samples: Number of samples to generate
        guidance_scale: Classifier-free guidance scale
    """ 
    
    
    model.eval()
    
    # Get latent size
    im_size = dataset_config['patch_size_m'] // dataset_config['res']
    latent_size = im_size // (2 ** sum(autoencoder_model_config['down_sample']))
    
    print("\n" + "="*50)
    print("Sampling Configuration")
    print("="*50)
    print(f"Image size: {im_size}x{im_size}")
    print(f"Latent size: {latent_size}x{latent_size}")
    print(f"Number of samples: {num_samples}")
    print(f"Guidance scale: {guidance_scale}")
    
    # Load dataset to get real conditioning examples
    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    task_name = train_config['task_name']
    cache_dir = Path(big_data_storage_path) / "processed" / task_name
    dataset = UrbanInpaintingDataset(
        split='val',
        use_latents=False,
        latent_path=None,
        use_cached_patches=True,
        cache_dir=cache_dir
    )
    
    # set seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get a random sample for conditioning
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"\nUsing sample index {sample_idx} for conditioning")
    
    sample_data = dataset[sample_idx]
    if len(sample_data) == 2:
        _, cond_input = sample_data
    else:
        raise ValueError("Dataset must return conditioning information")
    
    # Prepare conditioning inputs
    # Add batch dimension and move to device
    for key in cond_input:
        if key == 'meta':
            continue
        if isinstance(cond_input[key], torch.Tensor):
            cond_input[key] = cond_input[key].unsqueeze(0).to(device)

    # Extract mask and prepare context for inpainting
    if 'image' in cond_input and 'meta' in cond_input:
        spatial_names = cond_input['meta']['spatial_names']
        try:
            mask_idx = spatial_names.index('inpaint_mask')
            mask_full = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
            print(f"Mask extracted from spatial tensor at channel {mask_idx}")
            print(f"Mask shape: {mask_full.shape}")
        except ValueError:
            raise ValueError(f"Mask not found. Available channels: {spatial_names}")
        
        # Encode spatial context (image conditioning)
        # and extract the masked RGB from spatial conditioning for context
        spatial_cond = cond_input['image']  # [1, C, H, W]
        print(f"\n{'='*50}")
        print("Channel Analysis")
        print(f"{'='*50}")
        print(f"Spatial conditioning shape: {spatial_cond.shape}")
        print(f"Available conditioning channels: {spatial_names}")
        
        # Find RGB channel indices explicitly
        rgb_channel_names = ['masked_image:blue', 'masked_image:green', 'masked_image:red']
        rgb_indices = []
        for channel_name in rgb_channel_names:
            try:
                idx = spatial_names.index(channel_name)
                rgb_indices.append(idx)
                print(f"✓ Found {channel_name} at index {idx}")
            except ValueError:
                print(f"⚠ Channel {channel_name} not found!")
        
        # Extract masked RGB channels
        if len(rgb_indices) == 3:
            # Extract RGB in BGR order (blue, green, red)
            masked_rgb_bgr = spatial_cond[:, rgb_indices, :, :]
            # Convert BGR to RGB
            masked_rgb = torch.stack([
                masked_rgb_bgr[:, 2, :, :],  # Red
                masked_rgb_bgr[:, 1, :, :],  # Green
                masked_rgb_bgr[:, 0, :, :]   # Blue
            ], dim=1)
            print(f"✓ Extracted RGB (converted from BGR) with shape: {masked_rgb.shape}")
        else:
            # Fallback: use first 3 channels
            masked_rgb = spatial_cond[:, :3, :, :]
            print(f"⚠ Using fallback: first 3 channels as RGB")
            
        print(f"\n{'='*50}")
        print("Value Range Analysis (Before VAE Encoding)")
        print(f"{'='*50}")
        print(f"Masked RGB stats:")
        print(f"  min: {masked_rgb.min().item():.4f}")
        print(f"  max: {masked_rgb.max().item():.4f}")
        print(f"  mean: {masked_rgb.mean().item():.4f}")
        print(f"  std: {masked_rgb.std().item():.4f}")
        
        # Check if data is in [0, 1] or [-1, 1]
        if masked_rgb.min() >= -0.1 and masked_rgb.max() <= 1.1:
            print("✓ Data appears to be in [0, 1] range")
            # Convert to [-1, 1] for VAE
            masked_rgb_normalized = masked_rgb * 2.0 - 1.0
            print("✓ Converted to [-1, 1] for VAE")
        elif masked_rgb.min() >= -1.1 and masked_rgb.max() <= 1.1:
            print("✓ Data already in [-1, 1] range")
            masked_rgb_normalized = masked_rgb
        else:
            print(f"⚠ WARNING: Data range unexpected!")
            print(f"  Clamping to [0, 1] then converting to [-1, 1]")
            masked_rgb_normalized = torch.clamp(masked_rgb, 0., 1.) * 2.0 - 1.0
        
        # Encode to latent space
        with torch.no_grad():
            x_context, _, _ = vae.encode(masked_rgb_normalized)
            print(f"\n{'='*50}")
            print("Latent Space Analysis")
            print(f"{'='*50}")
            print(f"Context latent shape: {x_context.shape}")
            print(f"Context latent stats:")
            print(f"  min: {x_context.min().item():.4f}")
            print(f"  max: {x_context.max().item():.4f}")
            print(f"  mean: {x_context.mean().item():.4f}")
            print(f"  std: {x_context.std().item():.4f}")
            
            # Get actual latent dimensions from VAE output
            actual_latent_h = x_context.shape[2]
            actual_latent_w = x_context.shape[3]
            print(f"Actual latent size from VAE: {actual_latent_h}x{actual_latent_w}")
            
            # Test decode to verify VAE reconstruction
            test_decoded = vae.decode(x_context)
            if clamp_sampling:
                test_decoded = torch.clamp(test_decoded, -1., 1.)
            print(f"\nTest VAE reconstruction stats:")
            print(f"  min: {test_decoded.min().item():.4f}")
            print(f"  max: {test_decoded.max().item():.4f}")
            print(f"  mean: {test_decoded.mean().item():.4f}")
    
        # Downsample mask to latent resolution
        mask_latent = F.interpolate(
            mask_full,
            size=(actual_latent_h, actual_latent_w),
            mode='nearest'
        )
        print(f"Mask latent shape: {mask_latent.shape}")
    else:
        raise ValueError("Conditioning input must contain 'image' and 'meta' keys")
    
    # Create unconditional input for classifier-free guidance
    uncond_input = {}
    for key in cond_input:
        if key == 'meta':
            uncond_input[key] = cond_input[key]
        elif isinstance(cond_input[key], torch.Tensor):
            uncond_input[key] = torch.zeros_like(cond_input[key])
    
    ################# Sampling Loop ########################
    print("\n" + "="*50)
    print("Starting Sampling")
    print("="*50)
    
    all_samples = []
    
    for sample_idx in range(num_samples):
        print(f"\nGenerating sample {sample_idx + 1}/{num_samples}")
        
        # Sample random noise latent
        xt = torch.randn(
            1,
            autoencoder_model_config['z_channels'],
            actual_latent_h,
            actual_latent_w
        ).to(device)
        
        # xt is the noisy latent at the current timestep
        # this replaces the known (unmasked) regions with the encoded context and keeps
        # the masked regions as noise
        xt = mask_latent * xt + (1 - mask_latent) * x_context
        
        for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), 
                     desc=f"Sampling"):
            t = torch.tensor([i]).long().to(device)
            
            with torch.no_grad():
                # Conditional prediction
                noise_pred_cond = model(xt, t, cond_input=cond_input)
                
                # Classifier-free guidance
                if guidance_scale > 1:
                    noise_pred_uncond = model(xt, t, cond_input=uncond_input)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond
                
                # Sample previous timestep with inpainting constraint
                xt, x0_pred = scheduler.sample_prev_timestep_inpainting(
                    xt, noise_pred, i, x_context, mask_latent
                )
        
        # Decode final latent
        with torch.no_grad():
            generated_img = vae.decode(xt)
            if clamp_sampling:
                generated_img = torch.clamp(generated_img, -1., 1.)
        
        all_samples.append(generated_img)
    
    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)  # [num_samples, C, H, W]
    
    # Also include the original masked image for comparison
    with torch.no_grad():
        original_decoded = vae.decode(x_context)
        if clamp_sampling:
            original_decoded = torch.clamp(original_decoded, -1., 1.)
    
    # Normalize to [0, 1]
    all_samples = torch.clamp(all_samples, -1., 1.) 
    all_samples = (all_samples + 1) / 2
    original_decoded = torch.clamp(original_decoded, -1., 1.)
    original_decoded = (original_decoded + 1) / 2
    
    # additional clamp for safety
    all_samples = torch.clamp(all_samples, 0., 1.)
    original_decoded = torch.clamp(original_decoded, 0., 1.)
    
    # Create grid with original first
    grid_images = torch.cat([original_decoded, all_samples], dim=0)
    grid = make_grid(grid_images, nrow=int(np.sqrt(num_samples + 1)) + 1, padding=4, pad_value=1.0)
    
    # Save results
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}/output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get next run index
    base_name = f'samples_guidance{guidance_scale}'
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx -= 1
        
    print(f"\n{'='*50}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*50}")
    
    # Save grid with run index
    output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}.png')
    save_image(grid, output_path)
    print(f"\n✓ Saved samples to {output_path}")
    
    # Save mask for visualization purposes
    mask_save_path = os.path.join(out_dir, f'mask_idx{run_idx}.npy')
    np.save(mask_save_path, mask_full.cpu().numpy().squeeze())
    print(f"✓ Saved mask to {mask_save_path}")
    
    # Also save individual samples with run index
    for idx, sample in enumerate(all_samples):
        individual_path = os.path.join(out_dir, f'sample_{idx}_idx{run_idx}.png')
        save_image(sample, individual_path)
    
    print(f"✓ Saved {num_samples} individual samples")
    
    return all_samples


def infer(args):
    ###### setup config variables #######
    config = load_configs()
    # repo_dir = config['repo_dir']
    data_config = config['data_config']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    print("="*50)
    print("Urban Inpainting Sampling")
    print("="*50)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    model_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    print("✓ Created noise scheduler")
    
    ########## Load Models #############
    print("\nLoading models...")
    
    # Load U-Net
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    
    model_path = os.path.join(
        model_dir,
        train_config.get('ldm_ckpt_name', 'ddpm_urban_inpainting_ckpt.pth')
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded U-Net from {model_path}")
    
    # Load VAE
    vae = VAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config
    ).to(device)
    
    vae_path = os.path.join(
        model_dir,
        train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth')
    )
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print(f"✓ Loaded VAE from {vae_path}")
    
    ########## Sample #############
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else train_config.get('cf_guidance_scale', 7.5)
    num_samples = args.num_samples
    overwrite_samples = args.overwrite_samples if args.overwrite_samples is not None else train_config.get('overwrite_samples', False)
    clamp_sampling = args.clamp_sampling if args.clamp_sampling is not None else train_config.get('clamp_sampling', True)
    
    with torch.no_grad():
        samples = sample_inpainting(
            model, scheduler, train_config, diffusion_model_config,
            autoencoder_model_config, diffusion_config, dataset_config, big_data_storage_path, vae,
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            overwrite_samples=overwrite_samples,
            clamp_sampling=clamp_sampling
        )
    
    print("\n" + "="*50)
    print("✓ Sampling Complete!")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from urban inpainting model')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=4,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=7.5,
        help='Classifier-free guidance scale'
    )
    parser.add_argument(
        '--overwrite_samples',
        action='store_true',
        help='Whether to overwrite existing samples',
        default=False,
        help='Whether to overwrite existing sample paths'
    )
    parser.add_argument(
        '--clamp_sampling',
        action='store_true',
        help='Whether to clamp sampling outputs to [-1, 1]',
        default=True,
        help='Whether to clamp sampling outputs to [-1, 1]'
    )
    args = parser.parse_args()
    infer(args)

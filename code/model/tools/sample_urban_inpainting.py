# Sampling script for urban inpainting
#### import libraries ######
# Standard libraries
import sys
import os
import random
import argparse
import yaml

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
from diffusion_blocks.unet_cond_base import Unet
from diffusion_blocks.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from utils.config_utils import get_config_value
from helpers.load_configs import load_configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_inpainting(model, scheduler, train_config, diffusion_model_config,
                     autoencoder_model_config, diffusion_config, dataset_config,
                     big_data_storage_path, vae,
                     num_samples=4, guidance_scale=7.5):
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
    dataset = UrbanInpaintingDataset(
        split='val',
        use_latents=False,
        latent_path=None
    )
    
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
    
    # Get mask and context latent for inpainting
    if 'mask' in cond_input:
        mask_full = cond_input['mask']  # [1, 1, H, W] in pixel space
        print(f"Mask shape: {mask_full.shape}")
    else:
        raise ValueError("Mask not found in conditioning input")
    
    # Encode spatial context (the image conditioning)
    if 'image' in cond_input:
        spatial_cond = cond_input['image']  # [1, C, H, W]
        print(f"Spatial conditioning shape: {spatial_cond.shape}")
        print(f"Spatial conditioning channels: {cond_input['meta']['spatial_names']}")
    
    # Extract the masked RGB from spatial conditioning for context
    # Assume first 3 channels are masked RGB
    if 'image' in cond_input:
        masked_rgb = spatial_cond[:, :3, :, :]  # First 3 channels
        print(f"Masked RGB shape: {masked_rgb.shape}")
        
        # Encode to latent space
        with torch.no_grad():
            x_context, _ = vae.encode(masked_rgb)
            print(f"Context latent shape: {x_context.shape}")
    
    # Downsample mask to latent resolution
    mask_latent = F.interpolate(
        mask_full,
        size=(latent_size, latent_size),
        mode='nearest'
    )
    print(f"Mask latent shape: {mask_latent.shape}")
    
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
            latent_size,
            latent_size
        ).to(device)
        
        # Replace known region with noisy context
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
        
        all_samples.append(generated_img)
    
    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)  # [num_samples, C, H, W]
    
    # Also include the original masked image for comparison
    with torch.no_grad():
        original_decoded = vae.decode(x_context)
    
    # Normalize to [0, 1]
    all_samples = torch.clamp(all_samples, -1., 1.)
    all_samples = (all_samples + 1) / 2
    original_decoded = torch.clamp(original_decoded, -1., 1.)
    original_decoded = (original_decoded + 1) / 2
    
    # Create grid with original first
    grid_images = torch.cat([original_decoded, all_samples], dim=0)
    grid = make_grid(grid_images, nrow=int(np.sqrt(num_samples + 1)) + 1, padding=4, pad_value=1.0)
    
    # Save results
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}/inpainting_samples"
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'samples_guidance{guidance_scale}.png')
    save_image(grid, output_path)
    
    print(f"\n✓ Saved samples to {output_path}")
    
    # Also save individual samples
    for idx, sample in enumerate(all_samples):
        individual_path = os.path.join(out_dir, f'sample_{idx}.png')
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
        train_config['task_name'],
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
        train_config['task_name'],
        train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth')
    )
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print(f"✓ Loaded VAE from {vae_path}")
    
    ########## Sample #############
    guidance_scale = train_config.get('cf_guidance_scale', 7.5)
    num_samples = args.num_samples
    
    with torch.no_grad():
        samples = sample_inpainting(
            model, scheduler, train_config, diffusion_model_config,
            autoencoder_model_config, diffusion_config, dataset_config, big_data_storage_path, vae,
            num_samples=num_samples,
            guidance_scale=guidance_scale
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
    args = parser.parse_args()
    infer(args)

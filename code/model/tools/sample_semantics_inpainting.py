# Sampling script for semantic inpainting with LST guidance
# Stage 1: Generate semantic layouts (buildings/roads/vegetation/height) with temperature control

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Data handling
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# Local libraries
from model.diffusion_blocks.unet_cond_base import Unet
from model.diffusion_blocks.vae import VAE
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.config_utils import get_prediction_channels, compute_patch_and_latent_sizes
from helpers.load_configs import load_configs
from helpers.indexed_outputs import get_next_run_idx
from model.lst_predictor.predictor import LSTPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_lst_predictor(checkpoint_path, device):
    """
    Load LST predictor model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        LSTPredictor model
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠ LST predictor checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model = LSTPredictor(
        in_channels=config.get('in_channels', 4),
        hidden_dims=config.get('hidden_dims', [64, 128, 256]),
        out_channels=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded LST predictor from {checkpoint_path}")
    return model


def apply_lst_guidance(
    x, 
    t, 
    model, 
    scheduler, 
    cond_input, 
    lst_predictor, 
    vae, 
    lst_target,
    semantic_channels,
    include_ndvi,
    guidance_scale=1.0,
    mask=None
):
    """
    Apply LST predictor guidance to steer semantic generation toward temperature target.
    
    Uses classifier guidance approach: modify noise prediction based on gradient of LST predictor.
    
    Args:
        x: Current latent [B, C, H, W]
        t: Current timestep
        model: Diffusion model
        scheduler: Noise scheduler
        cond_input: Conditioning input
        lst_predictor: LST predictor model
        vae: Semantic VAE
        lst_target: Target LST raster [B, 1, H, W]
        semantic_channels: List of semantic channel names
        include_ndvi: Whether NDVI is included
        guidance_scale: Strength of guidance
        mask: Optional mask to apply guidance only in masked region
        
    Returns:
        Guided noise prediction
    """
    # Get base noise prediction
    with torch.no_grad():
        noise_pred_base = model(x, t, cond_input=cond_input)
    
    if lst_predictor is None or guidance_scale == 0.0:
        return noise_pred_base
    
    # Enable gradients for guidance
    x_guide = x.detach().requires_grad_(True)
    
    # Predict noise with gradients
    noise_pred = model(x_guide, t, cond_input=cond_input)
    
    # Predict x0 from current noisy x
    # x0 = (x - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
    alpha_t = scheduler.alpha_cum_prod[t].to(x.device)
    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)
    
    x0_pred = (x_guide - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    x0_pred = torch.clamp(x0_pred, -3, 3)  # Clamp for stability
    
    # Decode to semantic space
    with torch.no_grad():
        semantic_pred = vae.decoder(x0_pred)
    
    # Build input for LST predictor
    semantic_tensor = []
    for ch_idx, ch_name in enumerate(semantic_channels):
        if ch_idx < semantic_pred.shape[1]:
            semantic_tensor.append(semantic_pred[:, ch_idx:ch_idx+1, :, :])
    
    semantic_input = torch.cat(semantic_tensor, dim=1)
    
    # Add NDVI if needed
    if include_ndvi:
        # Extract NDVI from conditioning or use zeros
        ndvi_channel = None
        if 'image' in cond_input and 'meta' in cond_input:
            spatial_names = cond_input['meta'].get('spatial_names', [])
            for idx, name in enumerate(spatial_names):
                if 'ndvi' in name.lower():
                    ndvi_channel = cond_input['image'][:, idx:idx+1, :, :]
                    break
        
        if ndvi_channel is None:
            ndvi_channel = torch.zeros(semantic_input.shape[0], 1, 
                                      semantic_input.shape[2], semantic_input.shape[3],
                                      device=semantic_input.device)
        
        semantic_input = torch.cat([semantic_input, ndvi_channel], dim=1)
    
    # Predict LST
    lst_pred = lst_predictor(semantic_input)
    
    # Resize to match lst_target if needed
    if lst_pred.shape[-2:] != lst_target.shape[-2:]:
        lst_pred = F.interpolate(lst_pred, size=lst_target.shape[-2:], mode='bilinear', align_corners=False)
    
    # Compute LST loss (MSE)
    if mask is not None:
        # Apply loss only inside mask
        mask_resized = F.interpolate(mask.float(), size=lst_pred.shape[-2:], mode='nearest')
        lst_loss = ((lst_pred - lst_target) ** 2 * mask_resized).mean()
    else:
        lst_loss = F.mse_loss(lst_pred, lst_target)
    
    # Compute gradient
    grad = torch.autograd.grad(lst_loss, x_guide)[0]
    
    # Apply guidance: noise_pred = noise_pred_base - guidance_scale * grad
    noise_pred_guided = noise_pred_base - guidance_scale * grad.detach()
    
    return noise_pred_guided


def sample_semantics(
    model, 
    scheduler, 
    train_config, 
    diffusion_model_config,
    autoencoder_model_config, 
    diffusion_config, 
    dataset_config,
    semantic_config,
    big_data_storage_path, 
    vae,
    lst_predictor=None,
    num_samples=4, 
    guidance_scale=7.5,
    lst_guidance_scale=1.0,
    use_lst_guidance=False,
    overwrite_samples=False
):
    """
    Sample semantic layouts using inpainting diffusion model with optional LST guidance.
    
    Args:
        model: Trained semantic diffusion U-Net
        scheduler: Noise scheduler
        train_config: Training configuration
        diffusion_model_config: Diffusion model config
        autoencoder_model_config: VAE config
        diffusion_config: Diffusion process config
        dataset_config: Dataset config
        semantic_config: Semantic configuration
        big_data_storage_path: Data storage path
        vae: Trained semantic VAE
        lst_predictor: Optional LST predictor for guidance
        num_samples: Number of samples to generate
        guidance_scale: Classifier-free guidance scale
        lst_guidance_scale: LST guidance scale
        use_lst_guidance: Whether to use LST guidance
        overwrite_samples: Whether to overwrite existing samples
        
    Returns:
        Generated semantic samples
    """
    model.eval()
    
    # Get semantic channels from condition config
    condition_config = diffusion_model_config.get('condition_config', {})
    semantic_channels = get_prediction_channels(condition_config)
    
    if not semantic_channels:
        # Fallback to default
        semantic_channels = [
            'osm:buildings',
            'osm:streets',
            'env:vegetation',
            'osm:buildings_heights'
        ]
    
    include_ndvi = train_config.get('lst_predictor_use_ndvi', True)
    
    # Get image and latent sizes using utility function
    im_size, latent_size, vae_factor, unet_factor, total_divisor = compute_patch_and_latent_sizes(
        dataset_config,
        autoencoder_model_config,
        diffusion_model_config,
        use_latents=True
    )
    
    print("\n" + "="*50)
    print("Semantic Sampling Configuration")
    print("="*50)
    print(f"Image size: {im_size}x{im_size} ({im_size * dataset_config['res']}m)")
    print(f"Latent size: {latent_size}x{latent_size}")
    print(f"VAE downsample: {vae_factor}x, U-Net downsample: {unet_factor}x")
    print(f"Number of samples: {num_samples}")
    print(f"Guidance scale (CFG): {guidance_scale}")
    print(f"LST guidance scale: {lst_guidance_scale}")
    print(f"Use LST guidance: {use_lst_guidance}")
    print(f"Semantic channels: {semantic_channels}")
    
    # Load dataset to get conditioning examples
    # Note: task_name should be from base train_config, not semantic_train_config
    task_name = train_config.get('task_name', 'urban_inpainting')
    cache_dir = Path(big_data_storage_path) / "processed" / task_name / "semantic"
    use_cached_patches = cache_dir.exists()
    
    # Check if latents exist for val split
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    latent_path = os.path.join(out_dir, "semantic_vae_latents_val.pt")
    use_latents = os.path.exists(latent_path)
    
    if use_latents:
        print(f"\n✓ Found validation latents at {latent_path}")
    else:
        print(f"\n⚠ Validation latents not found, will downsample conditioning on-the-fly")
    
    dataset = UrbanInpaintingDataset(
        split='val',
        mode='semantic',
        use_latents=use_latents,
        latent_path=latent_path if use_latents else None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    # Set seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get a random sample for conditioning
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"\nUsing sample index {sample_idx} for conditioning")
    
    sample_data = dataset[sample_idx]
    if len(sample_data) == 2:
        im, cond_input = sample_data
    else:
        cond_input = {}
    
    # Prepare conditioning inputs
    for key in cond_input:
        if isinstance(cond_input[key], torch.Tensor):
            cond_input[key] = cond_input[key].unsqueeze(0).to(device)
        elif key == 'meta' and isinstance(cond_input[key], dict):
            for k, v in cond_input[key].items():
                if isinstance(v, torch.Tensor):
                    cond_input[key][k] = v.unsqueeze(0).to(device)
    
    # Downsample conditioning to latent resolution if not using pre-computed latents
    if not use_latents and 'image' in cond_input:
        print(f"\n⚠ Downsampling conditioning from full resolution to latent resolution")
        cond_spatial = cond_input['image']  # [1, C, H, W]
        
        # Compute latent dimensions
        latent_h = latent_size
        latent_w = latent_size
        
        # Separate mask (nearest) from features (bilinear)
        downsampled_channels = []
        spatial_names = cond_input.get('meta', {}).get('spatial_names', [])
        
        for idx in range(cond_spatial.shape[1]):
            channel = cond_spatial[:, idx:idx+1, :, :]  # [1, 1, H, W]
            channel_name = spatial_names[idx] if idx < len(spatial_names) else ''
            
            # Use nearest for masks, bilinear for features
            mode = 'nearest' if 'mask' in channel_name.lower() else 'bilinear'
            downsampled = F.interpolate(
                channel,
                size=(latent_h, latent_w),
                mode=mode,
                align_corners=False if mode == 'bilinear' else None
            )
            
            downsampled_channels.append(downsampled)
        
        cond_input['image'] = torch.cat(downsampled_channels, dim=1)  # [1, C, H_latent, W_latent]
        print(f"✓ Downsampled conditioning: {cond_spatial.shape} → {cond_input['image'].shape}")
    
    # Extract mask and LST target
    mask_full = None
    lst_target = None
    
    if 'meta' in cond_input and 'inpainting_mask' in cond_input['meta']:
        mask_full = cond_input['meta']['inpainting_mask']
    
    # Extract LST target
    if 'image' in cond_input and 'meta' in cond_input:
        spatial_names = cond_input['meta'].get('spatial_names', [])
        for idx, name in enumerate(spatial_names):
            if 'LST' in name or 'lst' in name:
                lst_target = cond_input['image'][:, idx:idx+1, :, :]
                print(f"✓ Found LST target channel: {name}")
                break
    
    if use_lst_guidance and lst_target is None:
        print("⚠ LST guidance requested but no LST target found in data")
        use_lst_guidance = False
    
    # Create unconditional input for CFG
    uncond_input = {}
    for key in cond_input:
        if key == 'image':
            uncond_input[key] = torch.zeros_like(cond_input[key])
        elif key == 'meta':
            uncond_input[key] = cond_input[key].copy() if isinstance(cond_input[key], dict) else cond_input[key]
    
    # Get inpainting mode from semantic config
    inpainting_cfg = train_config.get('inpainting', {})
    mode = inpainting_cfg.get('mode', 'hard')
    
    print(f"\n✓ Inpainting mode: {mode}")
    
    ################# Sampling Loop ########################
    print("\n" + "="*50)
    print("Starting Sampling")
    print("="*50)
    
    all_samples = []
    
    for sample_idx in range(num_samples):
        print(f"\nGenerating sample {sample_idx + 1}/{num_samples}")
        
        # Start from random noise
        x = torch.randn(1, autoencoder_model_config['z_channels'], 
                       latent_size, latent_size).to(device)
        
        # For hard inpainting, start with context latent
        if mode == "hard":
            # Encode context to get base latent
            with torch.no_grad():
                # Build semantic tensor from conditioning
                semantic_tensor = []
                if 'image' in cond_input and 'meta' in cond_input:
                    spatial_names = cond_input['meta'].get('spatial_names', [])
                    
                    for sem_ch in semantic_channels:
                        found = False
                        for idx, name in enumerate(spatial_names):
                            if sem_ch == name or (sem_ch in name and '_context' not in name):
                                semantic_tensor.append(cond_input['image'][:, idx:idx+1, :, :])
                                found = True
                                break
                        
                        if not found:
                            _, _, H, W = cond_input['image'].shape
                            semantic_tensor.append(torch.zeros(1, 1, H, W, device=device))
                    
                    semantic_input = torch.cat(semantic_tensor, dim=1)
                    x_context, _, _ = vae.encode(semantic_input)
                    
                    # Ensure x_context matches x dimensions
                    if x_context.shape[-2:] != (latent_size, latent_size):
                        x_context = F.interpolate(
                            x_context,
                            size=(latent_size, latent_size),
                            mode='bilinear',
                            align_corners=False
                        )
                else:
                    x_context = torch.zeros_like(x)
                
                # Downsample mask to latent resolution
                if mask_full is not None:
                    mask_latent = F.interpolate(
                        mask_full.float(),
                        size=(latent_size, latent_size),
                        mode='nearest'
                    )
                else:
                    mask_latent = torch.ones(1, 1, latent_size, latent_size, device=device)
                
                # Initialize: keep context, randomize masked region
                x = mask_latent * x + (1 - mask_latent) * x_context
        else:
            # SD-like: start from pure noise
            mask_latent = None
            x_context = None
        
        # Sampling loop
        for i in tqdm(reversed(range(scheduler.num_timesteps)), desc="Denoising"):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if guidance_scale > 0:
                # Conditional prediction
                noise_pred_cond = model(x, t, cond_input=cond_input)
                
                # Unconditional prediction
                noise_pred_uncond = model(x, t, cond_input=uncond_input)
                
                # CFG
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, t, cond_input=cond_input)
            
            # Apply LST guidance
            if use_lst_guidance and lst_predictor is not None and lst_target is not None:
                noise_pred = apply_lst_guidance(
                    x, i, model, scheduler, cond_input,
                    lst_predictor, vae, lst_target,
                    semantic_channels, include_ndvi,
                    guidance_scale=lst_guidance_scale,
                    mask=mask_latent
                )
            
            # Denoise
            if mode == "hard" and mask_latent is not None:
                x, x0 = scheduler.sample_prev_timestep_inpainting(
                    x, noise_pred, i, x_context, mask_latent
                )
            else:
                x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
        
        all_samples.append(x)
    
    # Stack samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Decode to semantic space
    with torch.no_grad():
        semantic_samples = vae.decoder(all_samples)
    
    # Clamp semantic values
    semantic_samples = torch.clamp(semantic_samples, 0, 1)
    
    # Save results
    out_dir = f"{big_data_storage_path}/results/{task_name}/semantic_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get next run index
    base_name = f'semantics_cfg{guidance_scale}'
    if use_lst_guidance:
        base_name += f'_lst{lst_guidance_scale}'
    
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*50}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*50}")
    
    # Save visualization
    vis_samples = []
    for ch_idx, ch_name in enumerate(semantic_channels):
        if ch_idx < semantic_samples.shape[1]:
            ch = semantic_samples[:, ch_idx:ch_idx+1, :, :]
            if 'height' in ch_name:
                ch = torch.clamp(ch / 100.0, 0, 1)  # Normalize height
            vis_samples.append(ch)
    
    if vis_samples:
        vis_tensor = torch.cat(vis_samples, dim=1)
        grid = make_grid(vis_tensor, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
        output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}.png')
        save_image(grid, output_path)
        print(f"\n✓ Saved visualization to {output_path}")
    
    # Save individual samples as .pt files for Stage 2
    samples_dir = os.path.join(out_dir, f'{base_name}_idx{run_idx}_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    for idx in range(num_samples):
        sample_path = os.path.join(samples_dir, f'sample_{idx}.pt')
        torch.save({
            'semantic_tensor': semantic_samples[idx].cpu(),
            'semantic_channels': semantic_channels,
            'conditioning': {k: v[0].cpu() if isinstance(v, torch.Tensor) else v for k, v in cond_input.items()},
            'mask': mask_full[0].cpu() if mask_full is not None else None
        }, sample_path)
    
    print(f"✓ Saved {num_samples} samples to {samples_dir}")
    
    return semantic_samples


def infer(args, config):
    ###### setup config variables #######
    data_config = config['data_config']
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    ldm_config = config['ldm_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Get semantic-specific configs
    semantic_ldm_config = ldm_config.get('semantic', ldm_config)
    semantic_autoencoder_config = autoencoder_config.get('semantic', autoencoder_config)
    semantic_train_config = train_config.get('semantic', train_config)
    
    # Get semantic channels from condition config
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
    
    # Load Semantic VAE
    vae = VAE(
        im_channels=num_semantic_channels,
        model_config=semantic_autoencoder_config
    ).to(device)
    vae.eval()
    
    out_dir = f"{big_data_storage_path}/results/{train_config['task_name']}"
    vae_path = os.path.join(out_dir, semantic_train_config.get('autoencoder_ckpt_name', 'semantic_vae_ddp_ckpt.pth'))
    
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print(f"✓ Loaded Semantic VAE from {vae_path}")
    else:
        print(f"✗ Semantic VAE not found at {vae_path}")
        return
    
    # Load Semantic Diffusion Model
    model = Unet(
        im_channels=semantic_autoencoder_config['z_channels'],
        model_config=semantic_ldm_config,
        mode='semantic'
    ).to(device)
    model.eval()
    
    ldm_path = os.path.join(out_dir, semantic_train_config.get('ldm_ckpt_name', 'semantic_ldm_ddp_ckpt.pth'))
    
    if os.path.exists(ldm_path):
        model.load_state_dict(torch.load(ldm_path, map_location=device))
        print(f"✓ Loaded Semantic Diffusion Model from {ldm_path}")
    else:
        print(f"✗ Semantic Diffusion Model not found at {ldm_path}")
        return
    
    # Load LST Predictor (optional)
    lst_predictor = None
    if args.use_lst_guidance:
        # LST predictor path from base train_config
        base_train_config = config['train_params']
        lst_predictor_path = os.path.join(out_dir, base_train_config.get('lst_predictor_ckpt_name', 'lst_predictor_best.pth'))
        lst_predictor = load_lst_predictor(lst_predictor_path, device)
    
    ########## Sample Semantics #############
    samples = sample_semantics(
        model=model,
        scheduler=scheduler,
        train_config=semantic_train_config,
        diffusion_model_config=semantic_ldm_config,
        autoencoder_model_config=semantic_autoencoder_config,
        diffusion_config=diffusion_config,
        dataset_config=dataset_config,
        semantic_config=semantic_train_config,
        big_data_storage_path=big_data_storage_path,
        vae=vae,
        lst_predictor=lst_predictor,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        lst_guidance_scale=args.lst_guidance_scale,
        use_lst_guidance=args.use_lst_guidance,
        overwrite_samples=args.overwrite_samples
    )
    
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample semantic layouts with LST guidance')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--lst_guidance_scale', type=float, default=1.0, help='LST guidance scale')
    parser.add_argument('--use_lst_guidance', action='store_true', help='Use LST predictor guidance')
    parser.add_argument('--overwrite_samples', action='store_true', help='Overwrite existing samples')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        # Set config path for load_configs
        os.environ['CONFIG_PATH'] = args.config
    
    config = load_configs()
    
    infer(args, config)

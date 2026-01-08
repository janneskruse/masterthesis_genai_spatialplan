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


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


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
        semantic_pred = vae.decode(x0_pred)
    
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
    repo_dir, 
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
    task_name = train_config.get('task_name', 'urban_inpainting')
    cache_dir = Path(big_data_storage_path) / "processed" / task_name / "semantic"
    use_cached_patches = cache_dir.exists()
    
    # Check if latents exist for val split
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    latent_dir_name = train_config.get('latent_dir_name', 'semantic_vae_latents')
    latent_path = os.path.join(data_dir, latent_dir_name + "_val")
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
    seed = train_config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n✓ Set random seed: {seed}")
    
    # Get a random sample for conditioning
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"Using sample index {sample_idx} for conditioning")
    
    sample_data = dataset[sample_idx]
    if len(sample_data) == 2:
        im, cond_input = sample_data
    else:
        cond_input = {}
    
    # Save full-resolution conditioning BEFORE any processing
    # This is needed to build semantic ground truth for VAE encoding
    cond_input_fullres = {}
    for key in cond_input:
        if isinstance(cond_input[key], torch.Tensor):
            cond_input_fullres[key] = cond_input[key].clone()
        elif key == 'meta' and isinstance(cond_input[key], dict):
            cond_input_fullres[key] = {}
            for k, v in cond_input[key].items():
                if isinstance(v, torch.Tensor):
                    cond_input_fullres[key][k] = v.clone()
                else:
                    cond_input_fullres[key][k] = v
        else:
            cond_input_fullres[key] = cond_input[key]
    
    # Prepare conditioning inputs for batch
    for key in cond_input:
        # unsqueeze batch dimension
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
    
    # Extract mask and LST target from conditioning channels
    mask_full = None
    mask_fullres = None
    lst_target = None
    
    # Diagnostic: Print conditioning channels
    if 'image' in cond_input and 'meta' in cond_input:
        spatial_names = cond_input['meta'].get('spatial_names', [])
        print(f"\n✓ Conditioning channels available ({len(spatial_names)}):")
        for idx, name in enumerate(spatial_names):
            ch = cond_input['image'][0, idx:idx+1, :, :]
            print(f"  {idx:02d} {name:40s} shape={tuple(ch.shape)} mean={ch.mean():.4f}")
        
        # Extract inpainting mask (channel name is 'inpaint_mask')
        for idx, name in enumerate(spatial_names):
            if name == 'inpaint_mask' or 'inpaint_mask' in name:
                mask_full = cond_input['image'][:, idx:idx+1, :, :]
                print(f"\n✓ Found inpainting mask channel: {name}")
                print(f"  Mask coverage: {mask_full.mean():.2%} (1=inpaint, 0=keep)")
                break
        
        # Extract LST target
        for idx, name in enumerate(spatial_names):
            if 'LST' in name or 'lst' in name:
                lst_target = cond_input['image'][:, idx:idx+1, :, :]
                print(f"✓ Found LST target channel: {name}")
                break
    
    # Also extract full-resolution mask for visualization
    if 'image' in cond_input_fullres and 'meta' in cond_input_fullres:
        spatial_names_fullres = cond_input_fullres['meta'].get('spatial_names', [])
        for idx, name in enumerate(spatial_names_fullres):
            if name == 'inpaint_mask' or 'inpaint_mask' in name:
                mask_fullres = cond_input_fullres['image'][idx:idx+1, :, :]  # [1, H, W]
                break
    
    if use_lst_guidance and lst_target is None:
        print("⚠ LST guidance requested but no LST target found in data")
        use_lst_guidance = False
    
    # Create unconditional input for CFG
    # Zero out _context channels but keep mask and other structural info
    uncond_input = {}
    for key in cond_input:
        if key == 'image':
            # Zero out context channels for CFG, but keep mask channel
            uncond_image = cond_input[key].clone()
            if 'meta' in cond_input:
                spatial_names = cond_input['meta'].get('spatial_names', [])
                for idx, name in enumerate(spatial_names):
                    # Zero out _context channels (OSM and env features)
                    if '_context' in name:
                        uncond_image[:, idx:idx+1, :, :] = 0.0
            uncond_input[key] = uncond_image
        elif key == 'meta' and isinstance(cond_input[key], dict):
            # Deep copy meta dict and ensure all tensors are at latent resolution
            uncond_input[key] = {}
            for k, v in cond_input[key].items():
                if isinstance(v, torch.Tensor):
                    # Ensure all meta tensors match latent resolution
                    if v.shape[-2:] != (latent_size, latent_size):
                        uncond_input[key][k] = F.interpolate(
                            v.float(),
                            size=(latent_size, latent_size),
                            mode='nearest' if 'mask' in k.lower() else 'bilinear',
                            align_corners=False if 'mask' not in k.lower() else None
                        )
                    else:
                        uncond_input[key][k] = v.clone()
                else:
                    uncond_input[key][k] = v
        else:
            uncond_input[key] = cond_input[key]
    
    # Get inpainting mode from semantic config
    inpainting_cfg = semantic_config.get('inpainting', {})
    mode = inpainting_cfg.get('mode', 'hard')
    
    print(f"\n✓ Inpainting mode: {mode}")
    
    # Print initial GPU memory
    print("\nInitial GPU memory:")
    print_gpu_memory()
    
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
        
        # For hard inpainting, encode ground truth semantic context
        mask_latent = None
        x_context = None
        
        if mode == "hard":
            with torch.no_grad():
                # Get ground truth semantic sample (the image we're trying to inpaint)
                # For semantic mode, dataset returns semantic tensor (buildings, streets, vegetation, height)
                if len(sample_data) == 2 and use_latents:
                    # Using pre-computed latents - im is already a latent
                    im_latent, _ = sample_data
                    x_context = im_latent.unsqueeze(0).to(device)
                    
                    # Ensure x_context matches expected latent dimensions
                    if x_context.shape[-2:] != (latent_size, latent_size):
                        x_context = F.interpolate(
                            x_context,
                            size=(latent_size, latent_size),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    print(f"✓ Using pre-computed latent context")
                    
                elif len(sample_data) == 2 and not use_latents:
                    # Not using latents - need to build semantic tensor and encode
                    # IMPORTANT: Use full-resolution data, NOT downsampled conditioning
                    if 'image' in cond_input_fullres and 'meta' in cond_input_fullres:
                        semantic_tensor = []
                        meta = cond_input_fullres['meta']
                        spatial_names = meta.get('spatial_names', [])
                        
                        # cond_input_fullres['image'] has shape [C, H, W] at FULL resolution (256x256)
                        print(f"\n✓ Building semantic tensor from FULL resolution: {cond_input_fullres['image'].shape}")
                        print(f"  Looking for channels: {semantic_channels}")
                        print(f"  Available channels: {spatial_names}")
                        
                        # Extract semantic channels based on configuration (same as train_vae_ddp.py)
                        # Only extract NON-context versions (buildings, streets, vegetation, height)
                        for sem_ch in semantic_channels:
                            found = False
                            print(f"  → Searching for: '{sem_ch}'")
                            for idx, name in enumerate(spatial_names):
                                # Skip _context channels - we only want the base semantic channels
                                if '_context' in name:
                                    continue
                                    
                                # Use exact matching only - fuzzy matching causes issues
                                # (e.g., 'osm:buildings' would match 'osm:buildings_heights')
                                print(f"    Comparing '{sem_ch}' == '{name}' ? {name == sem_ch}")
                                if name == sem_ch:
                                    # cond_input_fullres['image'] has shape [C, H, W] at FULL resolution
                                    # Extract channel: [C, H, W][idx:idx+1, :, :] -> [1, H, W]
                                    ch = cond_input_fullres['image'][idx:idx+1, :, :]
                                    semantic_tensor.append(ch)
                                    print(f"    ✓ Found {sem_ch} at index {idx}, shape={ch.shape}")
                                    found = True
                                    break
                            
                            if not found:
                                print(f"    ⚠ Warning: Semantic channel '{sem_ch}' not found. Filling with zeros.")
                                # Channel not found, create zeros at FULL resolution
                                C, H, W = cond_input_fullres['image'].shape
                                semantic_tensor.append(torch.zeros(1, H, W, device=cond_input_fullres['image'].device))
                        
                        im_semantic = torch.cat(semantic_tensor, dim=0).unsqueeze(0)  # [1, 4, H, W] at FULL res
                        print(f"  → Built semantic tensor: {im_semantic.shape}")
                        
                        # Encode ground truth semantics to latent space
                        im_semantic = im_semantic.to(device)
                        x_context, _, _ = vae.encode(im_semantic)
                        
                        # Ensure x_context matches expected latent dimensions
                        if x_context.shape[-2:] != (latent_size, latent_size):
                            x_context = F.interpolate(
                                x_context,
                                size=(latent_size, latent_size),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        print(f"✓ Built semantic tensor ({im_semantic.shape[1]} channels) and encoded to latent space")
                    else:
                        print(f"⚠ Could not build semantic tensor from conditioning input")
                        print(f"⚠ Falling back to pure noise generation")
                        mask_latent = None
                        x_context = None
                else:
                    print("⚠ Hard inpainting requested but no semantic ground truth available, using pure noise")
                    mask_latent = None
                    x_context = None
                
                # Setup mask if we have valid context
                if x_context is not None:
                    # Downsample mask to latent resolution
                    if mask_full is not None:
                        mask_latent = F.interpolate(
                            mask_full.float(),
                            size=(latent_size, latent_size),
                            mode='nearest'
                        )
                        print(f"✓ Downsampled mask: {mask_full.shape} → {mask_latent.shape}")
                    else:
                        print(f"⚠ Warning: No inpainting mask found, creating full mask (generate entire image)")
                        mask_latent = torch.ones(1, 1, latent_size, latent_size, device=device)
                    
                    # Initialize: keep context outside mask, noise inside mask
                    x = mask_latent * x + (1 - mask_latent) * x_context
                    
                    print(f"✓ Hard inpainting: preserving context outside mask ({(1 - mask_latent.mean()):.1%} of latent)")
                    print(f"  Mask latent stats: mean={mask_latent.mean():.4f}, min={mask_latent.min():.4f}, max={mask_latent.max():.4f}")
        
        # Sampling loop
        for i in tqdm(reversed(range(scheduler.num_timesteps)), desc="Denoising"):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            # Print GPU memory every 500 steps
            if i % 500 == 0:
                print_gpu_memory()
            
            # Classifier-free guidance with memory optimization
            if guidance_scale > 0:
                with torch.no_grad():
                    # Conditional prediction
                    noise_pred_cond = model(x, t, cond_input=cond_input)
                    torch.cuda.empty_cache()
                    
                    # Unconditional prediction
                    noise_pred_uncond = model(x, t, cond_input=uncond_input)
                    torch.cuda.empty_cache()
                
                # CFG
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                with torch.no_grad():
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
        semantic_samples = vae.decode(all_samples)
    
    # Clamp semantic values
    semantic_samples = torch.clamp(semantic_samples, 0, 1)
    
    # Save results
    out_dir = f"{repo_dir}/results/{task_name}/semantic_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get next run index
    base_name = f'semantics_cfg{guidance_scale}'
    if use_lst_guidance:
        base_name += f'_lst{lst_guidance_scale}'
    
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx -= 1
    
    print(f"\n{'='*50}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*50}")

    # Save visualization - each channel separately (like VAE training)
    for ch_idx, ch_name in enumerate(semantic_channels):
        if ch_idx < semantic_samples.shape[1]:
            ch = semantic_samples[:, ch_idx:ch_idx+1, :, :]
            
            if 'height' in ch_name:
                # Continuous height channel: normalize by max height
                ch_vis = torch.clamp(ch / 100.0, 0, 1)
            else:
                # Binary channels: already in [0, 1] after VAE decode
                ch_vis = torch.clamp(ch, 0, 1)
            
            # Also save original ground truth for comparison
            if mask_fullres is not None and 'image' in cond_input_fullres and 'meta' in cond_input_fullres:
                spatial_names_fullres = cond_input_fullres['meta'].get('spatial_names', [])
                for idx, name in enumerate(spatial_names_fullres):
                    # Skip _context channels - we only want the base semantic channels
                    if '_context' in name:
                        continue
                    
                    if name == ch_name or ch_name in name or name in ch_name:
                        # Extract original channel
                        ch_orig = cond_input_fullres['image'][idx:idx+1, :, :].unsqueeze(0).to(ch_vis.device)  # [1, 1, H, W]
                        
                        if 'height' in ch_name:
                            ch_orig_vis = torch.clamp(ch_orig / 100.0, 0, 1)
                        else:
                            ch_orig_vis = torch.clamp(ch_orig, 0, 1)
                        
                        # Replicate for all samples
                        ch_orig_vis = ch_orig_vis.repeat(num_samples, 1, 1, 1)
                        
                        # Save original without border
                        grid_orig = make_grid(ch_orig_vis, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
                        output_path_orig = os.path.join(out_dir, f'{base_name}_idx{run_idx}_{ch_name.replace(":", "_")}_original.png')
                        save_image(grid_orig, output_path_orig)
                        break
            
            # Overlay red mask border if mask is available
            if mask_fullres is not None:
                # Convert grayscale to RGB for red border overlay
                ch_vis_rgb = ch_vis.repeat(1, 3, 1, 1)  # [B, 3, H, W]
                
                # Compute mask boundary (edge detection)
                mask_tensor = mask_fullres.unsqueeze(0).float().to(ch_vis.device)  # [1, 1, H, W]
                
                # Create erosion kernel (3x3 all ones)
                kernel = torch.ones(1, 1, 3, 3, device=ch_vis.device)
                
                # Erode the mask (shrink it inward)
                import torch.nn.functional as F_conv
                mask_eroded = F_conv.conv2d(mask_tensor, kernel, padding=1)
                mask_eroded = (mask_eroded == 9).float()  # Only keep pixels where all 9 neighbors were 1
                
                # Boundary = original mask - eroded mask (pixels on the edge)
                mask_boundary = mask_tensor - mask_eroded
                mask_boundary = (mask_boundary > 0).float()
                
                # Repeat boundary for all samples
                mask_boundary = mask_boundary.repeat(num_samples, 1, 1, 1)
                
                # Apply red border (set R=1, G=0, B=0 where boundary)
                ch_vis_rgb[:, 0:1, :, :] = torch.where(mask_boundary > 0, torch.ones_like(ch_vis_rgb[:, 0:1, :, :]), ch_vis_rgb[:, 0:1, :, :])  # Red channel
                ch_vis_rgb[:, 1:2, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(ch_vis_rgb[:, 1:2, :, :]), ch_vis_rgb[:, 1:2, :, :])  # Green channel
                ch_vis_rgb[:, 2:3, :, :] = torch.where(mask_boundary > 0, torch.zeros_like(ch_vis_rgb[:, 2:3, :, :]), ch_vis_rgb[:, 2:3, :, :])  # Blue channel
                
                ch_vis_final = ch_vis_rgb
            else:
                ch_vis_final = ch_vis
            
            # Create grid for this channel
            grid = make_grid(ch_vis_final, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
            output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_{ch_name.replace(":", "_")}.png')
            save_image(grid, output_path)
    
    # Save mask visualization
    if mask_fullres is not None:
        # Use full-resolution mask for visualization
        mask_vis = mask_fullres.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # [B, 1, H, W]
        grid = make_grid(mask_vis, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
        output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_inpainting_mask.png')
        save_image(grid, output_path)
        print(f"✓ Saved inpainting mask visualization")
    elif mask_full is not None:
        # Fallback to upsampling if full-res not available
        mask_upsampled = F.interpolate(
            mask_full,
            size=(semantic_samples.shape[2], semantic_samples.shape[3]),
            mode='nearest'
        )
        mask_vis = mask_upsampled.repeat(num_samples, 1, 1, 1)
        grid = make_grid(mask_vis, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
        output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_inpainting_mask.png')
        save_image(grid, output_path)
        print(f"✓ Saved inpainting mask visualization (upsampled)")
    
    print(f"\n✓ Saved {len(semantic_channels)} channel visualizations to {out_dir}")    # Save individual samples as .pt files for Stage 2
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
    
    data_dir = f"{big_data_storage_path}/results/{train_config['task_name']}"
    vae_path = os.path.join(data_dir, semantic_train_config.get('autoencoder_ckpt_name', 'semantic_vae_ddp_ckpt.pth'))
    
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
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing for memory efficiency")
    
    ldm_path = os.path.join(data_dir, semantic_train_config.get('ldm_ckpt_name', 'semantic_ldm_ddp_ckpt.pth'))
    
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
        lst_predictor_path = os.path.join(data_dir, base_train_config.get('lst_predictor_ckpt_name', 'lst_predictor_best.pth'))
        lst_predictor = load_lst_predictor(lst_predictor_path, device)
    
    ########## Sample Semantics #############
    samples = sample_semantics(
        model=model,
        repo_dir=config.get('repo_dir', '.'),
        scheduler=scheduler,
        train_config=train_config,
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

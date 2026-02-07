"""
Sampling script for Stage 2: Satellite Rendering from Semantics
"""

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import glob

# Data handling
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# Local libraries
from model.blocks.unet_cond_base import Unet
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.vae_registry import VAERegistry
from model.utils.config_utils import compute_patch_and_latent_sizes, build_unet_condition_config
from model.utils.checkpoint import load_checkpoint, check_existing_paths
from model.utils.diffusion_utils import (
    mask_conditioning_latents,
    apply_seam_mode,
    sample_with_repaint,
    make_uncond_input_keep_mask
)
from model.scheduler.scheduler_factory import get_inpainting_sampler_for_stage
from helpers.load_configs import load_configs
from helpers.indexed_outputs import get_next_run_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_semantic_samples(semantic_dir):
    """
    Load semantic samples from Stage 1 directory.
    
    Args:
        semantic_dir: Directory containing saved semantic samples (e.g., semantics_cfg7.5_idx0_samples/)
        
    Returns:
        List of dictionaries with semantic data
    """
    sample_files = sorted(glob.glob(os.path.join(semantic_dir, 'sample_*.pt')))
    
    if not sample_files:
        print(f"✗ No semantic samples found in {semantic_dir}")
        print(f"  Expected format: {semantic_dir}/sample_*.pt")
        return []
    
    samples = []
    for sample_file in sample_files:
        sample_data = torch.load(sample_file, map_location='cpu', weights_only=False)
        samples.append(sample_data)
    
    print(f"✓ Loaded {len(samples)} semantic samples from {semantic_dir}")
    return samples


def render_satellite_from_semantics(
    semantic_samples,
    model,
    scheduler,
    vae_registry: VAERegistry,
    config,
    stage_config,
    dataset,
    guidance_scale=7.5,
    num_samples=None,
    overwrite_samples=False,
    existing_vae_paths=None
):
    """
    Render satellite imagery from semantic layouts using Stage 2 diffusion model.

    Args:
        semantic_samples: List of semantic sample dicts from Stage 1
        model: Trained satellite diffusion U-Net
        scheduler: Noise scheduler
        vae_registry: VAERegistry with all VAE models
        config: Full config dict
        stage_config: Satellite diffusion stage config
        dataset: UrbanInpaintingDataset for loading environmental conditioning
        guidance_scale: Classifier-free guidance scale
        num_samples: Number of samples to render (None = all)
        overwrite_samples: Whether to overwrite existing samples
        existing_vae_paths: Dict of group_name -> checkpoint_path overrides
        
    Returns:
        Generated satellite images [N, 3, H, W]
    """
    model.eval()
    
    # Initialize existing paths if not provided
    if existing_vae_paths is None:
        existing_vae_paths = {}
    
    # get repo directory
    repo_dir = config.get('repo_dir', '.')
    
    # Extract configs
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    vae_groups = config['vae_groups']
    
    pred_group = stage_config.get('prediction_group')
    conditioning_config = stage_config.get('conditioning', {})
    inpainting_cfg = stage_config.get('inpainting', {})
    
    # Get VAE configs
    pred_vae_config = vae_groups[pred_group]
    unet_config = stage_config.get('unet_config', {})
    
    # Compute sizes
    im_size, latent_size, vae_factor, unet_factor, total_divisor = compute_patch_and_latent_sizes(
        dataset_config,
        pred_vae_config,
        unet_config,
        use_latents=True
    )
    
    # Get inpainting mode and sampling-time masking config
    inpainting_mode = inpainting_cfg.get('mode', 'sdlike')
    cfg_config = inpainting_cfg.get('cfg', {})
    sample_mask_groups = cfg_config.get('sample_mask_groups', [])
    
    # Seam improvement configuration
    seam_config = inpainting_cfg.get('seam', {})
    seam_mode_sampling = seam_config.get('sampling', None) if isinstance(seam_config, dict) else None
    seam_settings = seam_config.get('config', {}) if isinstance(seam_config, dict) else {}
    blur_radius = seam_settings.get('blur_radius', 3)
    
    # Initialize inpainting sampler (RePaint/LanPaint) if configured
    sampler_cfg = inpainting_cfg.get('sampler', {'type': 'standard'})
    inpainting_sampler_type = sampler_cfg.get('type', 'standard')
    inpainting_sampler = None
    if inpainting_sampler_type != 'standard':
        inpainting_sampler = get_inpainting_sampler_for_stage(config, 'satellite', device)
        print(f"✓ Loaded {inpainting_sampler_type.upper()} inpainting sampler for satellite stage")
    
    print("\n" + "="*60)
    print("Satellite Rendering Configuration")
    print("="*60)
    print(f"Image size: {im_size}x{im_size} ({im_size * dataset_config['res']}m)")
    print(f"Latent size: {latent_size}x{latent_size}")
    print(f"VAE downsample: {vae_factor}x, U-Net downsample: {unet_factor}x")
    print(f"Prediction group: {pred_group}")
    print(f"Number of samples: {num_samples or len(semantic_samples)}")
    print(f"Guidance scale (CFG): {guidance_scale}")
    print(f"Inpainting mode: {inpainting_mode}")
    print(f"Sample mask groups (sampling-time): {sample_mask_groups}")
    print(f"Seam mode (sampling): {seam_mode_sampling if seam_mode_sampling else 'None'}")
    
    # Load prediction VAE (satellite) - use existing_vae_paths if available
    big_data_storage_path = dataset_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    task_name = train_config.get('task_name', 'urban_inpainting')
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    
    if pred_group in existing_vae_paths:
        vae_path = existing_vae_paths[pred_group]
        print(f"✓ Using existing VAE path for {pred_group}: {vae_path}")
    else:
        vae_checkpoint = pred_vae_config.get('checkpoint_name', f'{pred_group}_vae_ckpt.pth')
        vae_path = os.path.join(data_dir, vae_checkpoint)
    vae_registry.load_vae(
        group_name=pred_group,
        checkpoint_path=vae_path,
        autoencoder_config=pred_vae_config,
    )
    pred_vae = vae_registry.get_vae(pred_group)
    
    # Load conditioning VAEs (semantic, environmental) - use existing_vae_paths if available
    latent_cond_groups = conditioning_config.get('latent_space', [])
    for cond_spec in latent_cond_groups:
        cond_group = cond_spec['group']
        cond_vae_config = vae_groups[cond_group]
        
        if cond_group in existing_vae_paths:
            cond_vae_path = existing_vae_paths[cond_group]
            print(f"✓ Using existing VAE path for {cond_group}: {cond_vae_path}")
        else:
            cond_checkpoint = cond_vae_config.get('checkpoint_name', f'{cond_group}_vae_ckpt.pth')
            cond_vae_path = os.path.join(data_dir, cond_checkpoint)
        
        vae_registry.load_vae(
            group_name=cond_group,
            checkpoint_path=cond_vae_path,
            autoencoder_config=cond_vae_config,
        )
        print(f"✓ Loaded {cond_group} VAE for conditioning")
    
    # Limit number of samples if specified
    if num_samples is not None:
        semantic_samples = semantic_samples[:num_samples]
    
    # Setup output directories BEFORE sampling loop
    task_name = train_config.get('task_name', 'urban_inpainting')
    out_dir = f"{repo_dir}/results/{task_name}/satellite_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get run index
    base_name = f'satellite_cfg{guidance_scale}'
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*60}")
    
    # Create samples directory
    samples_dir = os.path.join(out_dir, f'{base_name}_idx{run_idx}_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    ################# Sampling Loop ########################
    print("\n" + "="*60)
    print("Starting Satellite Rendering")
    print("="*60)
    
    all_renders = []
    
    for sample_idx, sample_data in enumerate(semantic_samples):
        print(f"\nRendering sample {sample_idx + 1}/{len(semantic_samples)}")
        
        # Extract data from Stage 1 output
        semantic_tensor = sample_data['semantic_tensor']  # [C, H, W] in pixel space
        semantic_channels = sample_data['semantic_channels']
        mask = sample_data.get('mask')  # [1, H, W] in pixel space
        
        if mask is None:
            print("⚠ No mask found in sample, creating full mask")
            mask = torch.ones(1, semantic_tensor.shape[1], semantic_tensor.shape[2])
        
        # Move to device
        semantic_tensor = semantic_tensor.to(device).unsqueeze(0)  # [1, C, H, W]
        mask = mask.to(device).unsqueeze(0)  # [1, 1, H, W]
        
        # Build conditioning dict
        cond_input = {'meta': [{}]}
        
        # 1. Pixel-space conditioning: inpainting mask (downsampled to latent resolution)
        pixel_cond_list = []
        pixel_cond_names = []
        
        for cond_spec in conditioning_config.get('pixel_space', []):
            layer_name = cond_spec.get('layer')
            interpolation_mode = cond_spec.get('interpolation', 'nearest')
            
            if layer_name == 'inpainting_mask':
                # Downsample mask to latent resolution
                mask_latent = F.interpolate(
                    mask.float(),
                    size=(latent_size, latent_size),
                    mode=interpolation_mode
                )
                pixel_cond_list.append(mask_latent)
                pixel_cond_names.append('inpainting_mask')
        
        if pixel_cond_list:
            cond_input['image'] = torch.cat(pixel_cond_list, dim=1)  # [1, C_pixel, H_latent, W_latent]
            cond_input['meta'][0]['pixel_space_names'] = pixel_cond_names
        
        # Track latent group names for metadata
        latent_group_names = []
        
        # Load RGB context from dataset (for both hard and sdlike modes)
        # For hard mode: used during generation to preserve context in latent space
        # For sdlike mode: composited after decoding to preserve original context
        rgb_context_latent = None
        rgb_context_image = None
        patch_index = sample_data.get('patch_index')
        
        if patch_index is not None and dataset is not None:
            # Load matching patch from dataset to get RGB context
            dataset_sample = dataset[patch_index]
            
            if isinstance(dataset_sample, tuple) and len(dataset_sample) == 2:
                pred_data, dataset_cond = dataset_sample
            else:
                pred_data = None
                dataset_cond = {}
            
            # Get RGB context - either pre-encoded latent or full-res image
            if pred_data is not None:
                # Check if it's already a latent or needs encoding
                if pred_data.shape[-2:] == (latent_size, latent_size):
                    # Already encoded latent - decode it for pixel-space compositing
                    rgb_context_latent = pred_data.unsqueeze(0).to(device)
                    print(f"  ✓ Loaded RGB context latent from dataset: {rgb_context_latent.shape}")
                    
                    # Decode to pixel space for final compositing
                    with torch.no_grad():
                        rgb_context_image = pred_vae.decode(rgb_context_latent)
                        # Normalize
                        if pred_vae_config.get('tanh_activation', False):
                            rgb_context_image = torch.clamp(rgb_context_image, -1., 1.)
                            rgb_context_image = (rgb_context_image + 1) / 2
                        else:
                            rgb_context_image = torch.clamp(rgb_context_image, 0., 1.)
                    print(f"  ✓ Decoded RGB context to image: {rgb_context_image.shape}")
                else:
                    # Full-res image - use directly
                    rgb_context_image = pred_data.unsqueeze(0).to(device)
                    print(f"  ✓ Loaded RGB context image from dataset: {rgb_context_image.shape}")
                    
                    # Also encode to latent if needed for hard mode
                    if inpainting_mode == 'hard':
                        with torch.no_grad():
                            rgb_context_latent, _, _ = pred_vae.encode(rgb_context_image)
                        print(f"  ✓ Encoded RGB context to latent: {rgb_context_latent.shape}")
        
        # 2. Latent-space conditioning: encode semantic and environmental
        for cond_spec in latent_cond_groups:
            cond_group = cond_spec['group']
            cond_layers = cond_spec.get('layers', [])
            
            if cond_group == 'semantic':
                # Use generated semantic from Stage 1
                # Encode to latent space
                cond_vae = vae_registry.get_vae(cond_group)
                with torch.no_grad():
                    semantic_latent, _, _ = cond_vae.encode(semantic_tensor)
                
                cond_input[cond_group] = semantic_latent  # [1, C_latent, H_latent, W_latent]
                latent_group_names.append(cond_group)
                print(f"  ✓ Encoded semantic to latent: {semantic_latent.shape}")
                
            elif cond_group == 'environmental':
                # Get environmental conditioning from dataset using patch metadata
                patch_index = sample_data.get('patch_index')
                
                if patch_index is not None and dataset is not None:
                    # Load matching patch from dataset (may be latent or full-res image)
                    dataset_sample = dataset[patch_index]
                    
                    if isinstance(dataset_sample, tuple) and len(dataset_sample) == 2:
                        _, dataset_cond = dataset_sample
                    else:
                        dataset_cond = {}
                    
                    # Check if environmental is already latent or needs encoding
                    if cond_group in dataset_cond:
                        # Already encoded latent
                        env_latent = dataset_cond[cond_group].unsqueeze(0).to(device)
                        cond_input[cond_group] = env_latent
                        latent_group_names.append(cond_group)
                        print(f"  ✓ Loaded environmental conditioning latent from dataset patch {patch_index}")
                    elif f'{cond_group}_image' in dataset_cond:
                        # Full-res image needs encoding
                        print(f"  ⚠ {cond_group} latent not available, encoding on-the-fly")
                        cond_image = dataset_cond[f'{cond_group}_image'].unsqueeze(0).to(device)
                        
                        # Encode using environmental VAE
                        env_vae = vae_registry.get_vae(cond_group)
                        if env_vae is None:
                            raise ValueError(f"Environmental VAE not loaded in registry")
                        
                        with torch.no_grad():
                            cond_latent, _, _ = env_vae.encode(cond_image)
                        
                        cond_input[cond_group] = cond_latent
                        latent_group_names.append(cond_group)
                        print(f"  ✓ Encoded {cond_group} conditioning from image: {cond_image.shape} → {cond_latent.shape}")
                    else:
                        print(f"  ⚠ Context group '{cond_group}' not found in dataset, using zeros")
                        z_channels = vae_groups[cond_group]['z_channels']
                        cond_latent = torch.zeros(1, z_channels, latent_size, latent_size, device=device)
                        cond_input[cond_group] = cond_latent
                        latent_group_names.append(cond_group)
                else:
                    # Fallback to zeros if no patch metadata
                    print(f"  ⚠ No patch_index in sample metadata, using zero environmental conditioning")
                    z_channels = vae_groups[cond_group]['z_channels']
                    cond_latent = torch.zeros(1, z_channels, latent_size, latent_size, device=device)
                    cond_input[cond_group] = cond_latent
                    latent_group_names.append(cond_group)
        
        # Store latent group names in metadata
        if latent_group_names:
            cond_input['meta'][0]['latent_group_names'] = latent_group_names
        
        # 3. Apply sampling-time mask to specified conditioning groups (e.g., environmental)
        if sample_mask_groups and 'image' in cond_input and 'pixel_space_names' in cond_input['meta'][0]:
            pixel_names = cond_input['meta'][0]['pixel_space_names']
            if 'inpainting_mask' in pixel_names:
                mask_idx = pixel_names.index('inpainting_mask')
                mask_latent = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                
                # Mask conditioning latents (zeros out masked region) - sampling-time only!
                cond_input = mask_conditioning_latents(cond_input, mask_latent, sample_mask_groups)
                print(f"  ✓ Applied sampling-time mask to groups: {sample_mask_groups}")
        
        # For inpainting CFG, unconditional branch must see the mask, only latent groups are zeroed
        uncond_input = make_uncond_input_keep_mask(cond_input)
        
        # Initialize latent for satellite generation
        x = torch.randn(1, pred_vae_config['z_channels'], latent_size, latent_size, device=device)
        
        # Get mask at latent resolution
        mask_latent = F.interpolate(mask.float(), size=(latent_size, latent_size), mode='nearest')
        
        # FIX: Create fixed noise_context once per sample for temporal consistency
        noise_context = None
        noise_context_sd = None
        
        # For hard inpainting, initialize with RGB context from dataset
        if inpainting_mode == "hard":
            if rgb_context_latent is None:
                print(f"  ⚠ No RGB context available; using zeros outside mask (hard mode)")
                rgb_context_latent = torch.zeros_like(x)
            
            # Initialize: keep context outside mask, noise inside mask
            x = mask_latent * x + (1 - mask_latent) * rgb_context_latent
            print(f"  ✓ Hard mode: Initialized with RGB context from dataset")
            
            # FIX: Sample noise_context ONCE per sample for temporal consistency
            noise_context = torch.randn_like(rgb_context_latent)
        
        # For SD-like mode, prepare fixed noise for per-step outside reinsertion
        if inpainting_mode == "sdlike" and rgb_context_latent is not None:
            noise_context_sd = torch.randn_like(rgb_context_latent)
        
        # =====================================================================
        # INPAINTING SAMPLING
        # =====================================================================
        
        if inpainting_sampler is not None and inpainting_mode == "hard" and rgb_context_latent is not None:
            # Use advanced inpainting sampler (RePaint or LanPaint)
            print(f"  Using {inpainting_sampler_type.upper()} sampler...")
            x = inpainting_sampler.sample(
                model=model,
                x_init=x,
                x_context=rgb_context_latent,
                mask=mask_latent,
                cond_input=cond_input,
                uncond_input=uncond_input if guidance_scale > 0 else None,
                guidance_scale=guidance_scale,
                show_progress=True
            )
        else:
            # Standard sampling loop
            print(f"  Denoising {scheduler.num_timesteps} steps...")
            with torch.no_grad():
                for i in tqdm(reversed(range(scheduler.num_timesteps)), desc="  ", leave=False):
                    t = torch.full((1,), i, device=device, dtype=torch.long)
                    
                    # Classifier-free guidance
                    if guidance_scale > 0:
                        noise_pred_cond = model(x, t, cond_input=cond_input)
                        noise_pred_uncond = model(x, t, cond_input=uncond_input)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = model(x, t, cond_input=cond_input)
                    
                    # FIX: Denoise step with proper inpainting behavior
                    if inpainting_mode == "hard":
                        # Hard mode: use inpainting scheduler with fixed noise_context
                        x, x0 = scheduler.sample_prev_timestep_inpainting(
                            x, noise_pred, i, rgb_context_latent, mask_latent, noise_context=noise_context
                        )
                    else:
                        # SD-like: standard denoising
                        x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
                        
                        # RECOMMENDED: Enforce outside distribution each step for SD-like mode
                        # Prevents outside drift from affecting inside, improves seam quality
                        if rgb_context_latent is not None and noise_context_sd is not None and i > 0:
                            t_batch = torch.full((1,), i-1, device=device, dtype=torch.long)
                            x_context_noisy = scheduler.add_noise(rgb_context_latent, noise_context_sd, t_batch)
                            x = mask_latent * x + (1 - mask_latent) * x_context_noisy
        
        # Decode latent to RGB immediately
        print(f"  Decoding sample {sample_idx + 1}...")
        with torch.no_grad():
            rgb_render = pred_vae.decode(x)
        
        # Normalize to [0, 1]
        if pred_vae_config.get('tanh_activation', False):
            rgb_render = torch.clamp(rgb_render, -1., 1.)
            rgb_render = (rgb_render + 1) / 2
        else:
            rgb_render = torch.clamp(rgb_render, 0., 1.)
        
        # Apply seam mode: feathering for smooth compositing
        if seam_mode_sampling == 'feather' and rgb_context_image is not None:
            # Upsample mask to image resolution
            mask_pixel = F.interpolate(mask, size=rgb_render.shape[-2:], mode='nearest')
            
            # Apply feathering
            feathered_mask, _ = apply_seam_mode('feather', mask=mask_pixel, blur_radius=blur_radius)
            
            # Smooth composite with feathered mask
            rgb_render = feathered_mask * rgb_render + (1 - feathered_mask) * rgb_context_image
            print(f"  ✓ Applied feathered compositing (blur_radius={blur_radius})")
        
        # Standard composite with original RGB context (if no feathering applied)
        elif rgb_context_image is not None:
            # Upsample mask to image resolution
            mask_pixel = F.interpolate(mask, size=rgb_render.shape[-2:], mode='nearest')
            
            # Paste: keep original context outside mask, use generated inside mask
            rgb_render = mask_pixel * rgb_render + (1 - mask_pixel) * rgb_context_image
            print(f"  ✓ Composited with original RGB context")
        
        # Save individual sample visualization immediately
        sample_path = os.path.join(samples_dir, f'sample_{sample_idx}.png')
        save_image(rgb_render[0], sample_path)
        print(f"  ✓ Saved visualization: {sample_path}")
        
        # Also save as .pt for further processing
        sample_pt_path = os.path.join(samples_dir, f'sample_{sample_idx}.pt')
        torch.save({
            'rgb_tensor': rgb_render[0].cpu(),
            'semantic_source': semantic_samples[sample_idx].get('semantic_channels', []),
        }, sample_pt_path)
        print(f"  ✓ Saved tensor: {sample_pt_path}")
        
        # Keep for final grid
        all_renders.append(rgb_render)
        
        # Free GPU memory between samples
        torch.cuda.empty_cache()
    
    # Stack all RGB renders for final grid
    rgb_renders = torch.cat(all_renders, dim=0)  # [N, 3, H, W]
    
    # Save final grid visualization
    print("\nCreating grid visualization...")
    grid = make_grid(rgb_renders, nrow=int(np.sqrt(len(semantic_samples))) + 1, 
                    padding=4, pad_value=1.0)
    output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}.png')
    save_image(grid, output_path)
    print(f"✓ Saved grid visualization to {output_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Completed! All {len(semantic_samples)} samples saved to {samples_dir}")
    print(f"{'='*60}")
    
    return rgb_renders


def infer(args, config):
    """Main inference function"""
    
    ###### setup config variables #######
    dataset_config = config['dataset_params']
    diffusion_config = config['diffusion_params']
    train_config = config['train_params']
    vae_groups = config['vae_groups']
    diffusion_stages = config['diffusion_stages']
    
    repo_dir = config.get('repo_dir', '.')
    
    big_data_storage_path = dataset_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    
    # Mode is 'satellite' for Stage 2
    mode = 'satellite'
    
    # Validate mode
    if mode not in diffusion_stages:
        raise ValueError(
            f"Diffusion stage '{mode}' not found in config. "
            f"Available stages: {list(diffusion_stages.keys())}"
        )
    
    # Get stage config
    stage_config = diffusion_stages[mode]
    pred_group = stage_config.get('prediction_group')
    
    if pred_group not in vae_groups:
        raise ValueError(
            f"Prediction group '{pred_group}' not found in vae_groups. "
            f"Available: {list(vae_groups.keys())}"
        )
    
    # Get configs
    vae_config = vae_groups[pred_group]
    unet_config = stage_config.get('unet_config', {})
    
    # ========== Check for existing paths (use override paths from config) ==========
    # Check diffusion paths for satellite mode
    existing_diffusion = check_existing_paths(
        train_config=train_config,
        mode=mode,
        type='diffusion'
    )
    
    # Check VAE paths for prediction group (satellite)
    existing_vae_satellite = check_existing_paths(
        train_config=train_config,
        mode=pred_group,
        type='vae'
    )
    
    # Also check VAE paths for conditioning groups (semantic, environmental)
    conditioning_config = stage_config.get('conditioning', {})
    latent_cond_groups = conditioning_config.get('latent_space', [])
    
    existing_vae_paths = existing_vae_satellite.vae_checkpoints.copy()
    for cond_spec in latent_cond_groups:
        cond_group = cond_spec.get('group')
        if cond_group:
            cond_vae_result = check_existing_paths(
                train_config=train_config,
                mode=cond_group,
                type='vae'
            )
            existing_vae_paths.update(cond_vae_result.vae_checkpoints)
    
    # Print any warnings
    all_warnings = existing_diffusion.warnings + existing_vae_satellite.warnings
    for warning in all_warnings:
        print(f"⚠ {warning}")
    
    # Get resolved paths
    existing_diffusion_path = existing_diffusion.diffusion_checkpoint
    existing_patches_path = existing_diffusion.patches_path or existing_vae_satellite.patches_path
    
    print(f"\n{'='*60}")
    print(f"Satellite Rendering from Semantics")
    print(f"{'='*60}")
    print(f"Prediction group: {pred_group}")
    print(f"Stage config: {mode}")
    
    ########## Create Scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    print(f"\n✓ Created noise scheduler with {diffusion_config['num_timesteps']} timesteps")
    
    ########## Load Models #############
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    # Initialize VAE Registry
    vae_registry = VAERegistry(config, device)
    
    # Build condition_config for U-Net (same as training)
    condition_config = build_unet_condition_config(stage_config, vae_groups)
    
    # Add condition_config to unet_config
    unet_config_with_cond = unet_config.copy()
    unet_config_with_cond['condition_config'] = condition_config
    
    # Load Satellite Diffusion Model
    model = Unet(
        im_channels=vae_config['z_channels'],
        model_config=unet_config_with_cond
    ).to(device)
    model.eval()
    
    # Load checkpoint
    task_name = train_config.get('task_name', 'urban_inpainting')
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    
    # Get diffusion checkpoint path (use existing_paths if available)
    if existing_diffusion_path is not None:
        ldm_path = existing_diffusion_path
        print(f"✓ Using existing diffusion checkpoint from config: {ldm_path}")
    else:
        diffusion_train_config = train_config.get('diffusion_training', {}).get(mode, {})
        ldm_checkpoint = diffusion_train_config.get('checkpoint_name', f'{mode}_diffusion_ckpt.pth')
        ldm_path = os.path.join(data_dir, ldm_checkpoint)
    
    if os.path.exists(ldm_path):
        _, _ = load_checkpoint(
            checkpoint_path=ldm_path,
            model=model,
            optimizer=None,
            device=device,
            is_main=True
        )
    else:
        print(f"✗ Satellite Diffusion Model not found at {ldm_path}")
        print(f"  Train the satellite diffusion model first!")
        return
    
    ########## Load Semantic Samples #############
    # Use semantic_dir from args, or default to config-based path
    if args.semantic_dir:
        semantic_dir = args.semantic_dir
    else:
        # Default to semantic_output directory from config (same as semantic sampling)
        semantic_output_dir = f"{repo_dir}/results/{task_name}/semantic_output"
        
        # Find the latest semantics sample directory
        import glob
        sample_dirs = glob.glob(os.path.join(semantic_output_dir, 'semantics_cfg*_idx*_samples'))
        
        if not sample_dirs:
            print(f"\n✗ No semantic sample directories found in {semantic_output_dir}")
            print(f"  Please run semantic sampling first or specify --semantic_dir")
            return
        
        # Use the most recent directory
        semantic_dir = max(sample_dirs, key=os.path.getmtime)
        print(f"\n✓ Using most recent semantic samples: {os.path.basename(semantic_dir)}")
    
    semantic_samples = load_semantic_samples(semantic_dir)
    
    if not semantic_samples:
        return
    
    # Set seed for reproducibility (MUST be same as Stage 1 for dataset consistency)
    seed = train_config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n✓ Set random seed: {seed}")
    
    ########## Load Dataset for RGB Context and Environmental Conditioning #############
    print("\n" + "="*60)
    print("Loading Dataset for RGB Context and Environmental Conditioning")
    print("="*60)
    
    # Load dataset for:
    # 1. RGB satellite context (to preserve original imagery outside inpainting mask)
    # 2. Environmental conditioning (LST, NDVI)
    dataset = None
    try:
        dataset = UrbanInpaintingDataset(
            split='train',
            mode=f'diffusion:{mode}',
            use_cached_patches=True
        )
        print(f"✓ Loaded dataset with {len(dataset)} patches")
    except Exception as e:
        print(f"⚠ Failed to load dataset: {e}")
        print(f"  Will use zero environmental conditioning and no RGB context")
        dataset = None
    
    ########## Render Satellite Images #############
    renders = render_satellite_from_semantics(
        semantic_samples=semantic_samples,
        model=model,
        scheduler=scheduler,
        vae_registry=vae_registry,
        config=config,
        stage_config=stage_config,
        dataset=dataset,
        guidance_scale=args.guidance_scale,
        num_samples=args.num_samples,
        overwrite_samples=args.overwrite_samples,
        existing_vae_paths=existing_vae_paths
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Satellite Rendering Complete!")
    print(f"{'='*60}")
    
    return renders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render satellite imagery from semantic layouts (Stage 2)')
    parser.add_argument('--semantic_dir', type=str, default=None,
                       help='Directory containing Stage 1 semantic samples (e.g., semantics_cfg7.5_idx0_samples/). If not specified, uses latest from config output directory.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, 
                       help='Classifier-free guidance scale')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to render (None = all)')
    parser.add_argument('--overwrite_samples', action='store_true', 
                       help='Overwrite existing samples (use run_idx=0)')
    parser.add_argument('--config', type=str, default='two_stage_4.yml', 
                       help='Config file name (default: two_stage_4.yml)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        os.environ['CONFIG_PATH'] = args.config
    
    config = load_configs()
    
    infer(args, config)

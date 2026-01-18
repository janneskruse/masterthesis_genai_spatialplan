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
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.config_utils import compute_patch_and_latent_sizes
from model.utils.layer_config import count_layer_channels
from model.utils.vae_registry import VAERegistry
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
    pred_vae, 
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
        semantic_pred = pred_vae.decode(x0_pred)
    
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
    stage_config,
    big_data_storage_path, 
    vae_registry: VAERegistry,
    vae_groups,
    pred_group,
    mode='semantic',
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
        stage_config: Diffusion stage configuration
        big_data_storage_path: Data storage path
        vae_registry: VAERegistry instance with loaded VAEs
        vae_groups: VAE groups config
        pred_group: Prediction group name
        mode: Diffusion mode (e.g., 'semantic')
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
    
    # Validate prediction group
    if pred_group not in vae_groups:
        raise ValueError(f"Prediction group '{pred_group}' not found in VAE groups")
    
    # Get layers for this group to determine channels for visualization
    pred_group_config = vae_groups[pred_group]
    semantic_layers = pred_group_config.get('layers', [])
    
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
    print(f"Prediction group: {pred_group}")
    print(f"Semantic layers: {semantic_layers}")
    
    # Load dataset to get conditioning examples
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    print(f"\n✓ Loading dataset in diffusion mode: 'diffusion:{mode}'")
    
    dataset = UrbanInpaintingDataset(
        split='val',
        mode=f'diffusion:{mode}',
        use_cached_patches=True
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
    
    # Dataset in diffusion mode returns (pred_latent, cond_dict)
    pred_latent, cond_input = dataset[sample_idx]
    
    # Get prediction VAE from registry
    pred_vae = vae_registry.get_vae(pred_group)
    if pred_vae is None:
        raise ValueError(f"Prediction VAE for group '{pred_group}' not loaded in registry")
    
    # Check if pred_latent is actually a latent or full-res image that needs encoding
    needs_encoding = False
    if pred_latent.shape[-2:] != (latent_size, latent_size):
        # Full resolution image - needs encoding
        needs_encoding = True
        pred_image = pred_latent  # Rename for clarity
        
        print(f"\n⚠ No pre-computed prediction latent found, encoding on-the-fly")
        print(f"  Prediction image shape: {pred_image.shape}")
        
        # Encode to latent space
        with torch.no_grad():
            pred_image_batch = pred_image.unsqueeze(0).to(device)
            pred_latent, _, _ = pred_vae.encode(pred_image_batch)
            pred_latent = pred_latent.squeeze(0).cpu()  # Back to [C, H, W]
        
        print(f"  Encoded to latent: {pred_latent.shape}")
    
    # Check for latent-space conditioning groups that need encoding
    # Dataset marks these with "{group_name}_image" suffix
    groups_to_encode = []
    for key in list(cond_input.keys()):
        if key.endswith('_image') and key != 'image':  # Exclude pixel-space 'image'
            group_name = key[:-6]  # Remove '_image' suffix
            groups_to_encode.append((group_name, key))
    
    if groups_to_encode:
        print(f"\n⚠ Found {len(groups_to_encode)} latent-space conditioning groups needing encoding")
        
        for group_name, image_key in groups_to_encode:
            print(f"  Encoding {group_name}...")
            
            # Get VAE from registry (load if not already loaded)
            group_vae = vae_registry.get_vae(group_name)
            if group_vae is None:
                # Load VAE for this group

                group_vae_config = vae_groups[group_name].get('vae', {})
                ckpt_name = group_vae_config.get('checkpoint_name', f'{group_name}_vae_ckpt.pth')
                ckpt_path = os.path.join(out_dir, ckpt_name)
                
                group_vae = vae_registry.load_vae(
                    group_name=group_name,
                    checkpoint_path=ckpt_path,
                    autoencoder_config=group_vae_config,
                )
            
            # Encode the full-res image to latent
            with torch.no_grad():
                group_image = cond_input[image_key]
                group_image_batch = group_image.unsqueeze(0).to(device)
                group_latent, _, _ = group_vae.encode(group_image_batch)
                group_latent = group_latent.squeeze(0).cpu()  # Back to [C, H, W]
            
            # Replace image with latent in conditioning dict
            cond_input[group_name] = group_latent
            del cond_input[image_key]  # Remove the _image key
            
            print(f"    Shape: {group_image.shape} → {group_latent.shape}")
    
    # Prepare conditioning inputs for batch
    for key in cond_input:
        # unsqueeze batch dimension
        if isinstance(cond_input[key], torch.Tensor):
            cond_input[key] = cond_input[key].unsqueeze(0).to(device)
        elif key == 'meta':
            # Meta is a dict - keep as is (will be wrapped in list by collate_fn)
            pass
    
    # Extract mask and LST target from conditioning channels
    mask_latent = None
    lst_target = None
    
    if 'image' in cond_input and 'meta' in cond_input:
        # Access pixel_space_names from metadata
        pixel_space_names = cond_input['meta'].get('pixel_space_names', [])
        print(f"\n✓ Pixel-space conditioning channels ({len(pixel_space_names)}):")
        for idx, name in enumerate(pixel_space_names):
            ch = cond_input['image'][0, idx:idx+1, :, :]
            print(f"  {idx:02d} {name:40s} shape={tuple(ch.shape)} mean={ch.mean():.4f}")
        
        # Extract inpainting mask
        try:
            mask_idx = pixel_space_names.index('inpainting_mask')
            mask_latent = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
            print(f"\n✓ Found inpainting mask at index {mask_idx}")
            print(f"  Mask coverage: {mask_latent.mean():.2%} (1=inpaint, 0=keep)")
        except (ValueError, IndexError):
            print(f"\n⚠ Warning: No inpainting_mask found in pixel_space_names")
        
        # Extract LST target (if available in pixel-space conditioning)
        for idx, name in enumerate(pixel_space_names):
            if 'lst' in name.lower():
                lst_target = cond_input['image'][:, idx:idx+1, :, :]
                print(f"✓ Found LST target channel: {name}")
                break
    
    if use_lst_guidance and lst_target is None:
        print("⚠ LST guidance requested but no LST target found in data")
        use_lst_guidance = False
    
    # Create unconditional input for CFG
    # Zero out image conditioning (but keep meta for structural info)
    uncond_input = {}
    for key in cond_input:
        if key == 'image':
            # Zero out image conditioning for CFG
            uncond_input[key] = torch.zeros_like(cond_input[key])
        else:
            # Keep meta and other keys as is
            uncond_input[key] = cond_input[key]
    
    # Get inpainting mode from stage config
    inpainting_cfg = stage_config.get('inpainting', {})
    inpainting_mode = inpainting_cfg.get('mode', 'hard')
    
    print(f"\n✓ Inpainting mode: {inpainting_mode}")
    
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
        
        # For hard inpainting, use ground truth latent from dataset
        x_context = None
        
        if inpainting_mode == "hard":
            with torch.no_grad():
                # Use pred_latent from dataset as ground truth context
                # Dataset already provides this at the correct latent resolution
                x_context = pred_latent.unsqueeze(0).to(device)  # [1, C, H, W]
                
                print(f"✓ Using prediction latent as context: {x_context.shape}")
                
                # Initialize: keep context outside mask, noise inside mask
                if mask_latent is not None:
                    x = mask_latent * x + (1 - mask_latent) * x_context
                    
                    print(f"✓ Hard inpainting: preserving context outside mask ({(1 - mask_latent.mean()):.1%} of latent)")
                    print(f"  Mask latent stats: mean={mask_latent.mean():.4f}, min={mask_latent.min():.4f}, max={mask_latent.max():.4f}")
                else:
                    print(f"⚠ Warning: No inpainting mask found, creating full mask (generate entire image)")
                    mask_latent = torch.ones(1, 1, latent_size, latent_size, device=device)
                    x = mask_latent * x + (1 - mask_latent) * x_context
        
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
            # if use_lst_guidance and lst_predictor is not None and lst_target is not None:
            #     pred_vae = vae_registry.get_vae(pred_group)
            #     noise_pred = apply_lst_guidance(
            #         x, i, model, scheduler, cond_input,
            #         lst_predictor, pred_vae, lst_target,
            #         semantic_channels, include_ndvi,
            #         guidance_scale=lst_guidance_scale,
            #         mask=mask_latent
            #     )
            
            # Denoise
            if inpainting_mode == "hard" and mask_latent is not None:
                x, x0 = scheduler.sample_prev_timestep_inpainting(
                    x, noise_pred, i, x_context, mask_latent
                )
            else:
                x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
        
        all_samples.append(x)
    
    # Stack samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Decode to semantic space using prediction VAE
    pred_vae = vae_registry.get_vae(pred_group)
    with torch.no_grad():
        semantic_samples = pred_vae.decode(all_samples)
    
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

    # Save visualization - each layer separately (like VAE training)
    for ch_idx, layer_name in enumerate(semantic_layers):
        if ch_idx < semantic_samples.shape[1]:
            ch = semantic_samples[:, ch_idx:ch_idx+1, :, :]
            
            if 'height' in layer_name:
                # Continuous height channel: normalize by max height
                ch_vis = torch.clamp(ch / 100.0, 0, 1)
            else:
                # Binary channels: already in [0, 1] after VAE decode
                ch_vis = torch.clamp(ch, 0, 1)
            
            # Overlay red mask border if mask is available
            if mask_latent is not None:
                # Upsample mask to match semantic_samples resolution
                mask_upsampled = F.interpolate(
                    mask_latent,
                    size=(semantic_samples.shape[2], semantic_samples.shape[3]),
                    mode='nearest'
                )
                
                # Convert grayscale to RGB for red border overlay
                ch_vis_rgb = ch_vis.repeat(1, 3, 1, 1)  # [B, 3, H, W]
                
                # Compute mask boundary (edge detection)
                mask_tensor = mask_upsampled.float()  # [1, 1, H, W]
                
                # Create erosion kernel (3x3 all ones)
                kernel = torch.ones(1, 1, 3, 3, device=ch_vis.device)
                
                # Erode the mask (shrink it inward)
                mask_eroded = F.conv2d(mask_tensor, kernel, padding=1)
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
            
            # Create grid for this layer
            grid = make_grid(ch_vis_final, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
            output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_{layer_name}.png')
            save_image(grid, output_path)
    
    # Save mask visualization if available
    if mask_latent is not None:
        # Upsample mask to match semantic_samples resolution for visualization
        mask_upsampled = F.interpolate(
            mask_latent,
            size=(semantic_samples.shape[2], semantic_samples.shape[3]),
            mode='nearest'
        )
        mask_vis = mask_upsampled.repeat(num_samples, 1, 1, 1)
        grid = make_grid(mask_vis, nrow=int(np.sqrt(num_samples)) + 1, padding=4, pad_value=1.0)
        output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}_inpainting_mask.png')
        save_image(grid, output_path)
        print(f"✓ Saved inpainting mask visualization")
    
    print(f"\n✓ Saved {len(semantic_layers)} layer visualizations to {out_dir}")    # Save individual samples as .pt files for Stage 2
    samples_dir = os.path.join(out_dir, f'{base_name}_idx{run_idx}_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    for idx in range(num_samples):
        sample_path = os.path.join(samples_dir, f'sample_{idx}.pt')
        torch.save({
            'semantic_tensor': semantic_samples[idx].cpu(),
            'semantic_layers': semantic_layers,
            'conditioning': {k: v[0].cpu() if isinstance(v, torch.Tensor) else v for k, v in cond_input.items()},
            'mask': mask_latent[0].cpu() if mask_latent is not None else None
        }, sample_path)
    
    print(f"✓ Saved {num_samples} samples to {samples_dir}")
    
    return semantic_samples


def infer(args, config):
    ###### setup config variables #######
    data_config = config['data_config']
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    vae_groups = config['vae_groups']
    diffusion_stages = config['diffusion_stages']
    
    # Determine mode from args (default to 'semantic')
    mode = getattr(args, 'mode', 'semantic')
    
    # Validate mode
    if mode not in diffusion_stages:
        raise ValueError(
            f"Mode '{mode}' not found in diffusion_stages. "
            f"Available: {list(diffusion_stages.keys())}"
        )
    
    # Get stage config
    stage_config = diffusion_stages[mode]
    pred_group = stage_config.get('prediction_group')
    
    if pred_group not in vae_groups:
        raise ValueError(
            f"Prediction group '{pred_group}' not found in VAE groups. "
            f"Available: {list(vae_groups.keys())}"
        )
    
    # Get VAE config for prediction group
    vae_config = vae_groups[pred_group]
    unet_config = stage_config.get('unet_config', {})
    
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
    
    # Initialize VAE Registry
    vae_registry = VAERegistry(vae_config, device)
    
    # Load VAE for prediction group
    data_dir = f"{big_data_storage_path}/results/{train_config['task_name']}"
    vae_checkpoint = vae_config.get('checkpoint_name', f'{pred_group}_vae_ckpt.pth')
    vae_path = os.path.join(data_dir, vae_checkpoint)
    
    vae = vae_registry.load_vae(
        group_name=pred_group,
        checkpoint_path=vae_path,
        autoencoder_config=vae_config,
    )
    
    if vae is None:
        print(f"✗ Failed to load {pred_group} VAE from {vae_path}")
        return
    
    # Load Diffusion Model
    model = Unet(
        im_channels=vae_config['z_channels'],
        model_config=unet_config
    ).to(device)
    model.eval()
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing for memory efficiency")
    
    # Get diffusion checkpoint from training config
    diffusion_train_config = train_config.get('diffusion_training', {}).get(mode, {})
    ldm_checkpoint = diffusion_train_config.get('checkpoint_name', f'{mode}_diffusion_ckpt.pth')
    ldm_path = os.path.join(data_dir, ldm_checkpoint)
    
    if os.path.exists(ldm_path):
        checkpoint = torch.load(ldm_path, map_location=device)
        
        # Handle both new format (dict with 'model_state_dict') and legacy format (direct state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✓ Loaded Semantic Diffusion Model from {ldm_path} (epoch {epoch})")
        else:
            # Legacy format
            model_state = checkpoint
            print(f"✓ Loaded Semantic Diffusion Model from {ldm_path}")
        
        model.load_state_dict(model_state)
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
        diffusion_model_config=unet_config,
        autoencoder_model_config=vae_config,
        diffusion_config=diffusion_config,
        dataset_config=dataset_config,
        stage_config=stage_config,
        big_data_storage_path=big_data_storage_path,
        vae_registry=vae_registry,
        vae_groups=vae_groups,
        pred_group=pred_group,
        mode=mode,
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
    parser.add_argument('--mode', type=str, default='semantic', help='Diffusion stage to sample (e.g., semantic, satellite)')
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

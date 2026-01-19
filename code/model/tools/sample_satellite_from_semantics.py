# Stage 2: Render satellite imagery from generated semantics
# Uses satellite diffusion model to generate realistic imagery conditioned on semantic layout

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob

# Data handling
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# Local libraries
from model.diffusion_blocks.unet_cond_base import Unet
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.vae_registry import VAERegistry
from model.utils.config_utils import compute_patch_and_latent_sizes, build_unet_condition_config
from model.utils.checkpoint import load_checkpoint
from model.utils.diffusion_utils import mask_conditioning_latents
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
    overwrite_samples=False
):
    """
    Render satellite imagery from semantic layouts using Stage 2 diffusion model.
    
    Follows same pattern as semantic sampling but:
    - Prediction group: 'satellite' (RGB imagery)
    - Conditioning: semantic latents + environmental latents + pixel-space mask
    - Environmental latents loaded from dataset using saved patch metadata
    - Applies mask to environmental conditioning (as per config)
    
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
        
    Returns:
        Generated satellite images [N, 3, H, W]
    """
    model.eval()
    
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
    
    # Get inpainting mode
    inpainting_mode = inpainting_cfg.get('mode', 'sdlike')
    cfg_config = inpainting_cfg.get('cfg', {})
    mask_groups = cfg_config.get('mask_groups', [])
    
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
    print(f"Mask groups: {mask_groups}")
    
    # Load prediction VAE (satellite)
    big_data_storage_path = dataset_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    task_name = train_config.get('task_name', 'urban_inpainting')
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    
    vae_checkpoint = pred_vae_config.get('checkpoint_name', f'{pred_group}_vae_ckpt.pth')
    vae_path = os.path.join(data_dir, vae_checkpoint)
    vae_registry.load_vae(
        group_name=pred_group,
        checkpoint_path=vae_path,
        autoencoder_config=pred_vae_config,
    )
    pred_vae = vae_registry.get_vae(pred_group)
    
    # Load conditioning VAEs (semantic, environmental)
    latent_cond_groups = conditioning_config.get('latent_space', [])
    for cond_spec in latent_cond_groups:
        cond_group = cond_spec['group']
        cond_vae_config = vae_groups[cond_group]
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
    
    ################# Rendering Loop ########################
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
        
        # Build conditioning dict (following diffusion training pattern)
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
        
        # Load RGB context from dataset for hard inpainting (if available)
        rgb_context_latent = None
        patch_index = sample_data.get('patch_index')
        
        if patch_index is not None and dataset is not None and inpainting_mode == 'hard':
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
                    # Already encoded latent
                    rgb_context_latent = pred_data.unsqueeze(0).to(device)
                    print(f"  ✓ Loaded RGB context latent from dataset: {rgb_context_latent.shape}")
                else:
                    # Full-res image - encode it
                    print(f"  ⚠ Encoding RGB context on-the-fly from shape {pred_data.shape}")
                    with torch.no_grad():
                        pred_image_batch = pred_data.unsqueeze(0).to(device)
                        rgb_context_latent, _, _ = pred_vae.encode(pred_image_batch)
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
                print(f"  ✓ Encoded semantic to latent: {semantic_latent.shape}")
                
            elif cond_group == 'environmental':
                # Get environmental conditioning from dataset using patch metadata
                patch_index = sample_data.get('patch_index')
                
                if patch_index is not None and dataset is not None:
                    # Load matching patch from dataset (already has encoded latents)
                    dataset_sample = dataset[patch_index]
                    
                    if isinstance(dataset_sample, tuple) and len(dataset_sample) == 2:
                        _, dataset_cond = dataset_sample
                    else:
                        dataset_cond = {}
                    
                    # Extract environmental latents (already encoded by dataset)
                    if cond_group in dataset_cond:
                        env_latent = dataset_cond[cond_group].unsqueeze(0).to(device)
                        cond_input[cond_group] = env_latent
                        print(f"  ✓ Loaded environmental conditioning from dataset patch {patch_index}")
                    else:
                        print(f"  ⚠ Environmental group '{cond_group}' not found in dataset, using zeros")
                        z_channels = vae_groups[cond_group]['z_channels']
                        env_latent = torch.zeros(1, z_channels, latent_size, latent_size, device=device)
                        cond_input[cond_group] = env_latent
                else:
                    # Fallback to zeros if no patch metadata
                    print(f"  ⚠ No patch_index in sample metadata, using zero environmental conditioning")
                    z_channels = vae_groups[cond_group]['z_channels']
                    env_latent = torch.zeros(1, z_channels, latent_size, latent_size, device=device)
                    cond_input[cond_group] = env_latent
        
        # 3. Apply mask to specified conditioning groups (e.g., environmental)
        if mask_groups and 'image' in cond_input and 'pixel_space_names' in cond_input['meta'][0]:
            pixel_names = cond_input['meta'][0]['pixel_space_names']
            if 'inpainting_mask' in pixel_names:
                mask_idx = pixel_names.index('inpainting_mask')
                mask_latent = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                
                # Mask conditioning latents (zeros out masked region)
                cond_input = mask_conditioning_latents(cond_input, mask_latent, mask_groups)
                print(f"  ✓ Masked conditioning groups: {mask_groups}")
        
        # Create unconditional input for CFG
        uncond_input = {}
        for key in cond_input:
            if key == 'meta':
                uncond_input[key] = cond_input[key]
            elif key == 'image':
                # Zero out pixel-space conditioning
                uncond_input[key] = torch.zeros_like(cond_input[key])
            else:
                # Zero out latent-space conditioning groups
                uncond_input[key] = torch.zeros_like(cond_input[key])
        
        # Initialize latent for satellite generation
        x = torch.randn(1, pred_vae_config['z_channels'], latent_size, latent_size, device=device)
        
        # For hard inpainting, initialize with RGB context from dataset
        if inpainting_mode == "hard":
            # Get mask at latent resolution
            mask_latent = F.interpolate(mask.float(), size=(latent_size, latent_size), mode='nearest')
            
            if rgb_context_latent is not None:
                # Use actual RGB context from dataset (like semantic sampling does)
                # Keep context outside mask, noise inside mask
                x = mask_latent * x + (1 - mask_latent) * rgb_context_latent
                print(f"  ✓ Initialized with RGB context from dataset")
            else:
                # Fallback: use zeros if no context available
                print(f"  ⚠ No RGB context available, using zeros outside mask")
                x = mask_latent * x + (1 - mask_latent) * torch.zeros_like(x)
        
        # Sampling loop
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
            
                # Denoise step
                if inpainting_mode == "hard":
                    # Hard inpainting: preserve context outside mask
                    # Note: For satellite, we don't have original latent, so just use standard step
                    x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
                else:
                    # SD-like: standard denoising
                    x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
        
        all_renders.append(x)
        print(f"  ✓ Rendered sample {sample_idx + 1}")
        
        # Free GPU memory between samples
        torch.cuda.empty_cache()
    
    # Stack renders
    all_renders = torch.cat(all_renders, dim=0)  # [N, C_latent, H_latent, W_latent]
    
    # Decode to RGB using satellite VAE
    print("\nDecoding satellite latents to RGB...")
    with torch.no_grad():
        rgb_renders = pred_vae.decode(all_renders)
    
    # Normalize to [0, 1]
    # Check if VAE uses tanh activation
    if pred_vae_config.get('tanh_activation', False):
        # VAE output is in [-1, 1]
        rgb_renders = torch.clamp(rgb_renders, -1., 1.)
        rgb_renders = (rgb_renders + 1) / 2
    else:
        # VAE output is in [0, 1]
        rgb_renders = torch.clamp(rgb_renders, 0., 1.)
    
    # Save results
    out_dir = f"{big_data_storage_path}/results/{task_name}/satellite_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get run index
    base_name = f'satellite_cfg{guidance_scale}'
    run_idx = get_next_run_idx(out_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*60}")
    
    # Save grid
    grid = make_grid(rgb_renders, nrow=int(np.sqrt(len(semantic_samples))) + 1, 
                    padding=4, pad_value=1.0)
    output_path = os.path.join(out_dir, f'{base_name}_idx{run_idx}.png')
    save_image(grid, output_path)
    print(f"\n✓ Saved grid visualization to {output_path}")
    
    # Save individual renders
    samples_dir = os.path.join(out_dir, f'{base_name}_idx{run_idx}_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    for idx in range(len(semantic_samples)):
        sample_path = os.path.join(samples_dir, f'sample_{idx}.png')
        save_image(rgb_renders[idx], sample_path)
        
        # Also save as .pt for further processing
        sample_pt_path = os.path.join(samples_dir, f'sample_{idx}.pt')
        torch.save({
            'rgb_tensor': rgb_renders[idx].cpu(),
            'semantic_source': semantic_samples[idx].get('semantic_channels', []),
        }, sample_pt_path)
    
    print(f"✓ Saved {len(semantic_samples)} satellite renders to {samples_dir}")
    
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
        model_config=unet_config_with_cond,
        mode=mode
    ).to(device)
    model.eval()
    
    # Load checkpoint
    task_name = train_config.get('task_name', 'urban_inpainting')
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    
    diffusion_train_config = train_config.get('diffusion_training', {}).get(mode, {})
    ldm_checkpoint = diffusion_train_config.get('checkpoint_name', f'{mode}_diffusion_ckpt.pth')
    ldm_path = os.path.join(data_dir, ldm_checkpoint)
    
    if os.path.exists(ldm_path):
        load_checkpoint(
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
    
    ########## Load Dataset for Environmental Conditioning #############
    print("\n" + "="*60)
    print("Loading Dataset for Environmental Conditioning")
    print("="*60)
    
    # Check if we need dataset (has environmental conditioning)
    stage_config = diffusion_stages[mode]
    conditioning_config = stage_config.get('conditioning', {})
    latent_cond_groups = conditioning_config.get('latent_space', [])
    needs_environmental = any(spec['group'] == 'environmental' for spec in latent_cond_groups)
    
    dataset = None
    if needs_environmental:
        # Check for cached patches
        cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
        use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
        
        try:
            dataset = UrbanInpaintingDataset(
                split='train',
                mode=f'diffusion:{mode}',
                use_cached_patches=use_cached_patches,
                cache_dir=cache_dir
            )
            print(f"✓ Loaded dataset with {len(dataset)} patches for environmental conditioning")
        except Exception as e:
            print(f"⚠ Failed to load dataset: {e}")
            print(f"  Will use zero environmental conditioning")
            dataset = None
    else:
        print("✓ No environmental conditioning needed, skipping dataset load")
    
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
        overwrite_samples=args.overwrite_samples
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

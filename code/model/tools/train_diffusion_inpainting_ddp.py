# Training script for latent diffusion inpainting with DDP

###### import libraries ######
# system libraries
import os
import time
import yaml
import random
import numpy as np
from tqdm import tqdm
import argparse

# data science libraries
import torch
import torch.nn.functional as torchF
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

# local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.unet_cond_base import Unet
from model.utils.vae_registry import VAERegistry
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.scheduler.ddim_scheduler import DDIMScheduler
from model.utils.data_utils import collate_fn
from model.utils.diffusion_utils import (
    apply_classifier_free_guidance_dropout,
    apply_seam_mode,
    compute_boundary_aware_loss
)
from model.utils.loss_weighting import compute_loss_weights
from model.utils.scalar_controls import parse_scalar_controls_config
from model.utils.samples import save_layerwise_samples, save_rgb_composite
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.config_utils import build_unet_condition_config
from model.utils.layer_config import count_layer_channels, get_layer_info
from model.utils.checkpoint import load_checkpoint
from helpers.load_configs import load_configs, add_config_arguments

# Load CUDA
load_cuda()


def train(mode: str = 'semantic', load_checkpoint_path: str = None):
    """
    Generic diffusion training function supporting any diffusion stage defined in config.
    
    Args:
        mode: Diffusion stage name (e.g., 'semantic', 'satellite')
              Must match a key in config['diffusion_stages']
        load_checkpoint_path: Path to checkpoint to resume from (None = train from scratch)
    """
    # Record training start time
    training_start_time = time.time()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    ###### setup config variables #######
    config = load_configs()
    data_config = config['dataset_params']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Diffusion Training: {mode.upper()}")
        print(f"{'='*60}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        print(f"✓ Diffusion Stage: {mode}")
        print(f"\n{'='*50}")
        print("Configuration")
        print(f"{'='*50}")
        print(yaml.dump(config, default_flow_style=False))
    
    diffusion_config = config['diffusion_params']
    train_config_global = config['train_params']
    
    # Validate diffusion stage exists
    diffusion_stages = config.get('diffusion_stages', {})
    if mode not in diffusion_stages:
        raise ValueError(
            f"Diffusion stage '{mode}' not found in config. "
            f"Available stages: {list(diffusion_stages.keys())}"
        )
    
    # Get diffusion stage configuration
    stage_config = diffusion_stages[mode]
    prediction_group = stage_config.get('prediction_group')
    unet_config = stage_config.get('unet_config', {})
    conditioning_config = stage_config.get('conditioning', {})
    inpainting_config = stage_config.get('inpainting', {})
    validate_enabled = stage_config.get('validate', True)  # Enable validation by default
    
    if not prediction_group:
        raise ValueError(f"Diffusion stage '{mode}' has no prediction_group defined")
    
    # Get prediction VAE group configuration
    vae_groups = config.get('vae_groups', {})
    if prediction_group not in vae_groups:
        raise ValueError(
            f"Prediction group '{prediction_group}' not found in vae_groups. "
            f"Available groups: {list(vae_groups.keys())}"
        )
    
    prediction_vae_config = vae_groups[prediction_group]
    layers_registry = config.get('layers', {})
    
    # Parse prediction layers and compute channels
    prediction_layers = prediction_vae_config.get('layers', [])
    if not prediction_layers:
        raise ValueError(f"Prediction group '{prediction_group}' has no layers defined")
    
    num_prediction_channels = 0
    layer_names = []
    for layer_name in prediction_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_prediction_channels += num_channels
        layer_names.append(layer_name)
    
    # Get VAE architecture config
    vae_arch_config = {
        'z_channels': prediction_vae_config.get('z_channels', 4),
        'down_channels': prediction_vae_config.get('down_channels', [32, 64, 128, 128]),
        'mid_channels': prediction_vae_config.get('mid_channels', [128, 128]),
        'down_sample': prediction_vae_config.get('down_sample', [True, True, True]),
        'attn_down': prediction_vae_config.get('attn_down', [False, False, False]),
        'norm_channels': prediction_vae_config.get('norm_channels', 32),
        'num_heads': prediction_vae_config.get('num_heads', 2),
        'num_down_layers': prediction_vae_config.get('num_down_layers', 2),
        'num_mid_layers': prediction_vae_config.get('num_mid_layers', 2),
        'num_up_layers': prediction_vae_config.get('num_up_layers', 2),
    }
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Diffusion Stage: {mode}")
        print(f"{'='*60}")
        print(f"✓ Prediction group: {prediction_group}")
        print(f"✓ Prediction layers: {layer_names}")
        print(f"✓ Total prediction channels: {num_prediction_channels}")
        print(f"✓ Latent channels: {vae_arch_config['z_channels']}")
        print(f"{'='*60}\n")
    
    # Get training configuration
    diffusion_training_config = train_config_global.get('diffusion_training', {})
    train_config = diffusion_training_config.get(mode, {})
    
    # Path to prediction VAE latents
    latent_dir_name = prediction_vae_config.get('latents_dir', f'{prediction_group}_latents')
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    latent_path = f'{out_dir}/{latent_dir_name}'
    use_existing_latents = os.path.exists(latent_path) and len(os.listdir(latent_path)) > 0
    
    cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    # checkpoint path
    if load_checkpoint_path is not None:
        load_checkpoint_path = os.path.join(out_dir, load_checkpoint_path)
    
    # Create output directory
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    samples_dir_name = f'{mode}_diffusion_samples'
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, samples_dir_name), exist_ok=True)
    
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    
    ########## Create the noise scheduler #############
    # Training always uses DDPM (all timesteps for proper diffusion training)
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    # Create separate DDIM scheduler for fast validation sampling
    val_scheduler = DDIMScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        ddim_steps=train_config.get('val_sample_steps', 50),
        ddim_eta=0.0  # Deterministic for reproducible validation
    )
    
    if is_main:
        print(f"\n✓ Created DDPM training scheduler ({scheduler.num_timesteps} timesteps)")
        print(f"✓ Created DDIM validation scheduler ({val_scheduler.ddim_steps} steps)")
    
    ########## Load Dataset #############
    if is_main:
        print("\n" + "="*50)
        print(f"Loading Urban Dataset for {mode.upper()} Diffusion Training")
        print("="*50)
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=f'diffusion:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Using existing latents: {use_existing_latents}")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Prediction layers: {layer_names}")
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(
        urban_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    batch_size = train_config.get('batch_size', 4)
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    # Load validation dataset for proper validation sampling (only if validation enabled)
    val_loader = None
    if validate_enabled:
        try:
            val_dataset = UrbanInpaintingDataset(
                split='val',
                mode=f'diffusion:{mode}',
                use_cached_patches=use_cached_patches,
                cache_dir=cache_dir
            )
            
            if len(val_dataset) > 0:
                # No distributed sampling for validation (only main process samples)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_fn
                )
                
                if is_main:
                    print(f"✓ Loaded {len(val_dataset)} validation patches")
            else:
                if is_main:
                    print("⚠ Warning: Validation split is empty, will use training split for monitoring")
        except Exception as e:
            if is_main:
                print(f"⚠ Warning: Could not load validation split ({e}), will use training split for monitoring")
    else:
        if is_main:
            print("✓ Validation disabled (validate=False in config)")
    
    ########## Create Model #############
    if is_main:
        print("\n" + "="*50)
        print("Initializing Models")
        print("="*50)
    
    # Build condition_config for U-Net from stage conditioning configuration
    condition_config = build_unet_condition_config(stage_config, vae_groups, global_config=config)
    
    # Add condition_config to unet_config
    unet_config_with_cond = unet_config.copy()
    unet_config_with_cond['condition_config'] = condition_config
    
    # Instantiate the U-Net model for diffusion
    model = Unet(
        im_channels=vae_arch_config['z_channels'],
        model_config=unet_config_with_cond,
        mode=mode
    ).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if is_main:
            print("✓ Wrapped model in DistributedDataParallel")
    
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        print(f"✓ Created U-Net with {sum(p.numel() for p in model_unwrapped.parameters())/1e6:.2f}M parameters")
    
    ########## Setup EMA (Exponential Moving Average) #############
    ema_model = None
    if train_config.get('use_ema', False):
        from model.utils.ema import ExponentialMovingAverage
        
        ema_decay = train_config.get('ema_decay', 0.9999)
        model_for_ema = model.module if hasattr(model, 'module') else model
        ema_model = ExponentialMovingAverage(model_for_ema, decay=ema_decay, device=device)
        
        if is_main:
            print(f"\n✓ EMA enabled with decay={ema_decay}")
    
    # Load prediction VAE if no existing latents
    vae = None
    vae_registry = None
    if not use_existing_latents or validate_enabled:
        if is_main:
            print(f"\nLoading VAE models...")
        
        # Use VAERegistry for cleaner management
        vae_registry = VAERegistry(config, device)
        
        # Load prediction VAE
        if is_main:
            print(f"  - {prediction_group.upper()} (prediction group)")
        vae_registry.load_vae(
            group_name=prediction_group,
            checkpoint_path=os.path.join(out_dir, prediction_vae_config.get('checkpoint_name', f'{prediction_group}_vae_ckpt.pth')),
            autoencoder_config=prediction_vae_config
        )
        vae = vae_registry.get_vae(prediction_group)
        vae_registry.freeze(prediction_group)
        vae.eval()
        
        # Load conditioning VAEs for latent-space conditioning groups
        latent_space_specs = conditioning_config.get('latent_space', [])
        for spec in latent_space_specs:
            cond_group = spec.get('group')
            if cond_group and cond_group in vae_groups:
                if is_main:
                    print(f"  - {cond_group.upper()} (conditioning group)")
                cond_vae_config = vae_groups[cond_group]
                vae_registry.load_vae(
                    group_name=cond_group,
                    checkpoint_path=os.path.join(out_dir, cond_vae_config.get('checkpoint_name', f'{cond_group}_vae_ckpt.pth')),
                    autoencoder_config=cond_vae_config
                )
                vae_registry.freeze(cond_group)
        
        if is_main:
            print(f"✓ Loaded {len(vae_registry.vaes)} VAE model(s)")
    
    ########## Training Setup #############
    num_epochs = train_config.get('epochs', 300)
    
    # Scale learning rate with world size
    base_lr = train_config.get('lr', 0.00001)
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"\n✓ Scaled learning rate: {base_lr} -> {adjusted_lr} (x{world_size})")
    
    optimizer = Adam(model.parameters(), lr=adjusted_lr)
    
    ########## Setup LR Scheduler #############
    lr_scheduler = None
    if train_config.get('lr_scheduler', 'constant') != 'constant' or train_config.get('lr_warmup_steps', 0) > 0:
        from model.scheduler.lr_scheduler import get_lr_scheduler
        
        lr_scheduler_config = {
            'lr_scheduler': train_config.get('lr_scheduler', 'constant'),
            'lr_warmup_steps': train_config.get('lr_warmup_steps', 0),
            'num_epochs': num_epochs,
            'steps_per_epoch': len(data_loader)
        }
        lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_config)
        
        if is_main:
            print(f"\n✓ LR Scheduler configured")
    
    # Load checkpoint if specified
    start_epoch = 0
    checkpoint_dict = None
    if load_checkpoint_path:
        # Load model and optimizer
        start_epoch, checkpoint_dict = load_checkpoint(
            checkpoint_path=load_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            is_main=is_main
        )
        
        # Load EMA state if available
        if ema_model is not None and checkpoint_dict is not None and 'ema_state_dict' in checkpoint_dict:
            ema_model.load_state_dict(checkpoint_dict['ema_state_dict'])
            if is_main:
                print("✓ Loaded EMA state from checkpoint")
        
        # Load LR scheduler state if available
        if lr_scheduler is not None and checkpoint_dict is not None and 'lr_scheduler_state_dict' in checkpoint_dict:
            lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
            if is_main:
                print("✓ Loaded LR scheduler state from checkpoint")
    
    # Inpainting configuration
    inpainting_mode = inpainting_config.get('mode', 'hard')         # "hard" | "sdlike"
    loss_type = inpainting_config.get('loss', 'masked')  # "masked" | "weighted"
    mask_loss_weight = inpainting_config.get('mask_loss_weight', 8.0)
    
    if inpainting_mode == "hard":
        outside_weight = 0.0
        loss_type = "masked"  # Force masked loss for hard mode
        
        # Warn if config tries to set outside_weight > 0 in hard mode
        config_outside = inpainting_config.get('outside_weight', 0.0)
        if config_outside > 0.0:
            if is_main:
                print(f"⚠ WARNING: Hard mode with outside_weight={config_outside} detected.")
                print(f"⚠ Overriding to outside_weight=0.0 (hard mode: outside region not noised)")
    elif inpainting_mode == "sdlike":
        outside_weight = inpainting_config.get('outside_weight', 1.0)
    else:
        outside_weight = inpainting_config.get('outside_weight', 1.0)
    
    # Classifier-free guidance dropout
    cfg_config = inpainting_config.get('cfg', {})
    cond_drop_prob = cfg_config.get('drop_prob', 0.1)
    drop_groups = cfg_config.get('drop_groups', [])  # Which latent groups to drop
    
    keep_mask = True  # Always True for inpainting
    
    # Build scalar unconditional values dict for all enabled scalar controls
    scalar_uncond = {}
    stage_scalar_controls = stage_config.get('scalar_controls', None)
    
    scalar_controls_enabled = (
        isinstance(stage_scalar_controls, list) and len(stage_scalar_controls) > 0
    ) or (
        isinstance(stage_scalar_controls, bool) and stage_scalar_controls
    )
    
    if scalar_controls_enabled:
        # Parse enabled scalar controls for this stage
        stage_control_names = stage_scalar_controls if isinstance(stage_scalar_controls, list) else None
        control_specs = parse_scalar_controls_config(config, stage_control_names=stage_control_names)
        
        for spec in control_specs:
            scalar_keys = spec['keys']
            training_cfg = spec.get('training', {})
            uncond_value = training_cfg.get('unconditional_value', 0.0)
            
            # Add unconditional value for each scalar key
            for key in scalar_keys:
                scalar_uncond[key] = uncond_value
        
        if is_main and len(scalar_uncond) > 0:
            print(f"✓ Scalar controls unconditional values: {scalar_uncond}")
    
    # Seam improvement configuration
    seam_mode = inpainting_config.get('seam', None)
    seam_config = inpainting_config.get('seam_config', {})
    use_boundary_ring = (seam_mode == 'dilate')
    ring_width_px = seam_config.get('ring_width_px', 1)
    ring_weight = seam_config.get('ring_weight', 2.0)
    
    # Image save frequency
    img_save_steps = train_config_global.get('img_save_steps', 1000)
    
    # Gradient accumulation configuration
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    effective_batch_size = batch_size * world_size * gradient_accumulation_steps
    
    # Timestep loss weighting configuration
    timestep_loss_type = train_config.get('loss_type', 'simple')  # 'simple' | 'snr' | 'min_snr' | 'v_loss'
    min_snr_gamma = train_config.get('min_snr_gamma', 5.0)
    use_loss_weighting = (timestep_loss_type != 'simple')
    
    # Validation sampling configuration
    val_sample_epochs = train_config.get('val_sample_epochs', 10)  # Sample every N epochs (0 = disabled)
    val_num_samples = train_config.get('val_num_samples', 4)  # Number of validation samples to generate
    val_sample_steps = train_config.get('val_sample_steps', 50)  # DDIM steps for faster validation
    val_guidance_scale = train_config.get('val_guidance_scale', None)  # CFG scale for validation (None = no CFG)
    
    # Create validation samples directory
    validation_dir_name = f'{mode}_diffusion_validation'
    if is_main and validate_enabled and val_sample_epochs > 0:
        os.makedirs(os.path.join(out_dir, validation_dir_name), exist_ok=True)
    
    if is_main:
        print(f"\n{'='*50}")
        print("Training Configuration")
        print(f"{'='*50}")
        print(f"✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {adjusted_lr}")
        print(f"✓ Batch size per GPU: {batch_size}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"✓ Effective batch size: {effective_batch_size} ({batch_size} x {world_size} x {gradient_accumulation_steps})")
        print(f"✓ Timestep loss weighting: {timestep_loss_type}")
        if timestep_loss_type == 'min_snr':
            print(f"  - Min-SNR gamma: {min_snr_gamma}")
        print(f"✓ Inpainting mode: {inpainting_mode}")
        print(f"✓ Loss type: {loss_type}")
        print(f"✓ Mask loss weight: {mask_loss_weight}")
        print(f"✓ Outside weight: {outside_weight}")
        print(f"✓ Seam mode: {seam_mode if seam_mode else 'None'}")
        if use_boundary_ring:
            print(f"  - Ring width: {ring_width_px}px")
            print(f"  - Ring weight: {ring_weight}")
        print(f"✓ CFG dropout prob: {cond_drop_prob}")
        print(f"✓ CFG drop groups: {drop_groups}")
        print(f"✓ CFG keep mask: {keep_mask} (inpainting_mask always preserved)")
        if validate_enabled and val_sample_epochs > 0:
            print(f"✓ Validation sampling: every {val_sample_epochs} epochs")
            print(f"  - Num samples: {val_num_samples}")
            print(f"  - Sample steps: {val_sample_steps}")
            if val_guidance_scale is not None:
                print(f"  - Guidance scale: {val_guidance_scale}")
        elif not validate_enabled:
            print(f"✓ Validation: disabled")
        else:
            print(f"✓ Validation: disabled (val_sample_epochs=0)")
        print(f"{'='*50}\n")
    
    ########## Training Loop #############
    if is_main:
        print("\n" + "="*50)
        print(f"Starting Training")
        print("="*50)
    
    global_step = 0
    accumulation_counter = 0  # Track batches for gradient accumulation
    
    for epoch_idx in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        losses = []
        
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            # Only zero gradients at start of accumulation cycle
            if accumulation_counter == 0:
                optimizer.zero_grad()
            
            # Unpack data
            if len(data) == 2:
                prediction_data, cond_input = data
            else:
                prediction_data = data
                cond_input = {}
            
            prediction_data = prediction_data.float().to(device)
            
            # Move ALL conditioning tensors to device (pixel-space and latent-space groups)
            if 'image' in cond_input:
                cond_input['image'] = cond_input['image'].float().to(device)
            
            # Move latent-space conditioning groups to device (use metadata if available)
            if 'meta' in cond_input and 'latent_group_names' in cond_input['meta'][0]:
                latent_group_keys = cond_input['meta'][0]['latent_group_names']
            else:
                # Fallback: infer from keys (excludes image, meta, and scalar controls)
                latent_group_keys = [k for k in cond_input.keys() if k not in ['image', 'meta'] and isinstance(cond_input[k], torch.Tensor) and cond_input[k].ndim > 2]
            
            for group_key in latent_group_keys:
                cond_input[group_key] = cond_input[group_key].float().to(device)
            
            # Encode prediction to latent space
            if use_existing_latents:
                # Precomputed latents
                im_latent = prediction_data
            else:
                # Encode on-the-fly
                with torch.no_grad():
                    im_latent, _, _ = vae.encode(prediction_data)
            
            # Extract inpainting mask from conditioning (already in latent space)
            mask_latent = None
            if 'image' in cond_input and 'meta' in cond_input:
                # Access pixel_space_names from metadata (first item, same across batch)
                pixel_space_names = cond_input['meta'][0].get('pixel_space_names', [])
                
                if pixel_space_names:
                    try:
                        mask_idx = pixel_space_names.index('inpainting_mask')
                        mask_latent = cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                    except (ValueError, IndexError):
                        pass
            
            # Fallback: create full mask if not provided
            if mask_latent is None:
                mask_latent = torch.ones_like(im_latent[:, :1, :, :])
            
            # Sample timestep
            t = torch.randint(0, scheduler.num_timesteps, (im_latent.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(im_latent)
            
            # Apply noise according to inpainting mode
            if inpainting_mode == "hard":
                # Hard inpainting: only add noise inside mask
                noisy_im = im_latent.clone()
                noisy_region = scheduler.add_noise(im_latent, noise, t)
                noisy_im = mask_latent * noisy_region + (1 - mask_latent) * im_latent
            else:
                # SD-like: add noise everywhere
                noisy_im = scheduler.add_noise(im_latent, noise, t)
            
            # Apply classifier-free guidance dropout
            if cond_drop_prob > 0:
                cond_input_dropped = apply_classifier_free_guidance_dropout(
                    cond_input,
                    drop_prob=cond_drop_prob,
                    drop_groups=drop_groups,
                    keep_mask=keep_mask,  # Always True - preserves inpainting_mask
                    scalar_uncond=scalar_uncond  # Dict of unconditional values for all scalars
                )
            else:
                cond_input_dropped = cond_input
            
            # First batch validation
            if is_main and global_step == 0:
                print(f"\n{'='*50}")
                print(f"First Batch Validation on rank {rank}:")
                print(f"{'='*50}")
                print(f"Prediction latent shape: {im_latent.shape}")
                print(f"Mask latent shape: {mask_latent.shape}")
                print(f"Mask stats: min={mask_latent.min().item():.4f}, max={mask_latent.max().item():.4f}, mean={mask_latent.mean().item():.4f}")
                
                # Pixel-space conditioning
                if 'image' in cond_input:
                    print(f"\nPixel-space conditioning:")
                    print(f"  Shape: {cond_input['image'].shape}")
                    print(f"  Device: {cond_input['image'].device}")
                    print(f"  Dtype: {cond_input['image'].dtype}")
                    # Access pixel_space_names from metadata
                    pixel_space_names = cond_input['meta'][0].get('pixel_space_names', [])
                    if pixel_space_names:
                        print(f"  Channels: {pixel_space_names}")
                
                # Latent-space conditioning groups
                latent_cond_groups = [k for k in cond_input.keys() if k not in ['image', 'meta']]
                if latent_cond_groups:
                    print(f"\nLatent-space conditioning groups: {latent_cond_groups}")
                    for group_name in latent_cond_groups:
                        group_tensor = cond_input[group_name]
                        print(f"  {group_name}:")
                        print(f"    Shape: {group_tensor.shape}")
                        print(f"    Device: {group_tensor.device}")
                        print(f"    Dtype: {group_tensor.dtype}")
                        print(f"    Stats: min={group_tensor.min().item():.4f}, max={group_tensor.max().item():.4f}, mean={group_tensor.mean().item():.4f}")
                
                # CFG dropout config
                print(f"\nCFG Dropout:")
                print(f"  Drop probability: {cond_drop_prob}")
                print(f"  Drop groups: {drop_groups}")
                
                print(f"{'='*50}\n")
            
            # Predict noise (use dropped conditioning for CFG training)
            noise_pred = model(noisy_im, t, cond_input=cond_input_dropped)
            
            # Compute per-sample loss [B] with spatial weighting
            loss_per_sample = compute_boundary_aware_loss(
                noise_pred, noise, mask_latent,
                loss_type, mask_loss_weight, outside_weight,
                use_boundary_ring=use_boundary_ring,
                ring_width_px=ring_width_px,
                ring_weight=ring_weight,
                reduction='batch'  # Return per-sample loss [B]
            )
            
            # Apply timestep-dependent loss weighting (Min-SNR, SNR, etc.)
            if use_loss_weighting:
                loss_weights = compute_loss_weights(
                    t, scheduler, timestep_loss_type, min_snr_gamma
                )  # Returns [B] weights
                loss_per_sample = loss_per_sample * loss_weights
            
            # Final reduction to scalar
            loss = loss_per_sample.mean()
            
            # Scale loss by accumulation steps to maintain same average gradient
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Track unscaled loss for logging
            losses.append(loss.item())
            accumulation_counter += 1
            
            # Only update optimizer/scheduler/EMA after accumulating enough gradients
            should_step = (accumulation_counter >= gradient_accumulation_steps)
            
            if should_step:
                optimizer.step()
                
                # Update LR scheduler (step per optimizer update)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # Update EMA (after each optimizer step)
                if ema_model is not None:
                    model_for_ema = model.module if hasattr(model, 'module') else model
                    ema_model.update(model_for_ema)
                
                # Reset accumulation counter
                accumulation_counter = 0
                global_step += 1
            
            # Update progress bar
            if is_main:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'accum': f'{accumulation_counter}/{gradient_accumulation_steps}'
                })
            
            # Save sample predictions periodically
            if is_main and global_step % img_save_steps == 0:
                with torch.no_grad():
                    model.eval()
                    
                    # Generate a few samples
                    num_samples = min(4, im_latent.shape[0])
                    
                    # Start from pure noise in masked region
                    x_sample = im_latent[:num_samples].clone()
                    
                    # Fixed noise_context for hard mode
                    sample_noise_context = None
                    
                    if inpainting_mode == "hard":  # FIX: Use inpainting_mode variable, not stage name
                        x_sample = mask_latent[:num_samples] * torch.randn_like(x_sample) + (1 - mask_latent[:num_samples]) * x_sample
                        # FIX: Create fixed noise_context for temporal consistency
                        sample_noise_context = torch.randn_like(x_sample)
                    else:
                        x_sample = torch.randn_like(x_sample)
                    
                    # Quick sampling (DDIM-style with fewer steps for speed)
                    # Use more steps for preview to get cleaner results (100 is good balance)
                    sample_steps = min(100, scheduler.num_timesteps)
                    
                    # Create timestep schedule: evenly spaced from T to 0
                    timesteps = np.linspace(scheduler.num_timesteps - 1, 0, sample_steps).astype(int)
                    
                    # Prepare conditioning for sampling (slice all keys to num_samples)
                    sample_cond = {}
                    for key in cond_input:
                        sample_cond[key] = cond_input[key][:num_samples]
                    
                    for t_idx in range(len(timesteps)):
                        t = timesteps[t_idx]
                        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                        
                        # Predict noise
                        noise_pred = model(x_sample, t_tensor, cond_input=sample_cond)
                        
                        if inpainting_mode == "hard":
                            # Use inpainting scheduler with fixed noise_context
                            x_sample, _ = scheduler.sample_prev_timestep_inpainting(
                                x_sample, noise_pred, t,
                                im_latent[:num_samples],
                                mask_latent[:num_samples],
                                noise_context=sample_noise_context
                            )
                        else:
                            x_sample, _ = scheduler.sample_prev_timestep(x_sample, noise_pred, t)
                    
                    # Decode to pixel space
                    if vae is not None:
                        sample_decoded = vae.decode(x_sample)
                    else:
                        sample_decoded = x_sample
                    
                    # Save layerwise visualizations using unified utility
                    save_layerwise_samples(
                        tensor=sample_decoded,
                        layer_names=prediction_layers,
                        layers_registry=layers_registry,
                        save_dir=os.path.join(out_dir, samples_dir_name),
                        filename_prefix=f'sample_step_{global_step}',
                        n_samples=num_samples,
                        is_reconstruction=True,  # VAE decoding, may have different scale
                        use_colormaps=True
                    )
                    
                    # Also save RGB composite if available
                    if 'rgb' in [l.lower() for l in prediction_layers]:
                        rgb_save_path = os.path.join(out_dir, samples_dir_name, f'sample_step_{global_step}_RGB_composite.png')
                        save_rgb_composite(
                            tensor=sample_decoded,
                            layer_names=prediction_layers,
                            save_path=rgb_save_path,
                            n_samples=num_samples,
                            normalize_per_channel=True
                        )
                    
                    model.train()
        
        # Synchronize epoch metrics
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary
        if is_main:
            epoch_loss = np.mean(losses)
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
        
        # Validation sampling (use EMA weights if available)
        if is_main and validate_enabled and val_sample_epochs > 0 and (epoch_idx + 1) % val_sample_epochs == 0:
            print(f"\n{'='*50}")
            print(f"Generating Validation Samples (Epoch {epoch_idx + 1})")
            print(f"{'='*50}")
            
            with torch.no_grad():
                # Use EMA weights for validation if available
                if ema_model is not None:
                    model_for_ema = model.module if hasattr(model, 'module') else model
                    ema_model.store(model_for_ema)  # Store current weights
                    ema_model.copy_to(model_for_ema)  # Copy EMA weights to model
                    print("✓ Using EMA weights for validation sampling")
                
                model.eval()
                
                # Get validation batch from proper validation split (or training if unavailable)
                if val_loader is not None:
                    val_data = next(iter(val_loader))
                    print("✓ Using validation split")
                else:
                    val_data = next(iter(data_loader))
                    print("⚠ Using training split (validation unavailable)")
                if len(val_data) == 2:
                    val_prediction_data, val_cond_input = val_data
                else:
                    val_prediction_data = val_data
                    val_cond_input = {}
                
                val_prediction_data = val_prediction_data[:val_num_samples].float().to(device)
                
                # Move conditioning to device
                if 'image' in val_cond_input:
                    val_cond_input['image'] = val_cond_input['image'][:val_num_samples].float().to(device)
                
                # Slice metadata list (one dict per sample)
                if 'meta' in val_cond_input:
                    val_cond_input['meta'] = val_cond_input['meta'][:val_num_samples]
                
                # Encode conditioning groups that need encoding (*_image keys)
                # This happens when validation latents don't exist - dataset provides full-res images
                if vae_registry is not None:
                    groups_to_encode = [k for k in val_cond_input.keys() if k.endswith('_image')]
                    if groups_to_encode:
                        print(f"⚠ Encoding conditioning groups on-the-fly (validation latents missing): {groups_to_encode}")
                        for group_key in groups_to_encode:
                            group_name = group_key.replace('_image', '')
                            group_image = val_cond_input.pop(group_key)[:val_num_samples].float().to(device)
                            
                            # Encode through VAE
                            vae_model = vae_registry.get_vae(group_name)
                            if vae_model is not None:
                                with torch.no_grad():
                                    group_latent, _, _ = vae_model.encode(group_image)
                                val_cond_input[group_name] = group_latent
                            else:
                                raise ValueError(f"VAE for group '{group_name}' not found in registry")
                
                # Slice and move latent-space conditioning groups (use metadata if available)
                if 'meta' in val_cond_input and 'latent_group_names' in val_cond_input['meta'][0]:
                    val_latent_group_keys = val_cond_input['meta'][0]['latent_group_names']
                else:
                    # Fallback: infer from keys (excludes image, meta, and scalar controls)
                    val_latent_group_keys = [k for k in val_cond_input.keys() if k not in ['image', 'meta'] and isinstance(val_cond_input.get(k), torch.Tensor) and val_cond_input[k].ndim > 2]
                
                for group_key in val_latent_group_keys:
                    val_cond_input[group_key] = val_cond_input[group_key][:val_num_samples].float().to(device)
                
                expected_latent_size = urban_dataset.latent_size  
                is_already_latent = (val_prediction_data.shape[-1] == expected_latent_size)
                
                if is_already_latent:
                    # Data is already in latent space (precomputed latents)
                    val_im_latent = val_prediction_data
                elif vae is not None:
                    # Data is in pixel space - encode to latent
                    val_im_latent, _, _ = vae.encode(val_prediction_data)
                else:
                    raise ValueError("VAE is required for encoding pixel-space validation data to latents.")
                
                # Extract mask
                val_mask_latent = None
                if 'image' in val_cond_input and 'meta' in val_cond_input:
                    pixel_space_names = val_cond_input['meta'][0].get('pixel_space_names', [])
                    if pixel_space_names and 'inpainting_mask' in pixel_space_names:
                        mask_idx = pixel_space_names.index('inpainting_mask')
                        val_mask_latent = val_cond_input['image'][:, mask_idx:mask_idx+1, :, :]
                
                if val_mask_latent is None:
                    val_mask_latent = torch.ones_like(val_im_latent[:, :1, :, :])
                
                # Initialize from noise in masked region
                x_val = val_im_latent.clone()
                val_noise_context = None
                
                if inpainting_mode == "hard":
                    x_val = val_mask_latent * torch.randn_like(x_val) + (1 - val_mask_latent) * x_val
                    val_noise_context = torch.randn_like(x_val)
                else:
                    x_val = torch.randn_like(x_val)
                
                # Sampling loop with DDIM scheduler (fast validation)
                for step_idx in tqdm(reversed(range(val_scheduler.ddim_steps)), desc="DDIM validation sampling", total=val_scheduler.ddim_steps):
                    # Get full timestep value for model conditioning
                    t_value = val_scheduler.ddim_timesteps[step_idx].item()
                    t_tensor = torch.full((val_num_samples,), t_value, device=device, dtype=torch.long)
                    
                    # Predict noise
                    noise_pred = model(x_val, t_tensor, cond_input=val_cond_input)
                    
                    # Apply CFG if specified
                    if val_guidance_scale is not None and val_guidance_scale != 1.0:
                        # Unconditional prediction
                        from model.utils.diffusion_utils import make_uncond_input_keep_mask
                        uncond_input = make_uncond_input_keep_mask(val_cond_input)
                        noise_pred_uncond = model(x_val, t_tensor, cond_input=uncond_input)
                        # CFG: noise = uncond + scale * (cond - uncond)
                        noise_pred = noise_pred_uncond + val_guidance_scale * (noise_pred - noise_pred_uncond)
                    
                    # DDIM denoise step (uses step_idx, not timestep value)
                    if inpainting_mode == "hard":
                        x_val, _ = val_scheduler.sample_prev_timestep_inpainting(
                            x_val, noise_pred, step_idx,
                            val_im_latent,
                            val_mask_latent,
                            noise_context=val_noise_context
                        )
                    else:
                        x_val, _ = val_scheduler.sample_prev_timestep(x_val, noise_pred, step_idx)
                
                # Decode to pixel space
                if vae is not None:
                    val_decoded = vae.decode(x_val)
                else:
                    val_decoded = x_val
                
                # Save validation samples
                val_save_dir = os.path.join(out_dir, validation_dir_name, f'epoch_{epoch_idx + 1:04d}')
                os.makedirs(val_save_dir, exist_ok=True)
                
                save_layerwise_samples(
                    tensor=val_decoded,
                    layer_names=prediction_layers,
                    layers_registry=layers_registry,
                    save_dir=val_save_dir,
                    filename_prefix='validation',
                    n_samples=val_num_samples,
                    is_reconstruction=True,
                    use_colormaps=True
                )
                
                if 'rgb' in [l.lower() for l in prediction_layers]:
                    rgb_val_path = os.path.join(val_save_dir, 'validation_RGB_composite.png')
                    save_rgb_composite(
                        tensor=val_decoded,
                        layer_names=prediction_layers,
                        save_path=rgb_val_path,
                        n_samples=val_num_samples,
                        normalize_per_channel=True
                    )
                
                print(f"✓ Saved validation samples to {val_save_dir}")
                
                # Restore original weights if using EMA
                if ema_model is not None:
                    model_for_ema = model.module if hasattr(model, 'module') else model
                    ema_model.restore(model_for_ema)  # Restore original weights
                
                model.train()
        
        # Save checkpoint
        if is_main:
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_name = train_config.get('checkpoint_name', f'{mode}_diffusion_ckpt.pth')
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
            
            # Save checkpoint with training state for resuming
            checkpoint_state = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            
            # Save EMA state if available
            if ema_model is not None:
                checkpoint_state['ema_state_dict'] = ema_model.state_dict()
            
            # Save LR scheduler state if available
            if lr_scheduler is not None:
                checkpoint_state['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            
            torch.save(checkpoint_state, checkpoint_path)
            
            # Periodic checkpoint
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'{mode}_diffusion_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(checkpoint_state, periodic_path)
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        print('\n' + "="*50)
        print(f'✓ {mode.upper()} Diffusion Training Complete!')
        print(f'✓ Total training time: {training_time/3600:.2f} hours')
        print("="*50)
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Diffusion Model DDP for Urban Inpainting')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, default='semantic',
                        help='Diffusion stage to train (must match a key in config diffusion_stages, e.g., "semantic", "satellite")')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None = train from scratch)')
    
    args = parser.parse_args()
    
    train(mode=args.mode, load_checkpoint_path=args.load_checkpoint)

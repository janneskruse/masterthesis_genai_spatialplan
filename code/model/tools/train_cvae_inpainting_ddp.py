"""
==============================================================================
Training script for Conditional VAE (CVAE) inpainting with DDP.

Fine-tunes a pretrained VAE into a conditional inpainting model that:
  - Encodes masked input (context + mask channel) to latent z
  - Decodes z conditioned on environmental latents + scalar controls
  - Learns to reconstruct the full image, with emphasis on the masked region
==============================================================================
"""

###### import libraries ######
# Standard libraries
import os
import argparse
from pathlib import Path
import time
import yaml
import numpy as np
from tqdm import tqdm

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from model.blocks.cvae import ConditionalVAE
from model.blocks.discriminator import Discriminator
from model.blocks.vae_registry import VAERegistry
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.distributed import setup_distributed, cleanup_distributed
from model.utils.vae_utils import (
    save_vae_reconstruction_samples, 
    get_kl_weight, 
    PosWeightEMA, 
    compute_reconstruction_loss
)
from model.utils.layer_config import count_layer_channels, get_layer_info
from model.utils.checkpoint import load_checkpoint, check_existing_paths
from model.utils.config_utils import build_scalar_specs, compute_cvae_cond_channels
from helpers.load_configs import load_configs, add_config_arguments

# Load CUDA
load_cuda()


def compute_masked_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_names: list,
    layer_names: list,
    layers_registry: dict,
    mask_loss_weight: float = 2.0,
    outside_weight: float = 0.5,
    binary_weight: float = 1.0,
    continuous_weight: float = 1.0,
    layer_dice_config: dict = None,
    posw_ema=None,
    layer_weights: dict = None,
) -> tuple[dict, torch.Tensor]:
    """
    Compute reconstruction loss with separate weighting inside/outside mask.
    
    Uses per-pixel mask weighting on type-aware losses to emphasize the
    inpainting region while preserving context reconstruction quality.
    
    For binary channels, sigmoid is applied to logits before computing
    per-pixel differences (so L1 compares probabilities, not raw logits).
    For categorical channels, softmax is applied before pixel diff.
    Continuous channels are compared directly.
    
    Args:
        recon: Reconstructed output [B, C, H, W] (logits for binary/categorical)
        target: Ground truth [B, C, H, W]
        mask: Inpainting mask [B, 1, H, W] (1 = inpaint region)
        channel_names: List of channel names
        layer_names: List of layer names per channel
        layers_registry: Global layer config
        mask_loss_weight: Weight for loss inside masked region
        outside_weight: Weight for loss outside masked region
        binary_weight: Base weight for binary channels
        continuous_weight: Base weight for continuous channels
        layer_dice_config: Per-layer dice loss overrides
        posw_ema: Binary pos-weight EMA tracker
        layer_weights: Per-layer weight overrides
        
    Returns:
        (loss_dict, total_loss) tuple
    """
    # Compute full-image typed reconstruction loss (BCE, Dice, MSE, etc.)
    # This uses proper loss functions for each channel type
    loss_dict_full, recon_loss_full = compute_reconstruction_loss(
        recon, target, channel_names, layer_names,
        layers_registry,
        binary_weight=binary_weight,
        continuous_weight=continuous_weight,
        layer_dice_config=layer_dice_config,
        posw_ema=posw_ema,
        all_channels_tensor=target,
        layer_weights=layer_weights,
    )
    
    # Compute mask-weighted per-pixel loss for inpainting emphasis.
    # Convert logits → probabilities for binary/categorical channels so that
    # L1 differences are meaningful (comparing values in the same [0,1] range).
    mask_ratio = mask.mean()
    
    if recon.shape == target.shape:
        recon_activated = recon.clone()
        ch_idx = 0
        processed_categorical = set()
        
        for layer_name in layer_names:
            layer_info = layers_registry.get(layer_name, {})
            layer_type = layer_info.get('type', 'continuous')
            
            if layer_type == 'binary':
                recon_activated[:, ch_idx:ch_idx+1] = torch.sigmoid(recon[:, ch_idx:ch_idx+1])
                ch_idx += 1
            elif layer_type == 'categorical' and layer_name not in processed_categorical:
                processed_categorical.add(layer_name)
                num_classes = layer_info.get('num_classes', 1)
                recon_activated[:, ch_idx:ch_idx+num_classes] = torch.softmax(
                    recon[:, ch_idx:ch_idx+num_classes], dim=1
                )
                ch_idx += num_classes
            elif layer_type == 'categorical':
                ch_idx += 1  # already processed
            else:
                ch_idx += 1  # continuous: keep raw values
        
        # Per-pixel L1 on activated outputs vs targets (both in [0,1] for binary)
        pixel_diff = F.l1_loss(recon_activated, target, reduction='none')  # [B, C, H, W]
        
        # Area-normalized mask weighting: compute inside/outside losses separately
        # so the effective loss is independent of mask coverage
        mask_pixels = mask.sum().clamp(min=1.0)
        outside_pixels = (1.0 - mask).sum().clamp(min=1.0)
        
        inside_loss = (pixel_diff * mask).sum() / mask_pixels
        outside_loss = (pixel_diff * (1.0 - mask)).sum() / outside_pixels
        weighted_pixel_loss = mask_loss_weight * inside_loss + outside_weight * outside_loss
    else:
        weighted_pixel_loss = torch.tensor(0.0, device=recon.device)
    
    # Total loss: typed loss (correct gradients) + mask-weighted pixel loss (emphasis)
    total_loss = recon_loss_full + weighted_pixel_loss
    
    loss_dict_full['typed_recon'] = recon_loss_full.item() if isinstance(recon_loss_full, torch.Tensor) else recon_loss_full
    loss_dict_full['masked_pixel'] = weighted_pixel_loss.item() if isinstance(weighted_pixel_loss, torch.Tensor) else weighted_pixel_loss
    loss_dict_full['masked_recon'] = total_loss.item()
    loss_dict_full['mask_coverage'] = mask_ratio.item()
    
    return loss_dict_full, total_loss


########## Main Training Function #############
def train_cvae(mode: str = 'semantic', load_checkpoint_path: str = None):
    """
    CVAE inpainting training function.
    
    Fine-tunes a pretrained VAE into a conditional inpainting model using DDP.
    
    Args:
        mode: Target VAE group name (must match a key in config['cvae_inpainting'])
        load_checkpoint_path: Optional path to CVAE checkpoint to resume from
    """
    
    # //////////////////////////////////////////////////
    # ============= load config files =================
    # /////////////////////////////////////////////////
    config = load_configs()
    data_config = config['data_config']
    train_config_global = config['train_params']
    
    # //////////////////////////////////////////////////////////
    # === Setup training environment with all configuration ===
    # //////////////////////////////////////////////////////////
    
    training_start_time = time.time()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    ###### setup config variables #######
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    # Validate CVAE config exists
    cvae_configs = config.get('cvae_inpainting', {})
    if mode not in cvae_configs:
        raise ValueError(
            f"CVAE inpainting config for '{mode}' not found. "
            f"Available: {list(cvae_configs.keys())}"
        )
    cvae_config = cvae_configs[mode]
    
    # Validate VAE group exists
    vae_groups = config.get('vae_groups', {})
    target_group = cvae_config.get('target_group', mode)
    if target_group not in vae_groups:
        raise ValueError(
            f"Target VAE group '{target_group}' not found. "
            f"Available: {list(vae_groups.keys())}"
        )
    
    vae_group_config = vae_groups[target_group]
    layers_registry = config.get('layers', {})
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"CVAE Inpainting Training: {mode.upper()}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"{'='*60}")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}, Local rank: {local_rank}")
        print(f"  Target VAE group: {target_group}")
    
    # Parse layers and compute channels
    group_layers = vae_group_config.get('layers', [])
    if not group_layers:
        raise ValueError(f"VAE group '{target_group}' has no layers defined")
    
    num_input_channels = 0
    layer_names = []
    for layer_name in group_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_input_channels += num_channels
        layer_names.append(layer_name)
    
    if is_main:
        print(f"  Layers: {layer_names}")
        print(f"  Input channels: {num_input_channels}")
    
    # Get VAE architecture config
    autoencoder_config = {
        'z_channels': vae_group_config.get('z_channels', 4),
        'down_channels': vae_group_config.get('down_channels', [32, 64, 128, 128]),
        'mid_channels': vae_group_config.get('mid_channels', [128, 128]),
        'down_sample': vae_group_config.get('down_sample', [True, True, True]),
        'attn_down': vae_group_config.get('attn_down', [False, False, False]),
        'norm_channels': vae_group_config.get('norm_channels', 32),
        'num_heads': vae_group_config.get('num_heads', 2),
        'num_down_layers': vae_group_config.get('num_down_layers', 2),
        'num_mid_layers': vae_group_config.get('num_mid_layers', 2),
        'num_up_layers': vae_group_config.get('num_up_layers', 2),
    }
    
    # CVAE conditioning config (auto-computed from conditioning groups)
    cond_channels, cond_projected_channels = compute_cvae_cond_channels(cvae_config, vae_groups)
    cond_emb_dim = cvae_config.get('cond_emb_dim', 128)
    
    # Build scalar specs for CVAE
    scalar_specs = build_scalar_specs(config, cvae_config)
    
    if is_main:
        print(f"  Conditioning channels: {cond_channels}")
        print(f"  Projected channels: {cond_projected_channels}")
        print(f"  Scalar controls: {list(scalar_specs.keys()) if scalar_specs else 'None'}")
    
    # Get CVAE training config
    cvae_training_config = train_config_global.get('cvae_training', {}).get(mode, {})
    
    num_epochs = cvae_training_config.get('epochs', 100)
    batch_size = cvae_training_config.get('batch_size', 4)
    base_lr = cvae_training_config.get('lr', 0.00005)
    kl_weight_final = cvae_training_config.get('kl_weight', 0.001)
    kl_annealing_config = cvae_training_config.get('kl_annealing', {})
    mask_loss_weight = cvae_training_config.get('mask_loss_weight', 5.0)
    outside_weight = cvae_training_config.get('outside_weight', 1.0)
    binary_channel_weight = cvae_training_config.get('binary_channel_weight', 1.0)
    continuous_channel_weight = cvae_training_config.get('continuous_channel_weight', 2.0)
    dice_weight = cvae_training_config.get('dice_weight', 0.5)
    layer_weights = cvae_training_config.get('layer_weights', None)
    layer_dice_config = cvae_training_config.get('layer_dice_config', {})
    use_discriminator = cvae_training_config.get('use_discriminator', True)
    disc_weight = cvae_training_config.get('disc_weight', 0.5)
    disc_start_steps = cvae_training_config.get('disc_start', 5000)
    use_perceptual = cvae_training_config.get('use_perceptual', False)
    perceptual_weight = cvae_training_config.get('perceptual_weight', 0.0)
    max_grad_norm = cvae_training_config.get('max_grad_norm', 1.0)
    freeze_encoder_epochs = cvae_training_config.get('freeze_encoder_epochs', 0)
    img_save_steps = cvae_training_config.get('img_save_steps', 64)
    tanh_activation = cvae_training_config.get('tanh_activation', False)
    
    # Checkpoint naming from cvae_inpainting config
    checkpoint_name = cvae_config.get('checkpoint_name', f'{mode}_cvae_inpainting_ckpt.pth')
    samples_dir_name = cvae_config.get('samples_dir', f'{mode}_cvae_samples')
    
    # Create output directories
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    samples_dir = os.path.join(out_dir, samples_dir_name)
    
    if load_checkpoint_path is not None:
        load_checkpoint_path = os.path.join(out_dir, load_checkpoint_path)
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Resolve cached patches path
    existing_paths_result = check_existing_paths(
        train_config=train_config_global,
        mode=mode,
        type='vae'  # Reuse patches logic
    )
    existing_patches_path = existing_paths_result.patches_path
    
    if existing_patches_path is not None:
        cache_dir = existing_patches_path
    else:
        cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print(f"Loading Dataset for CVAE '{mode}' training")
        print(f"  Mode string: 'cvae:{mode}'")
        print(f"  Cache: {cache_dir} (cached={use_cached_patches})")
        print(f"{'='*50}")
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        mode=f'cvae:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    if is_main:
        print(f"  Loaded {len(urban_dataset)} training patches")
    
    sampler = DistributedSampler(
        urban_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    disc_start_epoch = disc_start_steps // max(len(data_loader), 1)
    
    ########## Create CVAE Model #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing ConditionalVAE")
        print(f"{'='*50}")
    
    # Add tanh config to autoencoder_config for CVAE init
    autoencoder_config['tanh_activation'] = tanh_activation
    autoencoder_config['tanh_scaling'] = cvae_training_config.get('tanh_scaling', 1.0)
    
    # Resolve pretrained VAE path:
    # 1) Check existing_paths.vae_checkpoints for the target group (absolute path)
    # 2) Fall back to vae_group_config.checkpoint_name in the results dir
    pretrained_path = existing_paths_result.vae_checkpoints.get(target_group, None)
    # print(["[DEBUG] Existing paths check:", pretrained_path, vae_group_config.get('checkpoint_name', None), existing_paths_result, target_group])
    if pretrained_path is None:
        # Use the checkpoint_name from the VAE group config
        vae_ckpt_name = vae_group_config.get('checkpoint_name', f'{target_group}_vae_ckpt.pth')
        pretrained_path = os.path.join(out_dir, vae_ckpt_name)
    
    if os.path.exists(pretrained_path) and load_checkpoint_path is None:
        if is_main:
            print(f"  Initializing from pretrained VAE: {pretrained_path}")
        
        model = ConditionalVAE.from_pretrained_vae(
            pretrained_checkpoint_path=pretrained_path,
            im_channels=num_input_channels,
            model_config=autoencoder_config,
            cond_channels=cond_channels,
            cond_projected_channels=cond_projected_channels,
            scalar_specs=scalar_specs if scalar_specs else None,
            cond_emb_dim=cond_emb_dim,
            device=device,
        )
    else:
        if is_main:
            if load_checkpoint_path:
                print(f"  Will load CVAE checkpoint: {load_checkpoint_path}")
            else:
                print(f"  ⚠ Pretrained VAE not found at {pretrained_path}")
                print(f"  Training from scratch (not recommended)")
        
        model = ConditionalVAE(
            im_channels=num_input_channels,
            model_config=autoencoder_config,
            cond_channels=cond_channels,
            cond_projected_channels=cond_projected_channels,
            scalar_specs=scalar_specs if scalar_specs else None,
            cond_emb_dim=cond_emb_dim,
        )
    
    model = model.to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Scalar MLPs may be unused if no scalars in batch
        )
        if is_main:
            print("  Wrapped in DistributedDataParallel")
    
    model.train()
    
    if is_main:
        model_unwrapped = model.module if hasattr(model, 'module') else model
        param_count = sum(p.numel() for p in model_unwrapped.parameters()) / 1e6
        print(f"  Parameters: {param_count:.2f}M")
        print(f"  Input channels: {num_input_channels} (+1 mask = {num_input_channels + 1} encoder)")
        print(f"  Latent channels: {autoencoder_config['z_channels']}")
        print(f"  Cond channels: {cond_channels} → {cond_projected_channels} projected")
    
    ########## Load Conditioning VAEs for on-the-fly encoding #############
    # If conditioning latents aren't pre-computed, encode full-res images on-the-fly
    cond_latent_groups = cvae_config.get('conditioning', {}).get('latent_space', [])
    vae_registry = None
    
    if cond_latent_groups:
        if is_main:
            print(f"\nLoading conditioning VAE models for on-the-fly encoding fallback...")
        
        vae_registry = VAERegistry(config, device)
        
        for cond_spec in cond_latent_groups:
            cond_group = cond_spec.get('group')
            if cond_group and cond_group in vae_groups:
                cond_vae_config = vae_groups[cond_group]
                
                # Resolve checkpoint: existing_paths first, then default
                cond_vae_ckpt_path = existing_paths_result.vae_checkpoints.get(cond_group, None)
                if cond_vae_ckpt_path is None:
                    default_cond_ckpt_name = cond_vae_config.get('checkpoint_name', f'{cond_group}_vae_ckpt.pth')
                    cond_vae_ckpt_path = os.path.join(out_dir, default_cond_ckpt_name)
                
                if os.path.exists(cond_vae_ckpt_path):
                    if is_main:
                        print(f"  - {cond_group.upper()} conditioning VAE: {cond_vae_ckpt_path}")
                    vae_registry.load_vae(
                        group_name=cond_group,
                        checkpoint_path=cond_vae_ckpt_path,
                        autoencoder_config=cond_vae_config
                    )
                    vae_registry.freeze(cond_group)
                else:
                    if is_main:
                        print(f"  ⚠ Conditioning VAE checkpoint not found: {cond_vae_ckpt_path}")
                        print(f"    On-the-fly encoding will fail if latents are missing for '{cond_group}'")
        
        if is_main and vae_registry is not None:
            print(f"  ✓ Loaded {len(vae_registry.vaes)} conditioning VAE(s)")
    
    # Discriminator
    discriminator = None
    optimizer_disc = None
    
    if use_discriminator:
        discriminator = Discriminator(
            im_channels=num_input_channels
        ).to(device)
        
        if is_main:
            disc_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
            print(f"  Discriminator: {disc_params:.2f}M params (starts epoch {disc_start_epoch})")
    
    ########## Training Setup #############
    
    # PosWeightEMA for binary channels
    posw_ema = None
    binary_channel_count = sum(
        1 for layer_name in layer_names
        if layers_registry.get(layer_name, {}).get('type') == 'binary'
    )
    if binary_channel_count > 0:
        posw_ema = PosWeightEMA(
            num_channels=binary_channel_count,
            momentum=0.95,
            init=1.0,
            device=device
        )
    
    # Scale learning rate with world size
    adjusted_lr = base_lr * world_size
    optimizer_cvae = Adam(model.parameters(), lr=adjusted_lr)
    
    if use_discriminator:
        optimizer_disc = Adam(discriminator.parameters(), lr=adjusted_lr)
    
    # Learning rate scheduler (cosine annealing)
    lr_scheduler_config = cvae_training_config.get('lr_scheduler', {})
    scheduler_type = lr_scheduler_config.get('type', 'cosine')
    eta_min = lr_scheduler_config.get('eta_min', 1e-6)
    
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer_cvae, T_max=num_epochs, eta_min=eta_min
        )
    
    # Load CVAE checkpoint if resuming
    start_epoch = 0
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        start_epoch, _ = load_checkpoint(
            checkpoint_path=load_checkpoint_path,
            model=model,
            optimizer=optimizer_cvae,
            device=device,
            is_main=is_main
        )
        # Load scheduler state if available, otherwise advance to correct epoch
        if scheduler is not None:
            ckpt_data = torch.load(load_checkpoint_path, map_location=device, weights_only=False)
            if 'scheduler_state_dict' in ckpt_data:
                scheduler.load_state_dict(ckpt_data['scheduler_state_dict'])
                if is_main:
                    print(f"  Loaded scheduler state (LR: {scheduler.get_last_lr()[0]:.8f})")
            else:
                # Advance scheduler to match resumed epoch
                for _ in range(start_epoch):
                    scheduler.step()
                if is_main:
                    print(f"  Advanced scheduler to epoch {start_epoch} (LR: {scheduler.get_last_lr()[0]:.8f})")
    
    if is_main:
        print(f"\n{'='*50}")
        print("Training Configuration")
        print(f"{'='*50}")
        print(f"  Epochs: {num_epochs} (starting from {start_epoch})")
        print(f"  Learning rate: {adjusted_lr:.6f} (base {base_lr:.6f} × {world_size})")
        print(f"  LR scheduler: {scheduler_type} (eta_min={eta_min:.1e})" if scheduler else "  LR scheduler: none")
        print(f"  Batch size: {batch_size} per GPU, {batch_size * world_size} effective")
        print(f"  KL weight: {kl_weight_final} (annealing: {kl_annealing_config.get('enabled', False)})")
        print(f"  Mask loss weight: {mask_loss_weight} (outside: {outside_weight})")
        print(f"  Encoder freeze epochs: {freeze_encoder_epochs}")
        if layer_weights:
            print(f"  Layer weights: {layer_weights}")
        print(f"{'='*50}")
    
    # /////////////////////////////////////////////////
    # =============== Training Loop ===================
    # /////////////////////////////////////////////////
    if is_main:
        print(f"\n{'='*50}")
        print(f"Starting CVAE Training")
        print(f"{'='*50}")
    
    global_step = 0
    
    for epoch_idx in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        
        # KL annealing
        current_kl_weight = get_kl_weight(epoch_idx, kl_annealing_config)
        
        # Encoder freezing
        model_unwrapped = model.module if hasattr(model, 'module') else model
        if freeze_encoder_epochs > 0:
            if epoch_idx < freeze_encoder_epochs:
                # Freeze encoder
                for name, param in model_unwrapped.named_parameters():
                    if name.startswith('encoder_') or name.startswith('pre_quant_conv'):
                        param.requires_grad = False
            elif epoch_idx == freeze_encoder_epochs:
                # Unfreeze encoder
                for param in model_unwrapped.parameters():
                    param.requires_grad = True
                if is_main:
                    print(f"\n  Unfreezing encoder at epoch {epoch_idx + 1}")
        
        losses_epoch = []
        losses_disc_epoch = []
        
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            # =============================================
            # Extract data from CVAE dataset mode
            # =============================================
            if len(data) == 2:
                target_tensor, cond_dict = data
            else:
                raise ValueError("CVAE mode must return (target, cond_dict)")
            
            # Move target to device
            target_tensor = target_tensor.float().to(device)
            
            # Extract conditioning components
            mask = cond_dict['mask'].float().to(device)  # [B, 1, H, W]
            decoder_cond = cond_dict['decoder_cond'].float().to(device)  # [B, cond_ch, H', W']
            
            # Handle on-the-fly encoding for conditioning groups
            # When latents weren't available, dataset returns {group_name}_image
            # We encode them here and rebuild decoder_cond
            if vae_registry is not None:
                for cond_spec in cond_latent_groups:
                    group_name = cond_spec['group']
                    image_key = f'{group_name}_image'
                    if image_key in cond_dict:
                        # Full-res image needs encoding → latent
                        cond_image = cond_dict[image_key].float().to(device)  # [B, C, H, W]
                        cond_vae = vae_registry.get_vae(group_name)
                        with torch.no_grad():
                            cond_latent, _, _ = cond_vae.encode(cond_image)  # [B, z, H', W']
                        
                        # Rebuild decoder_cond: replace or append the encoded latent
                        # decoder_cond was built from latent_cond_list = [mask_latent, ...latents...]
                        # If a group failed, it's missing from decoder_cond → append it
                        decoder_cond = torch.cat([decoder_cond, cond_latent], dim=1)
                        
                        if is_main and batch_idx == 0 and epoch_idx == start_epoch:
                            print(f"  On-the-fly encoded '{group_name}': {cond_image.shape} → {cond_latent.shape}")
                            print(f"  Updated decoder_cond shape: {decoder_cond.shape}")
            
            # Extract metadata
            meta = cond_dict.get('meta', {})
            if isinstance(meta, list) and len(meta) > 0:
                channel_names = meta[0].get('channel_names', [])
                layer_names_batch = meta[0].get('layer_names', [])
            elif isinstance(meta, dict):
                channel_names = meta.get('channel_names', [])
                layer_names_batch = meta.get('layer_names', [])
            else:
                channel_names = []
                layer_names_batch = []
            
            # Build scalar conditioning dict
            scalar_cond = {}
            if scalar_specs:
                for key in scalar_specs:
                    if key in cond_dict:
                        scalar_cond[key] = cond_dict[key].float().to(device)
            
            # Sanity check on first batch per epoch
            if is_main and batch_idx == 0:
                print(f"\n[Epoch {epoch_idx + 1}] KL weight: {current_kl_weight:.6f}")
                print(f"  Target shape: {target_tensor.shape}")
                print(f"  Mask shape: {mask.shape}, coverage: {mask.mean():.3f}")
                print(f"  Decoder cond shape: {decoder_cond.shape}")
                if scalar_cond:
                    print(f"  Scalar keys: {list(scalar_cond.keys())}")
                print(f"  Channels: {channel_names}")
            
            # =============================================
            # 1) CVAE / Generator step
            # =============================================
            if use_discriminator and epoch_idx >= disc_start_epoch:
                for p in discriminator.parameters():
                    p.requires_grad = False
            
            optimizer_cvae.zero_grad()
            
            # Forward pass
            recon, z, mean, logvar = model(
                target_tensor, mask, decoder_cond,
                scalar_cond=scalar_cond if scalar_cond else None
            )
            
            # Masked reconstruction loss
            loss_dict, recon_loss = compute_masked_reconstruction_loss(
                recon, target_tensor, mask,
                channel_names, layer_names_batch,
                layers_registry,
                mask_loss_weight=mask_loss_weight,
                outside_weight=outside_weight,
                binary_weight=binary_channel_weight,
                continuous_weight=continuous_channel_weight,
                layer_dice_config=layer_dice_config,
                posw_ema=posw_ema,
                layer_weights=layer_weights,
            )
            
            # KL divergence loss (per-pixel normalized, then weighted)
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp()
            )
            kl_loss = kl_loss / (target_tensor.shape[0] * target_tensor.shape[2] * target_tensor.shape[3])
            
            # Generator loss (fool discriminator)
            gen_loss = 0.0
            if use_discriminator and epoch_idx >= disc_start_epoch:
                disc_fake = discriminator(recon)
                gen_loss = -torch.mean(disc_fake)
            
            # Total CVAE loss
            cvae_loss = (
                recon_loss +
                current_kl_weight * kl_loss +
                disc_weight * gen_loss
            )
            
            cvae_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer_cvae.step()
            
            # =============================================
            # 2) Discriminator step
            # =============================================
            if use_discriminator and epoch_idx >= disc_start_epoch:
                for p in discriminator.parameters():
                    p.requires_grad = True
                
                optimizer_disc.zero_grad()
                
                disc_real = discriminator(target_tensor.detach())
                disc_fake = discriminator(recon.detach())
                disc_loss = (
                    torch.mean(torch.relu(1.0 - disc_real)) +
                    torch.mean(torch.relu(1.0 + disc_fake))
                )
                
                disc_loss.backward()
                optimizer_disc.step()
                losses_disc_epoch.append(disc_loss.item())
            
            losses_epoch.append(cvae_loss.item())
            global_step += 1
            
            # Update progress bar
            if is_main:
                postfix = {
                    'loss': f'{cvae_loss.item():.4f}',
                    'typed': f'{loss_dict.get("typed_recon", 0):.4f}',
                    'mpx': f'{loss_dict.get("masked_pixel", 0):.4f}',
                    'kl': f'{kl_loss.item():.6f}',
                    'kl_w': f'{current_kl_weight:.5f}',
                }
                if use_discriminator and epoch_idx >= disc_start_epoch and losses_disc_epoch:
                    postfix['disc'] = f'{disc_loss.item():.4f}'
                progress_bar.set_postfix(postfix)
            
            # Save sample reconstructions
            if is_main and global_step % img_save_steps == 0:
                with torch.no_grad():
                    save_vae_reconstruction_samples(
                        input_tensor=target_tensor,
                        recon_tensor=recon,
                        channel_names=channel_names,
                        layer_names=layer_names_batch,
                        layers_registry=layers_registry,
                        save_dir=samples_dir,
                        step=global_step,
                        n_samples=8,
                        save_rgb_composite=False,
                    )
        
        # Synchronize epoch
        if world_size > 1:
            dist.barrier()
        
        # Epoch summary
        if is_main:
            epoch_loss = np.mean(losses_epoch)
            current_lr = scheduler.get_last_lr()[0] if scheduler else adjusted_lr
            summary = f'\n  Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | KL_w: {current_kl_weight:.6f} | LR: {current_lr:.2e}'
            if use_discriminator and epoch_idx >= disc_start_epoch and losses_disc_epoch:
                summary += f' | Disc: {np.mean(losses_disc_epoch):.4f}'
            print(summary)
        
        # Save checkpoint (main process only)
        if is_main:
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
            
            checkpoint_state = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer_cvae.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': epoch_loss,
                'kl_weight': current_kl_weight,
                'cvae_config': cvae_config,
                'autoencoder_config': autoencoder_config,
                'scalar_specs': scalar_specs,
                'cond_channels': cond_channels,
                'cond_projected_channels': cond_projected_channels,
            }
            torch.save(checkpoint_state, checkpoint_path)
            
            # Periodic checkpoints
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(
                    out_dir,
                    f'{mode}_cvae_epoch_{epoch_idx + 1}.pth'
                )
                torch.save(checkpoint_state, periodic_path)
                print(f'  Saved periodic checkpoint: {periodic_path}')
        
        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        if world_size > 1:
            dist.barrier()
    
    # Training complete
    training_time = time.time() - training_start_time
    
    if is_main:
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print(f"\n{'='*60}")
        print(f"CVAE Inpainting Training Complete")
        print(f"  Mode: {mode}")
        print(f"  Time: {hours}h {minutes}m {seconds}s")
        print(f"  Final checkpoint: {os.path.join(out_dir, checkpoint_name)}")
        print(f"{'='*60}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train CVAE Inpainting DDP')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, default='semantic',
                        help='CVAE target group (must match key in config cvae_inpainting)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to CVAE checkpoint to resume training from')
    
    args = parser.parse_args()
    
    try:
        train_cvae(mode=args.mode, load_checkpoint_path=args.load_checkpoint)
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Training interrupted by user (Ctrl+C)")
        print("="*50)
    except Exception as e:
        print("\n" + "="*50)
        print(f"Training failed: {e}")
        print("="*50)
        raise
    finally:
        cleanup_distributed()

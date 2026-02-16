"""
==============================================================================
Sampling script for CVAE inpainting.

Loads a trained ConditionalVAE and generates diverse inpainting samples:
  1. Load evaluation dataset in 'cvae:<group>' mode
  2. Encode masked context → posterior (mean, logvar)
  3. Sample multiple z's from posterior (or prior with temperature scaling)
  4. Decode each z with environmental/scalar conditioning
  5. Composite: mask * generated + (1-mask) * original
  6. Apply post-processing (binary sharpening, etc.)
  7. Save outputs + compute building metrics
==============================================================================
"""

###### import libraries ######
# Standard libraries
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Data Science/ML libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# Local imports
from model.blocks.cvae import ConditionalVAE
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from model.utils.layer_config import count_layer_channels, get_layer_info
from model.utils.scalar_controls import parse_scalar_controls_config
from model.utils.post_process import apply_post_processing
from model.utils.building_metrics import aggregate_metrics_batch, print_metrics_summary
from model.utils.checkpoint import check_existing_paths
from model.utils.config_utils import compute_cvae_cond_channels
from helpers.load_configs import load_configs, add_config_arguments
from helpers.indexed_outputs import get_next_run_idx

# Load CUDA
load_cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_scalar_specs(config: dict, cvae_config: dict) -> dict:
    """
    Build scalar_specs dict for ConditionalVAE from config.
    
    Args:
        config: Full config dict
        cvae_config: CVAE inpainting config section for target group
        
    Returns:
        Dict mapping scalar keys to spec dicts with 'mlp_hidden'
    """
    scalar_control_names = cvae_config.get('scalar_controls', [])
    if not scalar_control_names:
        return {}
    
    control_specs = parse_scalar_controls_config(config, stage_control_names=scalar_control_names)
    
    scalar_specs = {}
    for spec in control_specs:
        conditioning_cfg = spec.get('conditioning', {})
        for key in spec['keys']:
            scalar_specs[key] = {
                'mlp_hidden': conditioning_cfg.get('mlp_hidden', 128)
            }
    
    return scalar_specs


def composite_inpainting(
    generated: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Composite generated region into original context.
    
    Args:
        generated: Full reconstruction from CVAE [B, C, H, W]
        original: Original target image [B, C, H, W]
        mask: Inpainting mask [B, 1, H, W] (1 = generated, 0 = keep original)
        
    Returns:
        Composited image [B, C, H, W]
    """
    return mask * generated + (1.0 - mask) * original


def run_sampling(
    mode: str = 'semantic',
    num_samples: int = -1,
    num_diverse: int = None,
    sampling_temperature: float = None,
    override_scalars: dict = None,
):
    """
    Main sampling function for CVAE inpainting.
    
    Args:
        mode: CVAE target group (must match key in config cvae_inpainting)
        num_samples: Number of eval samples to process (-1 = all)
        num_diverse: Override for num_diverse_samples from config
        sampling_temperature: Override for temperature from config
        override_scalars: Optional dict of scalar overrides {key: value}
    """
    
    # //////////////////////////////////////////////////
    # ============= load config files =================
    # /////////////////////////////////////////////////
    config = load_configs()
    data_config = config['data_config']
    train_config_global = config['train_params']
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    task_name = train_config_global.get('task_name', 'urban_inpainting')
    out_dir = f"{big_data_storage_path}/results/{task_name}"
    
    # Validate CVAE config
    cvae_configs = config.get('cvae_inpainting', {})
    if mode not in cvae_configs:
        raise ValueError(
            f"CVAE config for '{mode}' not found. Available: {list(cvae_configs.keys())}"
        )
    cvae_config = cvae_configs[mode]
    
    layers_registry = config.get('layers', {})
    vae_groups = config.get('vae_groups', {})
    
    target_group = cvae_config.get('target_group', mode)
    if target_group not in vae_groups:
        raise ValueError(f"Target VAE group '{target_group}' not found.")
    
    vae_group_config = vae_groups[target_group]
    
    # Sampling config
    sampling_config = config.get('cvae_sampling', {}).get(mode, {})
    if num_diverse is None:
        num_diverse = sampling_config.get('num_diverse_samples', 5)
    if sampling_temperature is None:
        sampling_temperature = sampling_config.get('temperature', 1.0)
    binary_sharpening = sampling_config.get('binary_sharpening', True)
    sharpening_threshold = sampling_config.get('sharpening_threshold', 0.5)
    output_dir_name = sampling_config.get('output_dir', f'{mode}_cvae_inpainting_samples')
    
    print(f"\n{'='*60}")
    print(f"CVAE Inpainting Sampling: {mode.upper()}")
    print(f"{'='*60}")
    print(f"  Diverse samples per input: {num_diverse}")
    print(f"  Sampling temperature: {sampling_temperature}")
    print(f"  Binary sharpening: {binary_sharpening}")
    print(f"  Output dir: {output_dir_name}")
    
    # Parse layers
    group_layers = vae_group_config.get('layers', [])
    num_input_channels = 0
    layer_names = []
    for layer_name in group_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_input_channels += num_channels
        layer_names.append(layer_name)
    
    print(f"  Layers: {layer_names} ({num_input_channels} channels)")
    
    # VAE architecture config
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
        'tanh_activation': False,
        'tanh_scaling': 1.0,
    }
    
    # CVAE conditioning config (auto-computed from conditioning groups)
    cond_channels, cond_projected_channels = compute_cvae_cond_channels(cvae_config, vae_groups)
    cond_emb_dim = cvae_config.get('cond_emb_dim', 128)
    scalar_specs = build_scalar_specs(config, cvae_config)
    
    # //////////////////////////////////////////////////
    # ============= Load CVAE Model ===================
    # /////////////////////////////////////////////////
    checkpoint_name = cvae_config.get('checkpoint_name', f'{mode}_cvae_inpainting_ckpt.pth')
    checkpoint_path = os.path.join(out_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CVAE checkpoint not found: {checkpoint_path}")
    
    print(f"\n  Loading CVAE from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Override configs from checkpoint if available (ensures sampling matches training)
    if 'autoencoder_config' in checkpoint:
        autoencoder_config.update(checkpoint['autoencoder_config'])
    if 'scalar_specs' in checkpoint:
        scalar_specs = checkpoint['scalar_specs']
    if 'cond_channels' in checkpoint:
        cond_channels = checkpoint['cond_channels']
    if 'cond_projected_channels' in checkpoint:
        cond_projected_channels = checkpoint['cond_projected_channels']
    
    model = ConditionalVAE(
        im_channels=num_input_channels,
        model_config=autoencoder_config,
        cond_channels=cond_channels,
        cond_projected_channels=cond_projected_channels,
        scalar_specs=scalar_specs if scalar_specs else None,
        cond_emb_dim=cond_emb_dim,
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded CVAE with {param_count:.2f}M parameters")
    
    # //////////////////////////////////////////////////
    # ============= Load Eval Dataset =================
    # /////////////////////////////////////////////////
    
    # Resolve cached patches
    existing_paths_result = check_existing_paths(
        train_config=train_config_global,
        mode=mode,
        type='vae'
    )
    existing_patches_path = existing_paths_result.patches_path
    
    if existing_patches_path is not None:
        cache_dir = existing_patches_path
    else:
        cache_dir = f"{big_data_storage_path}/processed/{task_name}/patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    print(f"\n  Loading eval dataset (mode='cvae:{mode}')")
    
    eval_dataset = UrbanInpaintingDataset(
        split='eval',
        mode=f'cvae:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    total_eval = len(eval_dataset)
    if num_samples > 0:
        total_eval = min(num_samples, total_eval)
    
    print(f"  Eval samples: {total_eval}")
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Process one at a time for diverse sampling
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # //////////////////////////////////////////////////
    # ============= Setup Output ====================
    # /////////////////////////////////////////////////
    
    # Create indexed output directory
    samples_out_dir = os.path.join(out_dir, output_dir_name)
    os.makedirs(samples_out_dir, exist_ok=True)
    
    run_idx = get_next_run_idx(samples_out_dir, 'run')
    run_dir = os.path.join(samples_out_dir, f'run_idx{run_idx}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save sampling config
    sampling_meta = {
        'mode': mode,
        'num_diverse_samples': num_diverse,
        'sampling_temperature': sampling_temperature,
        'binary_sharpening': binary_sharpening,
        'sharpening_threshold': sharpening_threshold,
        'checkpoint': checkpoint_path,
        'total_eval_samples': total_eval,
        'scalar_overrides': override_scalars,
    }
    with open(os.path.join(run_dir, 'sampling_config.json'), 'w') as f:
        json.dump(sampling_meta, f, indent=2)
    
    print(f"  Output: {run_dir}")
    
    # //////////////////////////////////////////////////
    # ============= Sampling Loop ====================
    # /////////////////////////////////////////////////
    
    print(f"\n{'='*60}")
    print(f"Generating {num_diverse} diverse samples per input")
    print(f"{'='*60}")
    
    all_metrics = []
    
    post_process_config = {
        'sharpen_binary': binary_sharpening,
        'threshold': sharpening_threshold,
    }
    
    with torch.no_grad():
        for sample_idx, data in enumerate(tqdm(eval_loader, total=total_eval, desc="Sampling")):
            if sample_idx >= total_eval:
                break
            
            # Extract data
            if len(data) == 2:
                target_tensor, cond_dict = data
            else:
                raise ValueError("CVAE mode must return (target, cond_dict)")
            
            target_tensor = target_tensor.float().to(device)
            mask = cond_dict['mask'].float().to(device)
            decoder_cond = cond_dict['decoder_cond'].float().to(device)
            
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
            
            # Build scalar conditioning
            scalar_cond = {}
            if scalar_specs:
                for key in scalar_specs:
                    if override_scalars and key in override_scalars:
                        scalar_cond[key] = torch.tensor(
                            [override_scalars[key]], dtype=torch.float32, device=device
                        )
                    elif key in cond_dict:
                        scalar_cond[key] = cond_dict[key].float().to(device)
            
            # Encode to get posterior parameters
            _, mean, logvar = model.module.encode(target_tensor, mask) if hasattr(model, 'module') else model.encode(target_tensor, mask)
            
            std = torch.exp(0.5 * logvar)
            
            # Generate diverse samples
            diverse_outputs = []
            
            for div_idx in range(num_diverse):
                # Sample z with temperature scaling
                eps = torch.randn_like(mean)
                z = mean + sampling_temperature * std * eps
                
                # Decode with conditioning
                m = model.module if hasattr(model, 'module') else model
                recon = m.decode(z, decoder_cond, scalar_cond=scalar_cond if scalar_cond else None)
                
                # Post-process
                recon_processed = apply_post_processing(
                    recon, layer_names_batch, layers_registry,
                    post_process_config=post_process_config,
                    mask=mask,
                )
                
                # Composite: keep original context, use generated inside mask
                composited = composite_inpainting(recon_processed, target_tensor, mask)
                
                diverse_outputs.append(composited)
            
            # Stack diverse outputs: [num_diverse, C, H, W]
            diverse_stack = torch.cat(diverse_outputs, dim=0)
            
            # Save individual sample outputs
            sample_dir = os.path.join(run_dir, f'sample_{sample_idx:04d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save target, mask, and diverse outputs
            torch.save(target_tensor.cpu(), os.path.join(sample_dir, 'target.pt'))
            torch.save(mask.cpu(), os.path.join(sample_dir, 'mask.pt'))
            torch.save(diverse_stack.cpu(), os.path.join(sample_dir, 'diverse_outputs.pt'))
            
            # Save visual comparison grid
            # Row 1: target | Row 2-N+1: diverse outputs
            # For each layer, create a grid
            for layer_idx, layer_name in enumerate(layer_names):
                ch_start = sum(
                    count_layer_channels(get_layer_info(layers_registry, ln))
                    for ln in layer_names[:layer_idx]
                )
                layer_config = get_layer_info(layers_registry, layer_name)
                n_ch = count_layer_channels(layer_config)
                
                # Extract channels for this layer
                target_layer = target_tensor[:, ch_start:ch_start + n_ch].cpu()
                
                # Build grid: [target, mask*target, diverse1, diverse2, ...]
                grid_tensors = [target_layer]
                masked_target = target_layer * (1.0 - mask.cpu())
                grid_tensors.append(masked_target)
                
                for div_idx in range(num_diverse):
                    div_layer = diverse_outputs[div_idx][:, ch_start:ch_start + n_ch].cpu()
                    grid_tensors.append(div_layer)
                
                # For binary layers, apply sigmoid for visualization
                is_binary = layers_registry.get(layer_name, {}).get('type', 'continuous') == 'binary'
                if is_binary:
                    grid_tensors = [torch.sigmoid(t) if i <= 1 else t for i, t in enumerate(grid_tensors)]
                
                # Make grid: [target, masked, div1, div2, ...]
                all_imgs = torch.cat(grid_tensors, dim=0)  # [N, n_ch, H, W]
                
                # For single-channel, repeat to make 3-channel for visualization
                if n_ch == 1:
                    all_imgs = all_imgs.repeat(1, 3, 1, 1)
                elif n_ch > 3:
                    # Take first 3 channels
                    all_imgs = all_imgs[:, :3]
                
                grid = make_grid(
                    all_imgs.clamp(0, 1),
                    nrow=2 + num_diverse,
                    padding=2,
                    normalize=False,
                )
                save_image(grid, os.path.join(sample_dir, f'{layer_name}_comparison.png'))
            
            # Compute building metrics (if buildings layer present)
            if 'buildings' in layer_names:
                buildings_idx = layer_names.index('buildings')
                ch_start = sum(
                    count_layer_channels(get_layer_info(layers_registry, ln))
                    for ln in layer_names[:buildings_idx]
                )
                
                true_buildings = target_tensor[:, ch_start:ch_start + 1]
                
                # Metrics for each diverse sample
                for div_idx in range(num_diverse):
                    pred_buildings = diverse_outputs[div_idx][:, ch_start:ch_start + 1]
                    
                    # Binarize predictions
                    pred_buildings_binary = (pred_buildings > sharpening_threshold).float()
                    true_buildings_binary = (true_buildings > 0.5).float()
                    
                    try:
                        metrics = aggregate_metrics_batch(
                            pred_buildings_binary,
                            true_buildings_binary,
                            mask,
                        )
                        metrics['sample_idx'] = sample_idx
                        metrics['diverse_idx'] = div_idx
                        all_metrics.append(metrics)
                    except Exception as e:
                        print(f"  Warning: Metrics failed for sample {sample_idx}, div {div_idx}: {e}")
            
            # Diversity metrics: compute pairwise L1 between diverse samples inside mask
            if num_diverse > 1:
                pairwise_diffs = []
                for i in range(num_diverse):
                    for j in range(i + 1, num_diverse):
                        diff = (diverse_outputs[i] - diverse_outputs[j]).abs()
                        # Only inside mask
                        masked_diff = (diff * mask).sum() / mask.sum().clamp(min=1)
                        pairwise_diffs.append(masked_diff.item())
                
                avg_diversity = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
                
                # Save diversity score
                with open(os.path.join(sample_dir, 'diversity.json'), 'w') as f:
                    json.dump({
                        'avg_pairwise_l1': avg_diversity,
                        'pairwise_diffs': pairwise_diffs,
                        'temperature': sampling_temperature,
                    }, f, indent=2)
    
    # //////////////////////////////////////////////////
    # ============= Summary ===========================
    # /////////////////////////////////////////////////
    
    print(f"\n{'='*60}")
    print(f"Sampling Complete")
    print(f"{'='*60}")
    print(f"  Saved {total_eval} samples × {num_diverse} diverse outputs")
    print(f"  Output directory: {run_dir}")
    
    # Aggregate and save metrics
    if all_metrics:
        # Save raw metrics
        with open(os.path.join(run_dir, 'building_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Compute summary 
        avg_metrics = {}
        for key in all_metrics[0]:
            if key in ('sample_idx', 'diverse_idx'):
                continue
            values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
            if values:
                avg_metrics[key] = float(np.mean(values))
        
        print_metrics_summary(avg_metrics, prefix="Average ")
        
        # Save summary
        with open(os.path.join(run_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=2)
    
    print(f"\nDone!")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CVAE Inpainting Sampling')
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, default='semantic',
                        help='CVAE target group')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of eval samples to process (-1 = all)')
    parser.add_argument('--num_diverse', type=int, default=None,
                        help='Number of diverse samples per input (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (overrides config)')
    
    args = parser.parse_args()
    
    run_sampling(
        mode=args.mode,
        num_samples=args.num_samples,
        num_diverse=args.num_diverse,
        sampling_temperature=args.temperature,
    )

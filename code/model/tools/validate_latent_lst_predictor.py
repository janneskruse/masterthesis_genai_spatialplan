# Validation script for Latent LST Predictor
# Creates prediction samples on validation set to assess latent-space LST predictor quality

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Data handling
import torch
import matplotlib.pyplot as plt

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.lst_predictor.latent_predictor import LatentLSTPredictor, load_latent_lst_predictor
from model.utils.data_utils import collate_fn
from model.utils.vae_registry import VAERegistry
from model.utils.checkpoint import check_existing_paths
from helpers.load_configs import load_configs, add_config_arguments
from helpers.indexed_outputs import get_next_run_idx
from model.utils.statistics import compute_lst_statistic
from model.utils.samples import save_latent_visualization
from model.utils.plot import save_lst_error_histogram, save_lst_prediction_scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_latent_lst_predictor(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    statistic: str,
    region: str,
    lst_max: float,
    save_dir: str,
    mode: str,
    num_samples: int = 100,
    save_latents: bool = True,
    vae: torch.nn.Module = None
):
    """
    Validate latent LST predictor on validation set.
    
    Args:
        model: Trained LatentLSTPredictor
        val_loader: Validation data loader
        statistic: Target statistic ('p95', 'mean', etc.)
        region: Region for statistic computation ('full' or 'mask')
        lst_max: Max LST value for Celsius conversion
        save_dir: Directory to save results
        mode: 'semantic' or 'satellite'
        num_samples: Max number of samples to validate
        save_latents: Whether to save latent visualizations
        vae: VAE model for encoding full-res images (optional, needed if latents unavailable)
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_errors = []
    sample_latents = []
    
    samples_processed = 0
    
    print(f"\n{'='*60}")
    print(f"Validating Latent LST Predictor ({mode})")
    print(f"{'='*60}")
    print(f"Statistic: {statistic}")
    print(f"Region: {region}")
    print(f"Max samples: {num_samples}")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc="Validating")):
            if samples_processed >= num_samples:
                break
            
            # Extract data: (latent_or_image, cond_dict)
            if len(data) != 2:
                continue
            
            latent_or_image, cond_dict = data
            
            # Get LST full-res from conditioning
            if 'image' not in cond_dict or cond_dict['image'] is None:
                continue
            
            cond_image = cond_dict['image']
            meta = cond_dict.get('meta', {})
            
            if isinstance(meta, list) and len(meta) > 0:
                pixel_space_names = meta[0].get('pixel_space_names', [])
                needs_encoding = meta[0].get('needs_encoding', False)
            else:
                pixel_space_names = meta.get('pixel_space_names', [])
                needs_encoding = meta.get('needs_encoding', False)
            
            # Encode full-res image to latent if needed
            if needs_encoding:
                if vae is None:
                    raise RuntimeError(
                        "Dataset returned full-res images but no VAE provided for encoding. "
                        "Either provide pre-computed latents or pass VAE to validation function."
                    )
                full_res_image = latent_or_image.float().to(device)
                latent, _, _ = vae.encode(full_res_image)
            else:
                latent = latent_or_image
            
            # Find LST channel in conditioning
            lst_idx = None
            mask_idx = None
            
            for i, name in enumerate(pixel_space_names):
                if name == 'lst':
                    lst_idx = i
                elif name == 'inpainting_mask':
                    mask_idx = i
            
            if lst_idx is None:
                continue
            
            # Extract LST and mask
            lst_fullres = cond_image[:, lst_idx:lst_idx+1, :, :].float().to(device)
            
            mask = None
            if region == 'mask' and mask_idx is not None:
                mask = cond_image[:, mask_idx:mask_idx+1, :, :].float().to(device)
            
            # Compute target statistic
            target = compute_lst_statistic(lst_fullres, statistic=statistic, mask=mask)
            target = target.to(device)
            
            # Forward pass
            latent = latent.float().to(device)
            pred = model(latent)
            
            # Compute error
            error = torch.abs(pred - target)
            
            # Store results
            all_targets.extend(target.cpu().numpy().flatten())
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_errors.extend(error.cpu().numpy().flatten())
            
            # Save some latents for visualization
            if save_latents and len(sample_latents) < 16:
                sample_latents.append(latent[:min(4, latent.shape[0])].cpu())
            
            samples_processed += latent.shape[0]
    
    # Convert to numpy
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_errors = np.array(all_errors)
    
    # Compute metrics
    mae = all_errors.mean() * lst_max  # Convert to Celsius
    rmse = np.sqrt((all_errors ** 2).mean()) * lst_max
    r2 = 1 - np.sum((all_targets - all_predictions) ** 2) / (np.sum((all_targets - all_targets.mean()) ** 2) + 1e-8)
    
    # Percentile errors
    p50_err = np.percentile(all_errors, 50) * lst_max
    p95_err = np.percentile(all_errors, 95) * lst_max
    p99_err = np.percentile(all_errors, 99) * lst_max
    
    metrics = {
        'num_samples': len(all_targets),
        'mae_celsius': float(mae),
        'rmse_celsius': float(rmse),
        'r2': float(r2),
        'p50_error_celsius': float(p50_err),
        'p95_error_celsius': float(p95_err),
        'p99_error_celsius': float(p99_err),
        'target_mean_celsius': float(all_targets.mean() * lst_max),
        'target_std_celsius': float(all_targets.std() * lst_max),
        'pred_mean_celsius': float(all_predictions.mean() * lst_max),
        'pred_std_celsius': float(all_predictions.std() * lst_max),
    }
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"Validation Results ({mode})")
    print(f"{'='*60}")
    print(f"  Samples validated: {metrics['num_samples']}")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²: {r2:.4f}")
    print(f"  P50 Error: {p50_err:.2f}°C")
    print(f"  P95 Error: {p95_err:.2f}°C")
    print(f"  P99 Error: {p99_err:.2f}°C")
    print(f"  Target mean: {all_targets.mean() * lst_max:.1f}°C ± {all_targets.std() * lst_max:.1f}°C")
    print(f"  Pred mean: {all_predictions.mean() * lst_max:.1f}°C ± {all_predictions.std() * lst_max:.1f}°C")
    print(f"{'='*60}")
    
    # Save visualizations
    print("\nCreating visualizations...")
    
    # Scatter plot
    scatter_path = os.path.join(save_dir, f'scatter_{mode}.png')
    save_lst_prediction_scatter(
        all_targets, all_predictions, scatter_path, lst_max,
        title=f'Latent LST Predictor ({mode}): Target vs Prediction'
    )
    print(f"  ✓ Saved scatter plot: {scatter_path}")
    
    # Error histogram
    hist_path = os.path.join(save_dir, f'error_histogram_{mode}.png')
    save_lst_error_histogram(
        all_errors, hist_path, lst_max,
        title=f'Prediction Error Distribution ({mode})'
    )
    print(f"  ✓ Saved error histogram: {hist_path}")
    
    # Latent visualization
    if save_latents and sample_latents:
        latent_batch = torch.cat(sample_latents, dim=0)
        latent_vis_path = os.path.join(save_dir, f'latent_samples_{mode}.png')
        save_latent_visualization(latent_batch, latent_vis_path, n_samples=8)
        print(f"  ✓ Saved latent visualization: {latent_vis_path}")
    
    # Save metrics JSON
    metrics_path = os.path.join(save_dir, f'metrics_{mode}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved metrics: {metrics_path}")
    
    # Save raw predictions for further analysis
    results_path = os.path.join(save_dir, f'predictions_{mode}.npz')
    np.savez(
        results_path,
        targets=all_targets,
        predictions=all_predictions,
        errors=all_errors
    )
    print(f"  ✓ Saved raw predictions: {results_path}")
    
    return metrics


def main(args, config):
    """Main validation function."""
    
    # Extract configs
    data_config = config['data_config']
    train_config = config['train_params']
    predictor_config = config.get('latent_lst_predictor', {})
    vae_groups = config.get('vae_groups', {})
    layers_registry = config.get('layers', {})
    
    big_data_storage_path = data_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    mode = args.mode
    
    # Check for existing paths (for VAE checkpoints)
    existing_paths_result = check_existing_paths(
        train_config=train_config,
        mode=mode,
        type='lst_latent'
    )
    existing_vae_paths = existing_paths_result.vae_checkpoints
    
    print(f"\n{'='*60}")
    print(f"Latent LST Predictor Validation Setup")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Mode: {mode}")
    print(f"Max samples: {args.num_samples}")
    
    # Validate mode
    if mode not in vae_groups:
        raise ValueError(f"Mode '{mode}' not found in VAE groups. Available: {list(vae_groups.keys())}")
    
    # Get predictor config for this mode
    mode_config = predictor_config.get('modes', {}).get(mode, {})
    
    # Architecture params
    z_channels = mode_config.get('z_channels', vae_groups[mode].get('z_channels', 3))
    latent_size = mode_config.get('latent_size', 64)
    hidden_dims = predictor_config.get('hidden_dims', [64, 128, 256])
    
    # Target params
    statistic = predictor_config.get('statistic', 'p95')
    region = predictor_config.get('region', 'full')
    
    # Get LST normalization range
    lst_config = layers_registry.get('lst', {})
    lst_max = lst_config.get('normalize_params', {}).get('max', 80)
    
    # Set seed
    seed = train_config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"✓ Set random seed: {seed}")
    
    ########## Load Dataset #############
    print(f"\n{'='*60}")
    print(f"Loading Validation Dataset (mode: lst:{mode})")
    print(f"{'='*60}")
    
    # Check for existing cached patches
    existing_patches_path = existing_paths_result.patches_path
    if existing_patches_path is not None:
        cache_dir = Path(existing_patches_path)
    else:
        cache_dir = Path(big_data_storage_path) / "processed" / task_name / "patches"
    use_cached_patches = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    val_dataset = UrbanInpaintingDataset(
        split='val',
        mode=f'lst:{mode}',
        use_cached_patches=use_cached_patches,
        cache_dir=str(cache_dir)
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    ########## Load Model #############
    print(f"\n{'='*60}")
    print(f"Loading Latent LST Predictor ({mode})")
    print(f"{'='*60}")
    
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    
    # Use the load helper
    model = load_latent_lst_predictor(
        config=config,
        mode=mode,
        device=device,
        checkpoint_dir=data_dir
    )
    
    if model is None:
        checkpoint_name = mode_config.get('checkpoint_name', f'latent_lst_predictor_{mode}.pth')
        print(f"\n✗ Could not load model. Expected checkpoint at:")
        print(f"  {os.path.join(data_dir, checkpoint_name)}")
        print(f"  Please train the latent LST predictor first.")
        return None
    
    ########## Load VAE for on-the-fly encoding (if needed) #############
    # Check if dataset will return full-res images (needs_encoding)
    # This happens when validation latents don't exist
    vae = None
    
    # Check first sample to see if encoding is needed
    sample_data = val_dataset[0]
    if len(sample_data) == 2:
        _, sample_cond = sample_data
        sample_meta = sample_cond.get('meta', {})
        if isinstance(sample_meta, list) and len(sample_meta) > 0:
            needs_encoding = sample_meta[0].get('needs_encoding', False)
        else:
            needs_encoding = sample_meta.get('needs_encoding', False)
        
        if needs_encoding:
            print(f"\n{'='*60}")
            print(f"Loading VAE for on-the-fly encoding")
            print(f"{'='*60}")
            
            # Use VAERegistry for cleaner management
            vae_registry = VAERegistry(config, device)
            
            # Determine VAE checkpoint path (use existing_paths if available)
            vae_config = vae_groups.get(mode, {})
            if mode in existing_vae_paths:
                vae_ckpt_path = existing_vae_paths[mode]
            else:
                default_ckpt_name = vae_config.get('checkpoint_name', f'{mode}_vae_ckpt.pth')
                vae_ckpt_path = os.path.join(data_dir, default_ckpt_name)
            
            # Load VAE
            print(f"  - {mode.upper()} VAE for encoding")
            vae_registry.load_vae(
                group_name=mode,
                checkpoint_path=vae_ckpt_path,
                autoencoder_config=vae_config
            )
            vae = vae_registry.get_vae(mode)
            vae_registry.freeze(mode)
            
            if vae is not None:
                print(f"✓ Loaded and froze {mode} VAE for encoding validation samples")
            else:
                print(f"⚠ Could not load VAE for mode '{mode}' from {vae_ckpt_path}. Validation may fail.")
    
    ########## Setup Output Directory #############
    repo_dir = config.get('repo_dir', '.')
    save_dir = f"{repo_dir}/results/{task_name}/latent_lst_predictor_validation"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get run index
    if args.overwrite_samples:
        run_idx = 0
    else:
        run_idx = get_next_run_idx(save_dir, f'scatter_{mode}')
    
    # Create run-specific directory
    run_dir = os.path.join(save_dir, f'run_{run_idx:03d}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"✓ Output directory: {run_dir}")
    
    ########## Run Validation #############
    metrics = validate_latent_lst_predictor(
        model=model,
        val_loader=val_loader,
        statistic=statistic,
        region=region,
        lst_max=lst_max,
        save_dir=run_dir,
        mode=mode,
        num_samples=args.num_samples,
        save_latents=args.save_latents,
        vae=vae
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Validation Complete!")
    print(f"  Results saved to: {run_dir}")
    print(f"{'='*60}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Latent LST Predictor')
    
    add_config_arguments(parser)
    
    parser.add_argument('--mode', type=str, default='semantic',
                       choices=['semantic', 'satellite'],
                       help='VAE group mode to validate (semantic or satellite)')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Maximum number of validation samples to process')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--overwrite_samples', action='store_true',
                       help='Overwrite existing validation results (use run_idx=0)')
    parser.add_argument('--save_latents', action='store_true', default=True,
                       help='Save latent visualizations (default: True)')
    parser.add_argument('--no_save_latents', action='store_false', dest='save_latents',
                       help='Do not save latent visualizations')
    
    args = parser.parse_args()
    
    config = load_configs()
    
    main(args, config)

"""
==============================================================================
Validation script for the Temperature predictor

Creates prediction samples on validation set 
to assess Temperature predictor quality.
==============================================================================
"""

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import numpy as np
from tqdm import tqdm

# Data handling
import torch
from torchvision.utils import make_grid, save_image

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.temperature_predictor.predictor import TemperaturePredictor
from model.utils.layer_config import get_layer_channels_from_names
from model.utils.colors import get_colormap_for_layer, apply_colormap_to_tensor
from helpers.load_configs import load_configs, add_config_arguments
from helpers.indexed_outputs import get_next_run_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_temperature_prediction_visualization(
    semantic_input: torch.Tensor,
    temperature_target: torch.Tensor,
    temperature_pred: torch.Tensor,
    semantic_layer_names: list,
    save_dir: str,
    sample_idx: int,
    n_samples: int = 8
):
    """
    Save Temperature prediction visualization with semantic inputs, Target Temperature, and predicted Temperature.
    
    Args:
        semantic_input: Semantic input tensor [B, C_semantic, H, W]
        temperature_target: Target Temperature tensor [B, 1, H, W]
        temperature_pred: Predicted Temperature tensor [B, 1, H, W]
        semantic_layer_names: List of semantic layer names
        save_dir: Directory to save visualizations
        sample_idx: Sample index for filename
        n_samples: Number of samples to visualize
    """
    n_samples = min(n_samples, semantic_input.shape[0])
    
    # Get Temperature colormap (rocket: cool to hot)
    temperature_cmap = get_colormap_for_layer('temperature')
    
    # Temperature is already normalized to [0, 1] by dataset (0°C = 0, 80°C = 1)
    temperature_target_norm = torch.clamp(temperature_target, 0, 1)
    temperature_pred_norm = torch.clamp(temperature_pred, 0, 1)
    
    # Apply colormap to Temperature
    temperature_target_colored = apply_colormap_to_tensor(temperature_target_norm[:n_samples], temperature_cmap)
    temperature_pred_colored = apply_colormap_to_tensor(temperature_pred_norm[:n_samples], temperature_cmap)
    
    # Compute error map (error is also in normalized [0, 1] range)
    error = torch.abs(temperature_pred - temperature_target)
    error_norm = torch.clamp(error / 0.25, 0, 1)  # Normalize error: 0.25 = 20°C/80°C max visible error
    error_colored = apply_colormap_to_tensor(error_norm[:n_samples], 'hot')
    
    # Save Temperature comparison (target vs prediction)
    temperature_comparison = torch.cat([temperature_target_colored, temperature_pred_colored], dim=0)
    grid = make_grid(temperature_comparison, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_temperature_comparison.png')
    save_image(grid, save_path)
    
    # Save error map
    grid_error = make_grid(error_colored, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
    save_path_error = os.path.join(save_dir, f'sample_{sample_idx}_temperature_error.png')
    save_image(grid_error, save_path_error)
    
    # Save semantic inputs (layer-wise)
    for ch_idx, layer_name in enumerate(semantic_layer_names):
        if ch_idx >= semantic_input.shape[1]:
            break
        
        semantic_ch = semantic_input[:n_samples, ch_idx:ch_idx+1, :, :]
        
        # Normalize to [0, 1]
        semantic_ch_norm = (semantic_ch - semantic_ch.min()) / (semantic_ch.max() - semantic_ch.min() + 1e-6)
        
        # Apply colormap based on layer type
        if 'vegetation' in layer_name.lower() or 'ndvi' in layer_name.lower():
            semantic_ch_colored = apply_colormap_to_tensor(semantic_ch_norm, get_colormap_for_layer('vegetation'))
        elif 'height' in layer_name.lower():
            semantic_ch_colored = apply_colormap_to_tensor(semantic_ch_norm, get_colormap_for_layer('buildings_heights'))
        else:
            # Binary layers (buildings, streets) - use grayscale
            semantic_ch_colored = semantic_ch_norm.repeat(1, 3, 1, 1)
        
        grid_semantic = make_grid(semantic_ch_colored, nrow=n_samples, normalize=False, padding=2, pad_value=1.0)
        save_path_semantic = os.path.join(save_dir, f'sample_{sample_idx}_semantic_{layer_name}.png')
        save_image(grid_semantic, save_path_semantic)


def validate_temperature_predictor(
    model,
    dataset,
    semantic_layers,
    include_ndvi,
    num_samples=8,
    save_dir=None,
    overwrite_samples=False,
    temp_max_celsius=80
):
    """
    Validate Temperature predictor by creating prediction samples on validation set.
    
    Args:
        model: Trained TemperaturePredictor model
        dataset: UrbanInpaintingDataset in validation mode (default mode)
        semantic_layers: List of semantic layer names
        include_ndvi: Whether NDVI is included in input
        num_samples: Number of validation samples to process
        save_dir: Directory to save samples
        overwrite_samples: Whether to overwrite existing samples
        temp_max_celsius: Maximum Temperature value in Celsius for normalization (from config)
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    print("\n" + "="*60)
    print("Temperature predictor Validation")
    print("="*60)
    print(f"Semantic layers: {semantic_layers}")
    print(f"Include NDVI: {include_ndvi}")
    print(f"Number of samples: {num_samples}")
    print(f"Split: validation")
    
    # Limit number of samples to dataset size
    num_samples = min(num_samples, len(dataset))
    
    # Get random validation samples
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"\n✓ Selected {num_samples} random validation samples")
    print(f"  Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    
    # Setup output directories
    if save_dir is None:
        save_dir = "./results/temperature_predictor_validation"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get next run index
    base_name = 'temperature_predictor_val'
    run_idx = get_next_run_idx(save_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*60}")
    
    # Create samples directory
    samples_dir = save_dir
    os.makedirs(samples_dir, exist_ok=True)
    
    ################# Validation Loop ########################
    print("\n" + "="*60)
    print("Starting Temperature predictor Validation")
    print("="*60)
    
    all_errors = []
    all_mae = []
    all_rmse = []
    
    all_semantic_inputs = []
    all_temperature_targets = []
    all_temperature_preds = []
    
    with torch.no_grad():
        for sample_idx, data_idx in enumerate(tqdm(sample_indices, desc="Validating")):
            # Load sample from dataset (default mode)
            data = dataset[data_idx]
            
            # Extract data from batch (default mode returns: rgb_image, cond_dict)
            if len(data) == 2:
                rgb_image, cond_dict = data
                
                # Extract conditioning image (contains all non-RGB layers)
                if 'image' not in cond_dict or cond_dict['image'] is None:
                    print(f"⚠ Warning: No conditioning channels in sample {data_idx}")
                    continue
                
                cond_image = cond_dict['image']
                
                # Extract metadata
                meta = cond_dict.get('meta', {})
                if isinstance(meta, list) and len(meta) > 0:
                    channel_names = meta[0].get('channel_names', [])
                    layer_names = meta[0].get('layer_names', [])
                elif isinstance(meta, dict):
                    channel_names = meta.get('channel_names', [])
                    layer_names = meta.get('layer_names', [])
                else:
                    channel_names = []
                    layer_names = []
            else:
                print(f"⚠ Warning: Unexpected data format in sample {data_idx}")
                continue
            
            if not channel_names:
                print(f"⚠ Warning: No channel names in sample {data_idx}")
                continue
            
            # Build semantic input tensor from semantic layers
            semantic_tensor_list = []
            for layer_name in semantic_layers:
                layer_matches = get_layer_channels_from_names(channel_names, layer_name)
                if not layer_matches:
                    # Layer not found - create zero channel
                    if len(semantic_tensor_list) > 0:
                        _, _, H, W = semantic_tensor_list[0].shape
                    else:
                        _, H, W = 1, cond_image.shape[1], cond_image.shape[2]
                    semantic_tensor_list.append(torch.zeros(1, 1, H, W, device=cond_image.device))
                    continue
                
                # Add all channels for this layer
                for idx, ch_name in layer_matches:
                    semantic_tensor_list.append(cond_image[idx:idx+1, :, :].unsqueeze(0))
            
            if not semantic_tensor_list:
                print(f"⚠ Warning: No semantic layers found in sample {data_idx}")
                continue
            
            # Extract Temperature target from conditioning
            temperature_target = None
            temperature_matches = get_layer_channels_from_names(channel_names, 'temperature')
            if temperature_matches:
                idx, _ = temperature_matches[0]
                temperature_target = cond_image[idx:idx+1, :, :].unsqueeze(0)
            
            if temperature_target is None:
                # Try alternative names
                for idx, ch_name in enumerate(channel_names):
                    if 'landsat_surface_temp' in ch_name.lower() or 'surface_temp' in ch_name.lower():
                        temperature_target = cond_image[idx:idx+1, :, :].unsqueeze(0)
                        break
            
            if temperature_target is None:
                print(f"⚠ Warning: Temperature target not found in sample {data_idx}")
                continue
            
            # Extract NDVI if needed
            ndvi_channel = None
            if include_ndvi:
                ndvi_matches = get_layer_channels_from_names(channel_names, 'ndvi')
                if ndvi_matches:
                    idx, _ = ndvi_matches[0]
                    ndvi_channel = cond_image[idx:idx+1, :, :].unsqueeze(0)
            
            # Build semantic input
            semantic_input = torch.cat(semantic_tensor_list, dim=1)  # [1, C_semantic, H, W]
            
            # Add NDVI if needed
            if include_ndvi:
                if ndvi_channel is not None:
                    semantic_input = torch.cat([semantic_input, ndvi_channel], dim=1)
                else:
                    # Create zero NDVI channel
                    B, _, H, W = semantic_input.shape
                    semantic_input = torch.cat([semantic_input, torch.zeros(B, 1, H, W, device=semantic_input.device)], dim=1)
            
            # Move to device
            semantic_input = semantic_input.float().to(device)
            temperature_target = temperature_target.float().to(device)
            
            # Forward pass
            temperature_pred = model(semantic_input)
            
            # Compute metrics (convert from normalized [0, 1] to Celsius for interpretability)
            # Temperature normalization: 0°C = 0, max°C = 1 (max from config)
            error = torch.abs(temperature_pred - temperature_target)
            mae = error.mean().item() * temp_max_celsius  # Convert to °C
            rmse = torch.sqrt((error ** 2).mean()).item() * temp_max_celsius  # Convert to °C
            
            all_errors.append(error.cpu())
            all_mae.append(mae)
            all_rmse.append(rmse)
            
            # Save individual sample
            sample_pt_path = os.path.join(samples_dir, f'sample_{sample_idx}.pt')
            torch.save({
                'semantic_input': semantic_input[0].cpu(),
                'temperature_target': temperature_target[0].cpu(),
                'temperature_pred': temperature_pred[0].cpu(),
                'error': error[0].cpu(),
                'mae': mae,
                'rmse': rmse,
                'semantic_layer_names': semantic_layers,
                'data_index': data_idx,
            }, sample_pt_path)
            
            # Keep for batch visualization
            all_semantic_inputs.append(semantic_input)
            all_temperature_targets.append(temperature_target)
            all_temperature_preds.append(temperature_pred)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    if all_semantic_inputs:
        # Stack all samples
        semantic_batch = torch.cat(all_semantic_inputs, dim=0)  # [N, C_semantic, H, W]
        temperature_target_batch = torch.cat(all_temperature_targets, dim=0)  # [N, 1, H, W]
        temperature_pred_batch = torch.cat(all_temperature_preds, dim=0)  # [N, 1, H, W]
        
        # Save batch visualization
        save_temperature_prediction_visualization(
            semantic_input=semantic_batch,
            temperature_target=temperature_target_batch,
            temperature_pred=temperature_pred_batch,
            semantic_layer_names=semantic_layers,
            save_dir=save_dir,
            sample_idx=run_idx,
            n_samples=len(all_semantic_inputs)
        )
        print(f"✓ Saved prediction visualizations to {save_dir}")
    
    # Compute overall metrics
    overall_mae = np.mean(all_mae)
    overall_rmse = np.mean(all_rmse)
    
    # Save metrics
    metrics = {
        'mae': overall_mae,
        'rmse': overall_rmse,
        'per_sample_mae': all_mae,
        'per_sample_rmse': all_rmse,
        'num_samples': len(all_mae)
    }
    
    metrics_path = os.path.join(samples_dir, f'metrics_run_{run_idx}.pt')
    torch.save(metrics, metrics_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Validation Complete!")
    print(f"  Processed {len(sample_indices)} validation samples")
    print(f"  Overall MAE: {overall_mae:.4f}°C")
    print(f"  Overall RMSE: {overall_rmse:.4f}°C")
    print(f"  Saved to: {samples_dir}")
    print(f"{'='*60}")
    
    return metrics


def main(args, config):
    """Main validation function"""
    
    # Extract configs
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    vae_groups = config.get('vae_groups', {})
    layers_registry = config.get('layers', {})
    
    big_data_storage_path = dataset_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    print(f"\n{'='*60}")
    print(f"Temperature predictor Validation Setup")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Number of samples: {args.num_samples}")
    
    # Get semantic layers from config
    if 'semantic' not in vae_groups:
        raise ValueError("Config must define 'semantic' VAE group for Temperature predictor validation")
    
    semantic_vae_config = vae_groups['semantic']
    semantic_layers = semantic_vae_config.get('layers', [])
    
    if not semantic_layers:
        raise ValueError("Semantic VAE group has no layers defined")
    
    # Get Temperature normalization range from config for metric conversion
    temperature_layer_config = layers_registry.get('temperature', {})
    temperature_normalize_params = temperature_layer_config.get('normalize_params', {})
    temp_max_celsius = temperature_normalize_params.get('max', 80)  # Default to 80°C if not specified
    
    # Count channels in semantic layers
    num_semantic_channels = 0
    for layer_name in semantic_layers:
        if layer_name not in layers_registry:
            raise ValueError(f"Layer '{layer_name}' not found in layers registry")
        layer_config = layers_registry[layer_name]
        channels = layer_config.get('channels', None)
        if channels:
            num_semantic_channels += len(channels)
        else:
            num_semantic_channels += 1  # Binary or single-channel layer
    
    # NDVI configuration
    include_ndvi = train_config.get('temperature_predictor_use_ndvi', True)
    if include_ndvi:
        num_input_channels = num_semantic_channels + 1
    else:
        num_input_channels = num_semantic_channels
    
    # Set seed for reproducibility
    seed = train_config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"✓ Set random seed: {seed}")
    
    ########## Load Dataset #############
    print("\n" + "="*60)
    print("Loading Validation Dataset")
    print("="*60)
    
    # Load dataset in default mode for validation split
    dataset = UrbanInpaintingDataset(
        split='val',
        mode='default',  # Get all layers: RGB as image, rest as conditioning
        use_cached_patches=True,
    )
    
    print(f"✓ Loaded {len(dataset)} validation patches")
    
    ########## Load Temperature predictor #############
    print("\n" + "="*60)
    print("Loading Temperature predictor Model")
    print("="*60)
    
    # Get checkpoint path
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    checkpoint_name = train_config.get('temperature_predictor_ckpt_name', 'temperature_predictor_best.pth')
    checkpoint_path = os.path.join(data_dir, checkpoint_name)
    
    print(f"Checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Temperature predictor checkpoint not found: {checkpoint_path}")
        print(f"  Please train the Temperature predictor first before validation")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    ckpt_config = checkpoint.get('config', {})
    hidden_dims = ckpt_config.get('hidden_dims', train_config.get('temperature_predictor_hidden_dims', [64, 128, 256]))
    
    # Create model
    model = TemperaturePredictor(
        in_channels=num_input_channels,
        hidden_dims=hidden_dims,
        out_channels=1
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded Temperature predictor from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"  Input channels: {num_input_channels}")
    print(f"  Hidden dims: {hidden_dims}")
    
    # Setup output directory
    repo_dir = config.get('repo_dir', '.')
    save_dir = f"{repo_dir}/results/{task_name}/temperature_predictor_validation"
    
    ########## Validate Temperature predictor #############
    metrics = validate_temperature_predictor(
        model=model,
        dataset=dataset,
        semantic_layers=semantic_layers,
        include_ndvi=include_ndvi,
        num_samples=args.num_samples,
        save_dir=save_dir,
        overwrite_samples=args.overwrite_samples,
        temp_max_celsius=temp_max_celsius
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Temperature predictor with prediction samples')
    
    add_config_arguments(parser)
    
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of validation samples to process')
    parser.add_argument('--overwrite_samples', action='store_true',
                       help='Overwrite existing validation samples (use run_idx=0)')
    
    args = parser.parse_args()
    
    config = load_configs()
    
    main(args, config)

# Validation script to test dataset loading and visualize samples
##### Import libraries #####
# Standard libraries
import sys
import os
import yaml
import argparse

# Data handling
import numpy as np

# Data Science/ML libraries
import torch
from torchvision.utils import make_grid

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import seaborn as sns

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.colors import get_colormap_for_layer
from helpers.load_configs import load_configs


def visualize_sample(sample_data, mode='default', save_path=None):
    """
    Visualize a dataset sample with all its components.
    
    Args:
        sample_data: Tuple from dataset - format depends on mode
        mode: Dataset mode string (e.g., 'default', 'vae:semantic', 'diffusion:satellite')
        save_path: Optional path to save visualization
    """
    # All modes return (tensor, dict)
    im, cond_dict = sample_data
    
    # Parse mode to determine what we're visualizing
    if mode != 'default':
        mode_parts = mode.split(':')
        mode_type = mode_parts[0]
        mode_target = mode_parts[1] if len(mode_parts) > 1 else None
    else:
        mode_type = 'default'
        mode_target = None
    
    
    # Convert tensors to numpy
    im_np = im.numpy()
    
    # Extract metadata and conditioning
    meta = cond_dict.get('meta', {})
    image_meta = cond_dict.get('image_meta', {})
    
    # For default mode: image_meta has main tensor info (layer_names, channel_names), meta has conditioning info
    # For VAE mode: image_meta has main tensor info (target_layer_names, target_channel_names)
    # For other modes: meta has everything
    if image_meta:
        # VAE or default mode - check which keys are present
        main_layer_names = image_meta.get('target_layer_names', image_meta.get('layer_names', []))
        main_channel_names = image_meta.get('target_channel_names', image_meta.get('channel_names', []))
    else:
        # Fallback to meta
        main_layer_names = meta.get('layer_names', [])
        main_channel_names = meta.get('channel_names', [])
    
    cond_layer_names = meta.get('layer_names', [])
    cond_channel_names = meta.get('channel_names', [])
    
    cond_image = cond_dict.get('image', None)
    
    # Determine what to visualize
    num_main_channels = im_np.shape[0]
    num_cond_channels = 0 if cond_image is None else cond_image.shape[0]
    
    # Calculate grid layout
    total_viz = num_main_channels + num_cond_channels
    n_cols = min(3, total_viz)
    n_rows = (total_viz + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    idx = 0  # Track position in axes array
    
    # Visualize main tensor (RGB, multi-channel, or latent)
    is_rgb = (num_main_channels == 3 and mode_type == 'default')
    is_latent = (mode_type == 'diffusion')
    
    if is_rgb:
        # RGB visualization
        rgb = im_np.transpose(1, 2, 0)
        rgb = (rgb + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        rgb = np.clip(rgb, 0, 1)
        
        # Enhance contrast
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        rgb = np.power(rgb, 0.7)  # Gamma correction
        
        axes[idx].imshow(rgb)
        axes[idx].set_title('Target RGB', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
        
    elif is_latent:
        # Latent space visualization (show first few channels)
        num_to_show = min(6, num_main_channels)
        for i in range(num_to_show):
            if idx >= len(axes):
                break
            axes[idx].imshow(im_np[i], cmap='viridis')
            axes[idx].set_title(f'Latent Ch {i}', fontsize=10)
            axes[idx].axis('off')
            idx += 1
    
    else:
        # Multi-channel visualization (VAE mode)
        for i in range(num_main_channels):
            if idx >= len(axes):
                break
            channel_name = main_channel_names[i] if i < len(main_channel_names) else f'Channel {i}'
            cmap = get_colormap_for_layer(channel_name)
            axes[idx].imshow(im_np[i], cmap=cmap)
            axes[idx].set_title(f'{channel_name}', fontsize=10)
            axes[idx].axis('off')
            idx += 1
                
    
    # Visualize conditioning channels
    if cond_image is not None:
        cond_np = cond_image.numpy()
        # Get conditioning channel names from metadata
        # The dataset already provides the correct channel_names in the metadata
        if 'pixel_space_names' in cond_dict:
            # Diffusion mode: explicit pixel space conditioning names
            cond_channel_names = cond_dict['pixel_space_names']
        else:
            # Default/VAE mode: use main_channel_names for conditioning display
            # (image_meta has main tensor names, meta has conditioning names)
            if image_meta:
                cond_channel_names = cond_channel_names  # From meta
            else:
                cond_channel_names = main_channel_names  # Fallback
        
        for i in range(cond_np.shape[0]):
            if idx >= len(axes):
                break
            channel_name = cond_channel_names[i] if i < len(cond_channel_names) else f'Cond {i}'
            cmap = get_colormap_for_layer(channel_name)
            axes[idx].imshow(cond_np[i], cmap=cmap)
            axes[idx].set_title(f'{channel_name}', fontsize=10)
            axes[idx].axis('off')
            idx += 1
    
    # Visualize latent-space conditioning (diffusion mode)
    if mode_type == 'diffusion':
        for key, value in cond_dict.items():
            if key not in ['meta', 'image', 'pixel_space_names'] and isinstance(value, torch.Tensor):
                if idx >= len(axes):
                    break
                latent_cond = value.numpy()
                if latent_cond.ndim >= 3:
                    axes[idx].imshow(latent_cond[0], cmap='viridis')
                    axes[idx].set_title(f'Latent: {key}', fontsize=10)
                    axes[idx].axis('off')
                    idx += 1
    
    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    # Print metadata
    print("\n" + "="*60)
    print(f"Sample Metadata ({mode}):")
    print("="*60)
    for key, value in meta.items():
        if key not in ['layer_names', 'channel_names']:
            print(f"  {key}: {value}")
    
    # Print main tensor channels
    if main_channel_names:
        print(f"\n  Main Tensor Channels ({len(main_channel_names)}):")
        for i, (layer, channel) in enumerate(zip(main_layer_names[:10], main_channel_names[:10])):
            print(f"    [{i}] {layer}: {channel}")
        if len(main_channel_names) > 10:
            print(f"    ... and {len(main_channel_names) - 10} more")
    
    # Print conditioning channels if different from main
    if cond_channel_names and cond_channel_names != main_channel_names:
        print(f"\n  Conditioning Channels ({len(cond_channel_names)}):")
        for i, (layer, channel) in enumerate(zip(cond_layer_names[:10], cond_channel_names[:10])):
            print(f"    [{i}] {layer}: {channel}")
        if len(cond_channel_names) > 10:
            print(f"    ... and {len(cond_channel_names) - 10} more")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def validate_dataset(
    num_samples=5, 
    config=None, 
    mode='default', 
    use_cached_patches=False, 
    recompute_layer_stats=False, 
    plot_samples=True
    ):
    """
    Validate dataset loading and visualize samples.
    
    Args:
        num_samples: Number of samples to visualize
        config: Configuration dict (if None, will load from load_configs)
        mode: Dataset mode - 'default', 'vae:<group>', or 'diffusion:<stage>'
        use_cached_patches: Whether to use cached patches (default: False for validation)
        recompute_layer_stats: Whether to recompute layer statistics (default: False)
        plot_samples: Whether to plot sample visualizations (default: True)
    """
    print("="*60)
    print(f"Dataset Validation - Mode: {mode}")
    print("="*60)
    
    ###### setup config variables #######
    if config is None:
        config = load_configs()
    data_config = config['data_config']
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    train_config = config['train_params']
    
    print(f"\n✓ Loaded configuration")
    
    # Create dataset
    print(f"\nLoading dataset with mode '{mode}'...")
    print(f"Using {'cached patches' if use_cached_patches else 'on-the-fly loading'}")
    try:
        dataset = UrbanInpaintingDataset(
            split='train',
            use_cached_patches=use_cached_patches,
            mode=mode,
            recompute_layer_stats=recompute_layer_stats
        )
        print(f"✓ Successfully loaded dataset!")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Dataset info
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    print(f"  Mode: {dataset.mode}")
    print(f"  Number of patches: {len(dataset)}")
    print(f"  Patch size: {dataset.patch_size}x{dataset.patch_size}")
    print(f"  Regions: {dataset.regions}")
    
    # Test loading samples
    print("\n" + "="*60)
    print("Testing Sample Loading")
    print("="*60)
    
    # Create output directory
    task_name = train_config.get('task_name', 'urban_inpainting')
    mode_safe = mode.replace(':', '_')
    output_dir = f"{big_data_storage_path}/results/{task_name}/dataset_validation_{mode_safe}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        try:
            sample = dataset[i]
            im, cond_dict = sample
            
            print(f"  Main tensor shape: {im.shape}")
            print(f"  Main tensor range: [{im.min():.3f}, {im.max():.3f}]")
            
            # Extract per-layer statistics
            meta = cond_dict.get('meta', {})
            image_meta = cond_dict.get('image_meta', {})
            
            # For default mode: image_meta has main tensor info (layer_names, channel_names), meta has conditioning info
            # For VAE mode: image_meta has main tensor info (target_layer_names, target_channel_names)
            # For other modes: meta has everything
            if image_meta:
                # VAE or default mode - check which keys are present
                main_layer_names = image_meta.get('target_layer_names', image_meta.get('layer_names', []))
                main_channel_names = image_meta.get('target_channel_names', image_meta.get('channel_names', []))
            else:
                # Fallback to meta
                main_layer_names = meta.get('layer_names', [])
                main_channel_names = meta.get('channel_names', [])
            
            cond_layer_names = meta.get('layer_names', [])
            cond_channel_names = meta.get('channel_names', [])
            
            if main_layer_names and main_channel_names:
                print(f"\n  Per-layer statistics (Main tensor):")
                # Group channels by layer
                layer_groups = {}
                for ch_idx, (layer_name, channel_name) in enumerate(zip(main_layer_names, main_channel_names)):
                    if layer_name not in layer_groups:
                        layer_groups[layer_name] = []
                    layer_groups[layer_name].append(ch_idx)
                
                # Extract statistics for each layer in main tensor
                for layer_name, channel_indices in layer_groups.items():
                    num_channels = len(channel_indices)
                    start_idx = channel_indices[0]
                    end_idx = channel_indices[-1] + 1
                    
                    layer_data = im[start_idx:end_idx]
                    
                    min_val = layer_data.min().item()
                    max_val = layer_data.max().item()
                    mean_val = layer_data.mean().item()
                    std_val = layer_data.std().item()
                    
                    print(f"    {layer_name:20s} [{num_channels}ch]: shape={layer_data.shape}, "
                          f"range=[{min_val:7.3f}, {max_val:7.3f}], "
                          f"mean={mean_val:7.3f}, std={std_val:6.3f}")
            
            if 'image' in cond_dict and cond_dict['image'] is not None:
                cond_image = cond_dict['image']
                print(f"\n  Conditioning tensor shape: {cond_image.shape}")
                print(f"  Conditioning range: [{cond_image.min():.3f}, {cond_image.max():.3f}]")
                
                # For diffusion mode, use pixel_space_names if available
                if 'pixel_space_names' in cond_dict:
                    pixel_space_names = cond_dict['pixel_space_names']
                    print(f"  Per-layer statistics (Conditioning - pixel space):")
                    
                    # Group channels by layer name
                    layer_groups = {}
                    for ch_idx, name in enumerate(pixel_space_names):
                        # Extract layer name (before any suffix like _ch0)
                        layer_name = name.split('_ch')[0] if '_ch' in name else name
                        if layer_name not in layer_groups:
                            layer_groups[layer_name] = []
                        layer_groups[layer_name].append(ch_idx)
                    
                    for layer_name, channel_indices in layer_groups.items():
                        num_channels = len(channel_indices)
                        # Use actual indices, not sequential
                        layer_data = cond_image[channel_indices]
                        
                        if layer_data.numel() > 0:
                            min_val = layer_data.min().item()
                            max_val = layer_data.max().item()
                            mean_val = layer_data.mean().item()
                            std_val = layer_data.std().item()
                            
                            print(f"    {layer_name:20s} [{num_channels}ch]: shape={layer_data.shape}, "
                                  f"range=[{min_val:7.3f}, {max_val:7.3f}], "
                                  f"mean={mean_val:7.3f}, std={std_val:6.3f}")
                        else:
                            print(f"    {layer_name:20s} [{num_channels}ch]: [empty tensor]")
                
                # Print per-layer stats for conditioning layers (non-diffusion modes)
                elif cond_layer_names and cond_channel_names:
                    print(f"  Per-layer statistics (Conditioning):")
                    
                    # Group channels by layer
                    layer_groups = {}
                    for ch_idx, (layer_name, channel_name) in enumerate(zip(cond_layer_names, cond_channel_names)):
                        if layer_name not in layer_groups:
                            layer_groups[layer_name] = []
                        layer_groups[layer_name].append(ch_idx)
                    
                    # Extract statistics for each layer in conditioning tensor
                    for layer_name, channel_indices in layer_groups.items():
                        num_channels = len(channel_indices)
                        start_idx = channel_indices[0]
                        end_idx = channel_indices[-1] + 1
                        
                        # Bounds check
                        if end_idx > cond_image.shape[0]:
                            print(f"    {layer_name:20s} [{num_channels}ch]: [skipped - indices {start_idx}:{end_idx} out of bounds for shape {cond_image.shape}]")
                            continue
                        
                        layer_data = cond_image[start_idx:end_idx]
                        
                        if layer_data.numel() > 0:
                            min_val = layer_data.min().item()
                            max_val = layer_data.max().item()
                            mean_val = layer_data.mean().item()
                            std_val = layer_data.std().item()
                            
                            print(f"    {layer_name:20s} [{num_channels}ch]: shape={layer_data.shape}, "
                                  f"range=[{min_val:7.3f}, {max_val:7.3f}], "
                                  f"mean={mean_val:7.3f}, std={std_val:6.3f}")
                        else:
                            print(f"    {layer_name:20s} [{num_channels}ch]: [empty tensor]")
            
            # Print latent-space conditioning stats (diffusion mode)
            for key, value in cond_dict.items():
                if key not in ['meta', 'image', 'image_meta', 'pixel_space_names'] and isinstance(value, torch.Tensor):
                    if value.ndim >= 2:
                        print(f"\n  Latent conditioning '{key}':")
                        print(f"    Shape: {value.shape}")
                        print(f"    Range: [{value.min():.3f}, {value.max():.3f}]")
                        print(f"    Mean: {value.mean():.3f}, Std: {value.std():.3f}")
            
            # Visualize
            if plot_samples:
                save_path = os.path.join(output_dir, f'sample_{i}.png')
                visualize_sample(sample, mode=mode, save_path=save_path)
            
            print(f"  ✓ Sample {i+1} validated successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print(f"\nVisualizations saved to: {output_dir}")
    
    # clean up
    dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate urban inpainting dataset')
    
    from helpers.load_configs import add_config_arguments
    add_config_arguments(parser)
    
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--mode', type=str, default='default',
                       help='Dataset mode: "default", "vae:<group>", or "diffusion:<stage>"')
    parser.add_argument('--use_cached_patches', action='store_true',
                       help='Use cached patches instead of on-the-fly loading (default: False)')
    parser.add_argument('--recompute_layer_stats', action='store_true',
                       help='Recompute layer statistics even if cached stats exist (default: False)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable sample visualizations (default: plots are enabled)')
    
    args = parser.parse_args()
    config = load_configs(parser)
    validate_dataset(
        num_samples=args.num_samples, 
        config=config, 
        mode=args.mode, 
        use_cached_patches=args.use_cached_patches, 
        recompute_layer_stats=args.recompute_layer_stats, 
        plot_samples=not args.no_plots
    )
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
    im, output_dict = sample_data
    
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
    meta = output_dict.get('meta', {})
    layer_names = meta.get('layer_names', [])
    channel_names = meta.get('channel_names', [])
    cond_image = output_dict.get('image', None)
    
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
            channel_name = channel_names[i] if i < len(channel_names) else f'Channel {i}'
            cmap = _get_colormap_for_channel(channel_name)
            axes[idx].imshow(im_np[i], cmap=cmap)
            axes[idx].set_title(f'{channel_name}', fontsize=10)
            axes[idx].axis('off')
            idx += 1
                
    
    # Visualize conditioning channels
    if cond_image is not None:
        cond_np = cond_image.numpy()
        # Get conditioning channel names from metadata
        # The dataset already provides the correct channel_names in the metadata
        if 'pixel_space_names' in output_dict:
            # Diffusion mode: explicit pixel space conditioning names
            cond_channel_names = output_dict['pixel_space_names']
        else:
            # Default/VAE mode: channel_names in meta already filtered for conditioning
            cond_channel_names = channel_names
        
        for i in range(cond_np.shape[0]):
            if idx >= len(axes):
                break
            channel_name = cond_channel_names[i] if i < len(cond_channel_names) else f'Cond {i}'
            cmap = _get_colormap_for_channel(channel_name)
            axes[idx].imshow(cond_np[i], cmap=cmap)
            axes[idx].set_title(f'{channel_name}', fontsize=10)
            axes[idx].axis('off')
            idx += 1
    
    # Visualize latent-space conditioning (diffusion mode)
    if mode_type == 'diffusion':
        for key, value in output_dict.items():
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
    if channel_names:
        print(f"\n  Channels ({len(channel_names)}):")
        for i, (layer, channel) in enumerate(zip(layer_names[:10], channel_names[:10])):
            print(f"    [{i}] {layer}: {channel}")
        if len(channel_names) > 10:
            print(f"    ... and {len(channel_names) - 10} more")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _get_colormap_for_channel(channel_name: str):
    """Get appropriate colormap based on channel name."""
    name_lower = channel_name.lower()
    
    if 'mask' in name_lower:
        colors = [(0, 0, 0), (1, 1, 1)]
        return LinearSegmentedColormap.from_list('binary', colors, N=100)
    elif 'temp' in name_lower or 'lst' in name_lower:
        return sns.color_palette("rocket", as_cmap=True)
    elif 'vegetation' in name_lower or 'ndvi' in name_lower:
        rdylgn = cm.get_cmap('RdYlGn', 256)
        newcolors = rdylgn(np.linspace(0.1, 1, 256))
        newcolors[0] = [0, 0, 0, 1]
        return ListedColormap(newcolors)
    elif 'height' in name_lower:
        return sns.color_palette("rocket", as_cmap=True)
    else:
        return 'gray'


def validate_dataset(num_samples=5, config=None, mode='default'):
    """
    Validate dataset loading and visualize samples.
    
    Args:
        num_samples: Number of samples to visualize
        config: Configuration dict (if None, will load from load_configs)
        mode: Dataset mode - 'default', 'vae:<group>', or 'diffusion:<stage>'
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
    try:
        dataset = UrbanInpaintingDataset(
            split='train',
            use_cached_patches=False,  # Use on-the-fly for validation
            mode=mode
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
            im, output_dict = sample
            
            print(f"  Main tensor shape: {im.shape}")
            print(f"  Main tensor range: [{im.min():.3f}, {im.max():.3f}]")
            
            if 'image' in output_dict and output_dict['image'] is not None:
                print(f"  Conditioning shape: {output_dict['image'].shape}")
                print(f"  Conditioning range: [{output_dict['image'].min():.3f}, {output_dict['image'].max():.3f}]")
            
            # Visualize
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate urban inpainting dataset')
    
    from helpers.load_configs import add_config_arguments
    add_config_arguments(parser)
    
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--mode', type=str, default='default',
                       help='Dataset mode: "default", "vae:<group>", or "diffusion:<stage>"')
    
    args = parser.parse_args()
    config = load_configs(parser)
    validate_dataset(args.num_samples, config, args.mode)

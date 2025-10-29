# Validation script to test dataset loading and visualize samples
import sys
import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import UrbanInpaintingDataset


def visualize_sample(sample_data, save_path=None):
    """
    Visualize a dataset sample with all its components.
    
    Args:
        sample_data: Tuple of (image, conditions) from dataset
        save_path: Optional path to save visualization
    """
    if len(sample_data) == 2:
        im, cond_input = sample_data
    else:
        im = sample_data
        cond_input = None
    
    # Convert tensors to numpy
    im_np = im.numpy()
    
    # Create figure
    if cond_input is not None and 'image' in cond_input:
        spatial_cond = cond_input['image'].numpy()
        num_cond_channels = spatial_cond.shape[0]
        
        # Calculate grid layout
        n_rows = 2 + (num_cond_channels + 2) // 3  # RGB + mask + conditions
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    idx = 0
    
    # Plot RGB satellite image (target)
    if im_np.shape[0] == 3:
        rgb = im_np.transpose(1, 2, 0)
        # Denormalize from [-1, 1] to [0, 1]
        rgb = (rgb + 1) / 2
        rgb = np.clip(rgb, 0, 1)
        axes[idx].imshow(rgb)
        axes[idx].set_title('Target RGB Image', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    if cond_input is not None:
        # Plot mask
        if 'mask' in cond_input:
            mask = cond_input['mask'].numpy()[0]
            axes[idx].imshow(mask, cmap='gray')
            axes[idx].set_title('Inpainting Mask\n(1=regenerate, 0=keep)', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            idx += 1
        
        # Plot spatial conditions
        if 'image' in cond_input:
            spatial_cond = cond_input['image'].numpy()
            spatial_names = cond_input.get('meta', {}).get('spatial_names', [])
            
            # Plot masked image first (RGB)
            if len(spatial_names) > 0 and 'masked_image' in spatial_names[0]:
                masked_rgb = spatial_cond[:3].transpose(1, 2, 0)
                masked_rgb = (masked_rgb + 1) / 2
                masked_rgb = np.clip(masked_rgb, 0, 1)
                axes[idx].imshow(masked_rgb)
                axes[idx].set_title('Masked Input\n(Context)', fontsize=12, fontweight='bold')
                axes[idx].axis('off')
                idx += 1
            
            # Plot each conditioning channel
            for i in range(spatial_cond.shape[0]):
                if idx >= len(axes):
                    break
                
                channel = spatial_cond[i]
                
                # Get channel name
                if i < len(spatial_names):
                    name = spatial_names[i]
                else:
                    name = f'Channel {i}'
                
                # Skip masked_image channels (already plotted)
                if 'masked_image:' in name:
                    continue
                
                # Plot channel
                if 'mask' in name.lower():
                    axes[idx].imshow(channel, cmap='gray', vmin=0, vmax=1)
                elif 'temp' in name.lower() or 'lst' in name.lower():
                    # Temperature - use hot colormap
                    axes[idx].imshow(channel, cmap='hot')
                elif 'ndvi' in name.lower():
                    # NDVI - use green colormap
                    axes[idx].imshow(channel, cmap='RdYlGn', vmin=-1, vmax=1)
                else:
                    # Binary/categorical
                    axes[idx].imshow(channel, cmap='viridis')
                
                axes[idx].set_title(name, fontsize=10)
                axes[idx].axis('off')
                idx += 1
        
        # Print metadata
        if 'meta' in cond_input:
            meta = cond_input['meta']
            print("\n" + "="*50)
            print("Sample Metadata:")
            print("="*50)
            for key, value in meta.items():
                if key != 'spatial_names':
                    print(f"  {key}: {value}")
            if 'spatial_names' in meta:
                print(f"\n  Spatial condition channels ({len(meta['spatial_names'])}):")
                for name in meta['spatial_names']:
                    print(f"    - {name}")
    
    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def validate_dataset(num_samples=5):
    """
    Validate dataset loading and visualize samples.
    
    Args:
        config_path: Path to config file
        num_samples: Number of samples to visualize
    """
    print("="*50)
    print("Dataset Validation")
    print("="*50)
    
    ###### setup config variables #######
    repo_name = 'masterthesis_genai_spatialplan'
    if not repo_name in os.getcwd():
        os.chdir(repo_name)

    p=os.popen('git rev-parse --show-toplevel')
    repo_dir = p.read().strip()
    p.close()

    with open(f"{repo_dir}/code/model/config/class_cond.yml", 'r') as stream:
        config = yaml.safe_load(stream)
        
    with open(f"{repo_dir}/code/data_acquisition/config.yml", 'r') as stream:
        data_config = yaml.safe_load(stream)

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    print("\n✓ Loaded configuration")
    
    # Create dataset
    print("\nLoading dataset...")
    try:
        dataset = UrbanInpaintingDataset(
            split='train',
            use_latents=False,
            latent_path=None
        )
        print(f"✓ Successfully loaded dataset!")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Dataset info
    print("\n" + "="*50)
    print("Dataset Information")
    print("="*50)
    print(f"  Number of patches: {len(dataset)}")
    print(f"  Patch size: {dataset.patch_size}x{dataset.patch_size} pixels")
    print(f"  Image channels: {dataset.im_channels}")
    print(f"  Selected date: {dataset.selected_date}")
    print(f"  Conditioning types: {dataset.condition_types}")
    print(f"  OSM layers: {dataset.osm_layers}")
    print(f"  Environmental layers: {dataset.environmental_layers}")
    
    # Test loading samples
    print("\n" + "="*50)
    print("Testing Sample Loading")
    print("="*50)
    
    # Create output directory
    output_dir = f"{big_data_storage_path}/results/{config['train_params'].get('task_name', 'urban_inpainting')}/dataset_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        try:
            sample = dataset[i]
            
            # Check shapes
            if len(sample) == 2:
                im, cond = sample
                print(f"  Image shape: {im.shape}")
                print(f"  Image range: [{im.min():.3f}, {im.max():.3f}]")
                
                if 'image' in cond:
                    print(f"  Spatial condition shape: {cond['image'].shape}")
                    print(f"  Spatial condition range: [{cond['image'].min():.3f}, {cond['image'].max():.3f}]")
                
                if 'mask' in cond:
                    print(f"  Mask shape: {cond['mask'].shape}")
                    print(f"  Mask unique values: {torch.unique(cond['mask']).tolist()}")
                    mask_ratio = (cond['mask'] == 1).float().mean().item()
                    print(f"  Mask ratio (to regenerate): {mask_ratio:.2%}")
            else:
                im = sample
                print(f"  Image shape: {im.shape}")
                print(f"  Image range: [{im.min():.3f}, {im.max():.3f}]")
            
            # Visualize
            save_path = os.path.join(output_dir, f'sample_{i}.png')
            visualize_sample(sample, save_path)
            
            print(f"  ✓ Sample {i+1} validated successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("Validation Complete!")
    print("="*50)
    print(f"\nVisualization saved to: {output_dir}")
    print("\nIf all samples loaded successfully, your dataset is ready for training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate urban inpainting dataset')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to visualize'
    )
    args = parser.parse_args()
    
    validate_dataset(args.num_samples)

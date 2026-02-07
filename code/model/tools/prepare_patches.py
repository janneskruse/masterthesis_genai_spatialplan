"""
Script to pre-generate cached patches for faster training.
Run this once before training to prepare the cache.
"""

###### import libraries ######
# Standard libraries
import argparse
from pathlib import Path

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from helpers.load_configs import load_configs, add_config_arguments

def main():
    """Prepare cached patches for training and validation"""
    
    parser = argparse.ArgumentParser(description='Prepare cached patches for training')
    add_config_arguments(parser)
    parser.add_argument('--max_patches', type=int, default=None,
                       help='Maximum number of patches to cache (for quick testing). None = all patches')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training set (only prepare val)')
    parser.add_argument('--skip_val', action='store_true',
                       help='Skip validation set (only prepare train)')
    args = parser.parse_args()
    
    config = load_configs()
    big_data_storage_path = config['data_config'].get("big_data_storage_path", "/work/zt75vipu-master/data")
    task_name = config['train_params']['task_name']
    print(f"\n{'='*80}")
    print(f"Preparing Cached Patches for Task: {task_name}")
    if args.max_patches:
        print(f"Test mode: Max {args.max_patches} patches per split")
    print(f"{'='*80}\n")
    
    # Prepare training patches
    if not args.skip_train:
        print("Step 1/2: Processing training sets...")
        train_dataset = UrbanInpaintingDataset(
            split='train',
            mode='default',
            use_cached_patches=False,  # Force Xarray loading
        )
        cache_dir = train_dataset.prepare_cached_patches(max_patches=args.max_patches)
    else:
        print("Step 1/2: Skipping training set (--skip_train)")
        cache_dir = None
    
    # Prepare validation patches
    if not args.skip_val:
        print("\nStep 2/2: Processing validation set...")
        val_dataset = UrbanInpaintingDataset(
            split='val',
            mode='default',
            use_cached_patches=False,
        )
        val_cache_dir = val_dataset.prepare_cached_patches(max_patches=args.max_patches)
        if cache_dir is None:
            cache_dir = val_cache_dir
    else:
        print("\nStep 2/2: Skipping validation set (--skip_val)")
    
    print(f"\n{'='*80}")
    print(f"✓ Cache preparation complete!")
    print(f"✓ Cached patches saved to: {cache_dir}")
    print(f"\nNext steps:")
    print(f"1. Generate latents using VAE with cached patches for both semantic and satellite data")
    print(f"2. Train the Temperature predictor")
    print(f"3. Train LDMs with both cached patches, latents and Temperature predictions")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
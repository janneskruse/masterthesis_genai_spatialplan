"""
Script to pre-generate cached patches for faster training.
Run this once before training to prepare the cache.
"""

###### import libraries ######
# Standard libraries
from pathlib import Path

# Local imports
from model.dataset.dataset import UrbanInpaintingDataset
from helpers.load_configs import load_configs

def main():
    """Prepare cached patches for training and validation"""
    
    config = load_configs()
    big_data_storage_path = config['data_config'].get("big_data_storage_path", "/work/zt75vipu-master/data")
    task_name = config['train_params']['task_name']
    cache_dir_semantics = Path(big_data_storage_path) / "processed" / task_name / "semantic"
    cache_dir_satellite = Path(big_data_storage_path) / "processed" / task_name / "satellite"
    
    print(f"\n{'='*80}")
    print(f"Preparing Cached Patches for Task: {task_name}")
    print(f"{'='*80}\n")
    
    # Prepare training patches
    print("Step 1/2: Processing training sets...")
    train_dataset = UrbanInpaintingDataset(
        split='train',
        mode='semantic',
        use_latents=False,
        use_cached_patches=False,  # Force Xarray loading
        cache_dir=cache_dir_semantics
    )
    train_dataset.prepare_cached_patches()
    
    train_dataset_sat = UrbanInpaintingDataset(
        split='train',
        mode='satellite',
        use_latents=False,
        use_cached_patches=False,  # Force Xarray loading
        cache_dir=cache_dir_satellite
    )
    train_dataset_sat.prepare_cached_patches()
    
    # Prepare validation patches
    print("\nStep 2/2: Processing validation set...")
    val_dataset = UrbanInpaintingDataset(
        split='val',
        mode='semantic',
        use_latents=False,
        use_cached_patches=False,
        cache_dir=cache_dir_semantics
    )
    val_dataset.prepare_cached_patches()
    val_dataset_sat = UrbanInpaintingDataset(
        split='val',
        mode='satellite',
        use_latents=False,
        use_cached_patches=False,
        cache_dir=cache_dir_satellite
    )
    val_dataset_sat.prepare_cached_patches()
    
    print(f"\n{'='*80}")
    print(f"✓ Cache preparation complete!")
    print(f"✓ Cached patches saved to: {cache_dir_semantics} and {cache_dir_satellite}")
    print(f"\nNext steps:")
    print(f"1. Generate latents using VAE with cached patches")
    print(f"2. Train LDM with both cached patches and latents")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
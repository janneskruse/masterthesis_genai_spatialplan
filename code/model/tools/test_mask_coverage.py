"""
Test script to check inpainting mask coverage statistics across dataset.
"""

###### import libraries ######
# Standard libraries
import os
from pathlib import Path

# Data handling
import torch
import numpy as np
from tqdm import tqdm

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from helpers.load_configs import load_configs


def analyze_mask_coverage(num_samples=50):
    """
    Analyze mask coverage statistics across dataset samples.
    
    Args:
        num_samples: Number of samples to analyze
    """
    # Load config
    config = load_configs()
    data_config = config['data_config']
    train_config = config['train_params']
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    # Check cache directory
    cache_dir = Path(big_data_storage_path) / "processed" / task_name / "patches"
    use_cached_patches = cache_dir.exists()
    
    # Check if latents exist for val split
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    latent_dir_name = train_config.get('latent_dir_name', 'semantic_vae_latents')
    latent_path = os.path.join(data_dir, latent_dir_name + "_val")
    use_latents = os.path.exists(latent_path)
    
    print("\n" + "="*60)
    print("Mask Coverage Analysis")
    print("="*60)
    print(f"Task: {task_name}")
    print(f"Use cached patches: {use_cached_patches}")
    print(f"Use latents: {use_latents}")
    print(f"Analyzing {num_samples} samples from validation set")
    print("="*60 + "\n")
    
    # Load dataset
    dataset = UrbanInpaintingDataset(
        split='val',
        mode='semantic',
        use_latents=use_latents,
        latent_path=latent_path if use_latents else None,
        use_cached_patches=use_cached_patches,
        cache_dir=cache_dir
    )
    
    print(f"✓ Loaded dataset with {len(dataset)} samples\n")
    
    # Analyze samples
    mask_stats = {
        'mean_coverage': [],
        'min_coverage': [],
        'max_coverage': [],
        'std_coverage': [],
        'num_ones': [],
        'num_zeros': [],
        'shape': None
    }
    
    num_samples = min(num_samples, len(dataset))
    
    for idx in tqdm(range(num_samples), desc="Analyzing masks"):
        sample_data = dataset[idx]
        
        # Debug first sample to understand structure
        if idx == 0:
            print(f"\nDebug sample 0 structure:")
            print(f"  Type: {type(sample_data)}")
            if isinstance(sample_data, (tuple, list)):
                print(f"  Length: {len(sample_data)}")
                for i, item in enumerate(sample_data):
                    print(f"  Item {i}: type={type(item)}")
                    if isinstance(item, dict):
                        print(f"    Keys: {list(item.keys())}")
                        if 'meta' in item:
                            print(f"    Meta keys: {list(item['meta'].keys())}")
            print()
        
        if len(sample_data) == 2:
            _, cond_input = sample_data
        else:
            cond_input = sample_data if isinstance(sample_data, dict) else {}
        
        # Extract mask - try multiple locations
        mask = None
        
        # Try meta.inpainting_mask
        if 'meta' in cond_input and 'inpainting_mask' in cond_input['meta']:
            mask = cond_input['meta']['inpainting_mask']
        
        # Try direct key
        elif 'inpainting_mask' in cond_input:
            mask = cond_input['inpainting_mask']
        
        # Try image channel with 'mask' in name
        elif 'image' in cond_input and 'meta' in cond_input:
            spatial_names = cond_input['meta'].get('spatial_names', [])
            for ch_idx, name in enumerate(spatial_names):
                if 'mask' in name.lower():
                    mask = cond_input['image'][ch_idx:ch_idx+1]
                    print(f"  Found mask in image channel {ch_idx}: {name}")
                    break
        
        if mask is not None:
            if mask_stats['shape'] is None:
                mask_stats['shape'] = tuple(mask.shape)
            
            mask_np = mask.numpy() if torch.is_tensor(mask) else mask
            
            # Coverage statistics
            mask_stats['mean_coverage'].append(mask_np.mean())
            mask_stats['min_coverage'].append(mask_np.min())
            mask_stats['max_coverage'].append(mask_np.max())
            mask_stats['std_coverage'].append(mask_np.std())
            mask_stats['num_ones'].append((mask_np == 1.0).sum())
            mask_stats['num_zeros'].append((mask_np == 0.0).sum())
        else:
            if idx == 0:
                print(f"⚠ Warning: No mask found in sample {idx}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Mask Statistics")
    print("="*60)
    
    if len(mask_stats['mean_coverage']) == 0:
        print("✗ No masks found in any samples!")
        print("This could mean:")
        print("  1. Masks are not included in cached patches")
        print("  2. Masks are stored in a different location")
        print("  3. Dataset structure has changed")
        return
    
    print(f"✓ Found masks in {len(mask_stats['mean_coverage'])}/{num_samples} samples")
    print(f"Mask shape: {mask_stats['shape']}")
    print(f"\nMask coverage (fraction of 1s = pixels to inpaint):")
    print(f"  Mean:   {np.mean(mask_stats['mean_coverage']):.2%} ± {np.std(mask_stats['mean_coverage']):.2%}")
    print(f"  Median: {np.median(mask_stats['mean_coverage']):.2%}")
    print(f"  Min:    {np.min(mask_stats['mean_coverage']):.2%}")
    print(f"  Max:    {np.max(mask_stats['mean_coverage']):.2%}")
    
    print(f"\nContext preservation (1 - coverage):")
    context_preservation = [1 - cov for cov in mask_stats['mean_coverage']]
    print(f"  Mean:   {np.mean(context_preservation):.2%} ± {np.std(context_preservation):.2%}")
    print(f"  Median: {np.median(context_preservation):.2%}")
    print(f"  Min:    {np.min(context_preservation):.2%}")
    print(f"  Max:    {np.max(context_preservation):.2%}")
    
    # Distribution
    print(f"\nCoverage distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(mask_stats['mean_coverage'], bins=bins)
    for i in range(len(hist)):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:3d} samples ({hist[i]/num_samples*100:5.1f}%)")
    
    # Check for all-ones or all-zeros masks
    all_ones_count = sum(1 for cov in mask_stats['mean_coverage'] if cov >= 0.999)
    all_zeros_count = sum(1 for cov in mask_stats['mean_coverage'] if cov <= 0.001)
    
    print(f"\nExtreme cases:")
    print(f"  All 1s (full generation): {all_ones_count} samples ({all_ones_count/num_samples*100:.1f}%)")
    print(f"  All 0s (no inpainting):   {all_zeros_count} samples ({all_zeros_count/num_samples*100:.1f}%)")
    
    # Show some example indices
    if all_ones_count > 0:
        print(f"\nExample indices with all 1s masks (first 5):")
        count = 0
        for idx, cov in enumerate(mask_stats['mean_coverage']):
            if cov >= 0.999:
                print(f"  Sample {idx}: coverage={cov:.4f}")
                count += 1
                if count >= 5:
                    break
    
    print("\n" + "="*60)


if __name__ == '__main__':
    analyze_mask_coverage(num_samples=50)

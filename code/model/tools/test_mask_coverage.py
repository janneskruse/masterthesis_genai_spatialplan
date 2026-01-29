"""
Test script to check inpainting mask coverage statistics across dataset.

Tests the mixed mask strategy with multiple methods (street_blocks, random_polygon, 
random_rectangle, random_square) and visualizes samples from each type.
"""

###### import libraries ######
# Standard libraries
import os
from pathlib import Path
from collections import defaultdict

# Data handling
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from helpers.load_configs import load_configs


def analyze_mask_coverage(num_samples=100, plot_samples=True, samples_per_type=2):
    """
    Analyze mask coverage statistics across dataset samples.
    
    Tests the mixed mask strategy and collects samples from each mask type.
    
    Args:
        num_samples: Number of samples to analyze
        plot_samples: Whether to plot sample visualizations
        samples_per_type: Number of samples to visualize per mask type
    """
    # Load config
    config = load_configs()
    data_config = config['data_config']
    train_config = config['train_params']
    repo_dir = Path(config.get('repo_dir', '.')).resolve()
    
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    # Check cache directory (use same logic as dataset)
    cache_dir = Path(big_data_storage_path) / "processed" / task_name / "patches"
    use_cached_patches = cache_dir.exists() and (cache_dir / "patches_metadata_val.csv").exists()
    
    print("\n" + "="*60)
    print("Mask Coverage Analysis")
    print("="*60)
    print(f"Task: {task_name}")
    print(f"Use cached patches: {use_cached_patches}")
    print(f"Cache directory: {cache_dir}")
    print(f"Analyzing {num_samples} samples from validation set")
    print("="*60 + "\n")
    
    # Load dataset in diffusion mode (automatically handles latents)
    dataset = UrbanInpaintingDataset(
        split='val',
        mode='diffusion:semantic',  # Automatically loads latents if available
        use_cached_patches=use_cached_patches,
        cache_dir=str(cache_dir) if use_cached_patches else None
    )
    
    print(f"✓ Loaded dataset with {len(dataset)} samples\n")
    
    # Analyze samples
    mask_stats = {
        'coverage_percent': [],
        'actual_type': [],
        'requested_type': [],
        'fallback_reason': [],
        'shape': None,
        'samples_by_type': defaultdict(list)  # Store sample indices by mask type
    }
    
    num_samples = min(num_samples, len(dataset))
    
    for idx in tqdm(range(num_samples), desc="Analyzing masks"):
        try:
            # Get sample (returns: pred_latent, conditioning_dict)
            pred_latent, cond_input = dataset[idx]
            
            # Extract mask from pixel-space conditioning
            mask = None
            mask_info = {}
            
            # Mask is in pixel-space within 'image' tensor
            if 'image' in cond_input and 'meta' in cond_input:
                pixel_space_names = cond_input['meta'].get('pixel_space_names', [])
                
                # Find inpainting mask channel
                try:
                    mask_idx = pixel_space_names.index('inpainting_mask')
                    mask = cond_input['image'][mask_idx]  # [H, W]
                except ValueError:
                    pass
            
            # Extract patch_info from meta (contains mask generation stats)
            if 'meta' in cond_input and 'patch_info' in cond_input['meta']:
                mask_info = cond_input['meta']['patch_info']
            
            if mask is not None:
                if mask_stats['shape'] is None:
                    mask_stats['shape'] = tuple(mask.shape)
                
                mask_np = mask.numpy() if torch.is_tensor(mask) else mask
                
                # Coverage statistics
                coverage_percent = (mask_np.sum() / mask_np.size) * 100
                mask_stats['coverage_percent'].append(coverage_percent)
                
                # Type tracking (from patch_info)
                actual_type = mask_info.get('actual_type', 'unknown')
                requested_type = mask_info.get('requested_type', 'unknown')
                fallback_reason = mask_info.get('fallback_reason', None)
                
                mask_stats['actual_type'].append(actual_type)
                mask_stats['requested_type'].append(requested_type)
                mask_stats['fallback_reason'].append(fallback_reason)
                
                # Store sample indices by type (for visualization)
                # Extract base type from actual_type (e.g., "street_blocks_fallback(random_rectangle)" -> "random_rectangle")
                base_type = actual_type
                if 'fallback(' in actual_type and ')' in actual_type:
                    base_type = actual_type.split('fallback(')[1].split(')')[0]
                elif 'mixed(' in actual_type and ')' in actual_type:
                    base_type = actual_type.split('mixed(')[1].split(')')[0]
                
                if len(mask_stats['samples_by_type'][base_type]) < samples_per_type * 2:  # Collect extra
                    mask_stats['samples_by_type'][base_type].append((idx, mask_np, coverage_percent, actual_type))
        
        except Exception as e:
            print(f"\n⚠ Warning: Failed to process sample {idx}: {e}")
            continue
    
    # Print statistics
    print("\n" + "="*60)
    print("Mask Statistics")
    print("="*60)
    
    if len(mask_stats['coverage_percent']) == 0:
        print("✗ No masks found in any samples!")
        print("This could mean:")
        print("  1. Dataset mode is incorrect")
        print("  2. Inpainting mask not included in pixel-space conditioning")
        print("  3. Dataset structure has changed")
        return
    
    print(f"✓ Found masks in {len(mask_stats['coverage_percent'])}/{num_samples} samples")
    print(f"Mask shape: {mask_stats['shape']}")
    
    # Coverage statistics
    coverages = np.array(mask_stats['coverage_percent'])
    print(f"\nMask coverage (percentage of pixels to inpaint):")
    print(f"  Mean:   {np.mean(coverages):.2f}% ± {np.std(coverages):.2f}%")
    print(f"  Median: {np.median(coverages):.2f}%")
    print(f"  Min:    {np.min(coverages):.2f}%")
    print(f"  Max:    {np.max(coverages):.2f}%")
    
    # Context preservation
    print(f"\nContext preservation (100% - coverage):")
    context = 100 - coverages
    print(f"  Mean:   {np.mean(context):.2f}% ± {np.std(context):.2f}%")
    print(f"  Median: {np.median(context):.2f}%")
    print(f"  Min:    {np.min(context):.2f}%")
    print(f"  Max:    {np.max(context):.2f}%")
    
    # Coverage distribution
    print(f"\nCoverage distribution:")
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    hist, _ = np.histogram(coverages, bins=bins)
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {bins[i]:3.0f}%-{bins[i+1]:3.0f}%: {hist[i]:3d} samples ({hist[i]/num_samples*100:5.1f}%)")
    
    # Mask type distribution
    print(f"\nMask type distribution:")
    type_counts = defaultdict(int)
    for actual_type in mask_stats['actual_type']:
        # Extract base type
        base_type = actual_type
        if 'fallback(' in actual_type:
            base_type = actual_type.split('fallback(')[1].split(')')[0]
        elif 'mixed(' in actual_type:
            base_type = actual_type.split('mixed(')[1].split(')')[0]
        type_counts[base_type] += 1
    
    for mask_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {mask_type:20s}: {count:3d} samples ({count/num_samples*100:5.1f}%)")
    
    # Fallback statistics
    fallback_count = sum(1 for r in mask_stats['fallback_reason'] if r is not None)
    if fallback_count > 0:
        print(f"\nFallback triggers:")
        print(f"  Total: {fallback_count} samples ({fallback_count/num_samples*100:.1f}%)")
        
        fallback_reasons = defaultdict(int)
        for reason in mask_stats['fallback_reason']:
            if reason is not None:
                fallback_reasons[reason] += 1
        
        for reason, count in sorted(fallback_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:30s}: {count:3d} samples")
    
    print("\n" + "="*60)
    
    # Plot samples if requested
    if plot_samples and len(mask_stats['samples_by_type']) > 0:
        plot_mask_samples(mask_stats['samples_by_type'], samples_per_type, task_name, repo_dir)


def plot_mask_samples(samples_by_type, samples_per_type, task_name, repo_dir):
    """
    Plot sample masks for each mask type.
    
    Args:
        samples_by_type: Dict mapping mask type to list of (idx, mask_np, coverage, actual_type)
        samples_per_type: Number of samples to plot per type
        task_name: Task name for plot title
        repo_dir: Repository root directory
    """
    # Filter to available types
    available_types = [t for t in samples_by_type.keys() if len(samples_by_type[t]) > 0]
    
    if len(available_types) == 0:
        print("\n⚠ No samples available for plotting")
        return
    
    print(f"\n{'='*60}")
    print(f"Plotting sample masks")
    print(f"{'='*60}")
    print(f"Available types: {available_types}")
    print(f"Samples per type: {samples_per_type}")
    print(f"{'='*60}\n")
    
    # Create figure
    n_types = len(available_types)
    n_samples = min(samples_per_type, min(len(samples_by_type[t]) for t in available_types))
    
    fig, axes = plt.subplots(n_types, n_samples, figsize=(3 * n_samples, 3 * n_types))
    
    # Handle single row/column edge cases
    if n_types == 1 and n_samples == 1:
        axes = np.array([[axes]])
    elif n_types == 1:
        axes = axes.reshape(1, -1)
    elif n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Inpainting Mask Samples - {task_name}", fontsize=16, y=0.995)
    
    # Plot samples
    for type_idx, mask_type in enumerate(available_types):
        samples = samples_by_type[mask_type][:n_samples]
        
        for sample_idx, (idx, mask_np, coverage, actual_type) in enumerate(samples):
            ax = axes[type_idx, sample_idx]
            
            # Plot mask (1=inpaint, 0=keep)
            im = ax.imshow(mask_np, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
            
            # Title with coverage
            title = f"{mask_type}\n"
            if 'fallback' in actual_type:
                title += f"(fallback)\n"
            title += f"Coverage: {coverage:.1f}%"
            ax.set_title(title, fontsize=9)
            
            ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('1 = Inpaint | 0 = Keep', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    
    # Save plot to results directory
    results_dir = repo_dir / "results" / task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "mask_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved mask samples to: {output_path}")
    
    plt.show()
    print(f"{'='*60}\n")


if __name__ == '__main__':
    analyze_mask_coverage(num_samples=100, plot_samples=True, samples_per_type=3)

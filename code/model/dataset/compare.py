"""Module for comparing VAE training patches with current dataset patches"""

###### import libraries ######
# Standard libraries
from pathlib import Path
from typing import Dict, List, Tuple

# Data handling
import pandas as pd


def load_vae_training_stats(stats_path: Path) -> pd.DataFrame:
    """
    Load VAE training statistics CSV.
    
    Args:
        stats_path: Path to inpainting_mask_stats_train.csv
        
    Returns:
        DataFrame with patch information from VAE training
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats CSV not found: {stats_path}")
    
    df = pd.read_csv(stats_path)
    print(f"✓ Loaded {len(df)} patch records from VAE training stats")
    
    # Validate required columns
    required_cols = ['index', 'region', 'y', 'x']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in stats CSV: {missing_cols}")
    
    return df


def compare_patches(
    vae_stats_df: pd.DataFrame,
    current_patches: List[Tuple[int, int, str]],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare VAE training patches with current dataset patches.
    
    Args:
        vae_stats_df: DataFrame from VAE training stats with columns ['index', 'region', 'y', 'x']
        current_patches: List of (y, x, region) tuples from current dataset
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dictionary with comparison results and index mapping
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Comparing VAE and Current Dataset Patches")
        print(f"{'='*60}")
    
    vae_patch_count = len(vae_stats_df)
    current_patch_count = len(current_patches)
    
    if verbose:
        print(f"VAE training patches: {vae_patch_count}")
        print(f"Current dataset patches: {current_patch_count}")
        print(f"Difference: {abs(vae_patch_count - current_patch_count)}")
    
    # Create lookup keys for both datasets
    vae_keys = set(
        vae_stats_df.apply(
            lambda row: (row['region'], int(row['y']), int(row['x'])),
            axis=1
        )
    )
    
    current_keys = set(
        (region, y, x) for y, x, region in current_patches
    )
    
    # Find matches and mismatches
    matching_keys = vae_keys & current_keys
    vae_only_keys = vae_keys - current_keys
    current_only_keys = current_keys - vae_keys
    
    if verbose:
        print(f"\n{'='*60}")
        print("Patch Correspondence Analysis")
        print(f"{'='*60}")
        print(f"✓ Matching patches: {len(matching_keys)}")
        print(f"⚠ VAE-only patches: {len(vae_only_keys)}")
        print(f"⚠ Current dataset-only patches: {len(current_only_keys)}")
    
    # Analyze VAE-only patches
    if vae_only_keys:
        print(f"\n{'='*60}")
        print("VAE Training Patches Not in LDM Dataset")
        print(f"{'='*60}")
        
        vae_only_df = vae_stats_df[
            vae_stats_df.apply(
                lambda row: (row['region'], int(row['y']), int(row['x'])) in vae_only_keys,
                axis=1
            )
        ]
        
        # Group by region
        region_counts = vae_only_df['region'].value_counts()
        print("\nBy region:")
        for region, count in region_counts.items():
            print(f"  {region}: {count} patches")
        
        # Show first few examples
        print(f"\nFirst {min(10, len(vae_only_df))} examples:")
        for idx, row in vae_only_df.head(10).iterrows():
            print(f"  Index {row['index']}: {row['region']} at (y={row['y']}, x={row['x']})")
    
    # Create index mapping for matching patches
    if verbose:
        print(f"\n{'='*60}")
        print("Creating Index Mapping")
        print(f"{'='*60}")
    
    # Map current dataset index to VAE training index
    current_to_vae_index = {}
    vae_to_current_index = {}
    
    # Create reverse lookup for VAE indices by spatial key
    vae_key_to_index = {}
    for _, row in vae_stats_df.iterrows():
        key = (row['region'], int(row['y']), int(row['x']))
        vae_idx = int(row['index'])
        if key in vae_key_to_index:
            if verbose:
                print(f"⚠ Duplicate patch in VAE stats: {key}")
        else:
            vae_key_to_index[key] = vae_idx
    
    for current_idx, (y, x, region) in enumerate(current_patches):
        key = (region, y, x)
        if key in vae_key_to_index:
            vae_idx = vae_key_to_index[key]
            current_to_vae_index[current_idx] = vae_idx
            vae_to_current_index[vae_idx] = current_idx
    
    if verbose:
        print(f"✓ Created mapping for {len(current_to_vae_index)} matching patches")
        
        # Check for index consistency
        index_matches = sum(1 for curr_idx, vae_idx in current_to_vae_index.items() if curr_idx == vae_idx)
        print(f"✓ Patches with matching indices: {index_matches}/{len(current_to_vae_index)}")
        
        if index_matches < len(current_to_vae_index):
            print(f"⚠ {len(current_to_vae_index) - index_matches} patches have mismatched indices")
            print("\nFirst 5 mismatches:")
            mismatch_count = 0
            for curr_idx, vae_idx in sorted(current_to_vae_index.items()):
                if curr_idx != vae_idx and mismatch_count < 5:
                    y, x, region = current_patches[curr_idx]
                    print(f"  Current index {curr_idx} → VAE index {vae_idx} ({region} at y={y}, x={x})")
                    mismatch_count += 1
    
    index_matches = sum(1 for curr_idx, vae_idx in current_to_vae_index.items() if curr_idx == vae_idx)
    
    return {
        'vae_patch_count': vae_patch_count,
        'current_patch_count': current_patch_count,
        'matching_count': len(matching_keys),
        'vae_only_count': len(vae_only_keys),
        'current_only_count': len(current_only_keys),
        'vae_only_keys': vae_only_keys,
        'current_only_keys': current_only_keys,
        'current_to_vae_index': current_to_vae_index,
        'vae_to_current_index': vae_to_current_index,
        'index_consistency': index_matches / len(current_to_vae_index) if current_to_vae_index else 0.0,
        'matching_keys': matching_keys
    }


def reconcile_patches_with_latents(
    stats_csv_path: Path,
    current_patches: List[Tuple[int, int, str]],
    latent_files: List[str],
    verbose: bool = True
) -> Tuple[List[Tuple[int, int, str]], List[str], Dict[str, any]]:
    """
    Reconcile current patches with available latents using VAE training stats.
    Returns filtered patches and latent file list that match.
    
    Args:
        stats_csv_path: Path to VAE training stats CSV
        current_patches: List of (y, x, region) tuples from current dataset
        latent_files: List of latent file paths
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (filtered_patches, filtered_latent_files, comparison_results)
    """
    # Load VAE training stats
    if not stats_csv_path.exists():
        if verbose:
            print(f"⚠ VAE training stats not found at {stats_csv_path}")
            print(f"⚠ Cannot reconcile patches - using all patches without latents")
        return current_patches, [], {}
    
    vae_stats_df = load_vae_training_stats(stats_csv_path)
    
    # Compare patches
    comparison_results = compare_patches(vae_stats_df, current_patches, verbose=verbose)
    
    # Extract latent indices from file names
    latent_indices = {int(Path(f).stem.split('_')[1]): f for f in latent_files}
    
    # Find patches that have matching latents
    current_to_vae = comparison_results['current_to_vae_index']
    
    filtered_patches = []
    filtered_latent_files = []
    
    for curr_idx, (y, x, region) in enumerate(current_patches):
        if curr_idx in current_to_vae:
            vae_idx = current_to_vae[curr_idx]
            if vae_idx in latent_indices:
                filtered_patches.append((y, x, region))
                filtered_latent_files.append(latent_indices[vae_idx])
    
    if verbose:
        print(f"\n{'='*60}")
        print("Patch Filtering Results")
        print(f"{'='*60}")
        print(f"Original patches: {len(current_patches)}")
        print(f"Patches with matching latents: {len(filtered_patches)}")
        print(f"Dropped patches: {len(current_patches) - len(filtered_patches)}")
        
        if len(filtered_patches) < len(current_patches):
            dropped_count = len(current_patches) - len(filtered_patches)
            dropped_percent = (dropped_count / len(current_patches)) * 100
            print(f"\n⚠ Dropped {dropped_count} patches ({dropped_percent:.1f}%) without matching latents")
    
    return filtered_patches, filtered_latent_files, comparison_results
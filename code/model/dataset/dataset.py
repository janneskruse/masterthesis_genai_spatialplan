###### import libraries ######
# Standard libraries
import os
from pathlib import Path
from typing import Optional, List, Dict
from tqdm.auto import tqdm

# Data handling
import numpy as np
import xarray as xr
import pandas as pd

# Data Science/ML libraries
import torch
from torch.utils.data.dataset import Dataset

# Local imports
from model.utils.diffusion_utils import load_latents
from model.utils.diffusion_utils import load_single_latent
from model.utils.data_utils import apply_layer_transform
from model.utils.config_utils import compute_patch_and_latent_sizes, get_default_configs
from model.utils.layer_config import (
    get_layer_info,
    count_layer_channels,
    get_channel_names,
    get_layer_channels_from_names,
)
from model.utils.inpainting import create_inpainting_mask
from helpers.load_configs import load_configs
from model.dataset.compare import reconcile_patches_with_latents


# Dataset class
class UrbanInpaintingDataset(Dataset):
    """
    Dataset for urban layout inpainting with multiple conditioning types:
    - Spatial context (surrounding areas via inpainting mask)
    - OSM features (buildings, streets, water, etc.)
    - Environmental data (NDVI, LST)
    - Satellite imagery
    
    Supports two modes:
    1. On-the-fly loading from Xarray (slower, flexible)
    2. Pre-saved patches (faster, recommended for training)
    """
    
    def __init__(self, split, 
                 use_cached_patches: bool = True,
                 cache_dir: Optional[str] = None,
                 mode: str = 'default',
        ):
        """
        :param split: 'train' or 'val'
        :param use_cached_patches: whether to use cached patches
        :param cache_dir: directory for cached patches
        :param mode: 'default', 'vae:satellite', 'vae:environmental', 'diffusion:semantic', etc.
        """
        
        ###### Setup config variables #######
        config = load_configs()
        data_config = config['data_config']
        dataset_config = config.get('dataset_params', None)
        
        if dataset_config is None:
            raise ValueError("Dataset configuration not found in config file")
        
        self.config = config
        self.data_config = data_config
        self.dataset_config = dataset_config
        self.mode = mode
        
        # Validate mode format
        if mode != 'default':
            mode_parts = mode.split(':')
            if len(mode_parts) != 2 or mode_parts[0] not in ['vae', 'diffusion']:
                raise ValueError(
                    f"Invalid mode: '{mode}'. Must be 'default', 'vae:<group_name>', or 'diffusion:<stage_name>'. "
                    f"Examples: 'vae:satellite', 'diffusion:semantic'"
                )
            self.mode_type = mode_parts[0]  # 'vae' or 'diffusion'
            self.mode_target = mode_parts[1]  # 'satellite', 'semantic', etc.
        else:
            self.mode_type = 'default'
            self.mode_target = None
        
        # Store global layer registry and VAE groups
        self.layers_registry = config.get('layers', {})
        self.vae_groups = config.get('vae_groups', {})
        self.diffusion_stages = config.get('diffusion_stages', {})
        
        # Get config for current mode
        if self.mode_type == 'vae':
            # VAE mode: use the target group's config
            if self.mode_target not in self.vae_groups:
                raise ValueError(
                    f"VAE group '{self.mode_target}' not found in config. "
                    f"Available groups: {list(self.vae_groups.keys())}"
                )
            vae_config = self.vae_groups[self.mode_target]
            # Use diffusion params as default U-Net config (for patch size calculation)
            unet_config = config.get('diffusion_stages', {}).get(self.mode_target, {}).get('unet_config', {})
            if not unet_config:
                unet_config = get_default_configs(self.vae_groups, self.diffusion_stages)[1]
            
        elif self.mode_type == 'diffusion':
            # Diffusion mode: use the target stage's config
            if self.mode_target not in self.diffusion_stages:
                raise ValueError(
                    f"Diffusion stage '{self.mode_target}' not found in config. "
                    f"Available stages: {list(self.diffusion_stages.keys())}"
                )
            stage_config = self.diffusion_stages[self.mode_target]
            pred_group = stage_config.get('prediction_group')
            
            if pred_group not in self.vae_groups:
                raise ValueError(
                    f"Prediction group '{pred_group}' not found in VAE groups. "
                    f"Available: {list(self.vae_groups.keys())}"
                )
            
            vae_config = self.vae_groups[pred_group]
            unet_config = stage_config.get('unet_config', {})
            
        else:
            # Default mode: use first available configs or defaults
            vae_config, unet_config = get_default_configs(self.vae_groups, self.diffusion_stages)
        
        # Basic parameters
        big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
        im_channels = dataset_config.get('im_channels', 3)
        min_valid_percent = dataset_config.get('min_valid_percent', 90)
        self.big_data_storage_path = big_data_storage_path

        # Latent space configuration - store latent maps per VAE group
        # Structure: {'group_name': [list of latent file paths]}
        self.group_latents = {}

        # Compute patch and latent sizes using mode-specific configs
        # use_latents=True for diffusion mode (always works in latent space)
        use_latents_for_sizing = (self.mode_type == 'diffusion')
        patch_size, latent_size, vae_downsample_factor, unet_downsample_factor, total_divisor = compute_patch_and_latent_sizes(
            dataset_config,
            vae_config,
            unet_config,
            use_latents=use_latents_for_sizing,
            self=self
        )
        
        # Store parameters
        self.split = split
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.stride_overlap = dataset_config.get('stride_overlap', 2)
        self.stride = int(patch_size // self.stride_overlap)  # compute stride based on overlap
        self.im_channels = im_channels
        self.min_valid_percent = min_valid_percent

        # Build list of ALL layers from global config (ordered)
        self.all_layer_names = list(self.layers_registry.keys())
        
        # Compute total channels across all layers
        self.total_channels = sum(
            count_layer_channels(layer_config) 
            for layer_config in self.layers_registry.values()
        )
        
        # Inpainting configuration (top-level, shared across all stages)
        self.inpainting_config = config.get('inpainting_params', {
            'type': 'random_square',
            'size_px': 64
        })
        
        # Alias for backward compatibility
        self.hole_config = self.inpainting_config
        
        # Select regions based on split
        train_regions = dataset_config.get('train_regions', ['Dresden', 'Hamburg', 'Stuttgart'])
        eval_regions = dataset_config.get('eval_regions', ['Leipzig'])
        self.regions = train_regions if self.split == 'train' else eval_regions
        
        # Store datasets and data layers per region
        self.datasets = {}
        self.data_layers_per_region = {}
        
        # Store statistics for normalization
        # Structure: {'layer_name': {'min': float, 'max': float, 'mean': float, 'std': float, 'q01': float, 'q99': float}}
        self.layer_stats = {}
        
        # Store runtime statistics (for tracking)
        self.stats = {
            "inpainting_mask": []
        }
        
        # Cache directory setup
        if cache_dir is None:
            task_name = config['train_params']['task_name']
            cache_dir = Path(big_data_storage_path) / "processed" / task_name / self.mode
            
        self.cache_dir = Path(cache_dir)
        self.use_cached_patches = use_cached_patches
        
        # Initialize data loading strategy
        if use_cached_patches:
            print(f"\n{'='*60}")
            print(f"Attempting to load cached patches from: {self.cache_dir}")
            print(f"{'='*60}")
            
            if self._load_cached_patches():
                print(f"✓ Successfully loaded {len(self.patches)} cached patches")
            else:
                print(f"⚠ No cached patches found. Falling back to on-the-fly loading")
                print(f"⚠ Run `prepare_cached_patches()` to generate cache for faster training")
                self.use_cached_patches = False
                self._initialize_xarray_loading()
        else:
            print(f"\n{'='*60}")
            print(f"Using on-the-fly Xarray loading (slower)")
            print(f"{'='*60}")
            self._initialize_xarray_loading()
            
        # Load latents for diffusion mode (config-driven)
        if self.mode_type == 'diffusion':
            self._load_diffusion_latents()
        
        # Final summary
        self._print_summary()
        
    def _initialize_xarray_loading(self):
        """Initialize on-the-fly Xarray data loading"""
        processed_data_path = self.data_config.get("big_data_storage_path", "/work/zt75vipu-master/data") + "/processed"
        zarr_name = self.dataset_config.get('zarr_name', 'input_data.zarr')
    
        # Load datasets for all regions
        for region in self.regions:
            region_zarr_path = os.path.join(processed_data_path, region.lower(), zarr_name)
            print(f"Loading zarr dataset from {region_zarr_path}...")
            self.datasets[region] = xr.open_zarr(region_zarr_path, consolidated=True)
        
        # Compute global statistics for normalization
        self._compute_layer_statistics()
        
        # Load patches
        self.patches = self._load_patches()
        
    def _compute_layer_statistics(self):
        """
        Compute global statistics for each layer across all regions and dates.
        
        This ensures consistent normalization across all patches.
        Statistics are stored in self.layer_stats.
        """
        print(f"\n{'='*60}")
        print("Computing global layer statistics for normalization...")
        print(f"{'='*60}")
        
        rgb_layer = get_layer_info(self.layers_registry, 'rgb')
        rgb_layer_name = rgb_layer.get('layer', 'planetscope_sr_4band')
        
        for layer_name in self.all_layer_names:
            if layer_name == 'inpainting_mask':
                # Skip inpainting mask (generated on-the-fly)
                continue
            
            layer_config = self.layers_registry[layer_name]
            source_layer = get_layer_info(self.layers_registry, layer_name).get('layer', layer_name)
            layer_channels = layer_config.get('channels', None)
            
            # Check if this layer needs normalization
            normalize_method = layer_config.get('normalize', None)
            if normalize_method is None:
                continue
            if normalize_method == 'custom':
                normalize_params = layer_config.get('normalize_params', {})
                if 'min' in normalize_params and 'max' in normalize_params:
                    # Custom min/max provided - skip stats computation
                    print(f"  ✓ Skipping '{layer_name}' (custom min/max provided)")
                    continue
            
            print(f"\nComputing statistics for '{layer_name}' (source: {source_layer})...")
            
            # Collect data across all regions and dates
            all_data = []
            
            for region in self.regions:
                merged_xs = self.datasets[region]
                
                if source_layer not in merged_xs:
                    print(f"  ⚠ Layer '{source_layer}' not found in {region}")
                    continue
                
                # Get valid dates
                valid_dates = (
                    merged_xs[rgb_layer_name]
                    .notnull()
                    .sum(dim=['x', 'y']) > 0
                ).any(dim='channel').compute()
            
                valid_dates = merged_xs['time'].where(valid_dates, drop=True).values
            
                if len(valid_dates) == 0:
                    print(f"No valid dates found for region {region}")
                    continue
                
                for date in valid_dates:
                    date_data = merged_xs.sel(time=date)
                    
                    if source_layer not in date_data:
                        continue
                    
                    # Extract layer data
                    if layer_channels is not None:
                        layer_da = date_data[source_layer].sel(channel=layer_channels)
                    else:
                        layer_da = date_data[source_layer]
                    
                    # Load data (materialize from dask)
                    data_np = layer_da.values
                    
                    # Remove NaN values
                    data_np = data_np[~np.isnan(data_np)]
                    
                    if len(data_np) > 0:
                        all_data.append(data_np)
            
            if not all_data:
                print(f"  ⚠ No valid data found for '{layer_name}'")
                continue
            
            # Concatenate all data
            all_data_concat = np.concatenate(all_data)
            
            # Compute statistics
            stats = {
                'min': float(np.min(all_data_concat)),
                'max': float(np.max(all_data_concat)),
                'mean': float(np.mean(all_data_concat)),
                'std': float(np.std(all_data_concat)),
                'q01': float(np.percentile(all_data_concat, 1)),
                'q99': float(np.percentile(all_data_concat, 99)),
                'q02': float(np.percentile(all_data_concat, 2)),
                'q98': float(np.percentile(all_data_concat, 98)),
            }
            
            self.layer_stats[layer_name] = stats
            
            print(f"  ✓ min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            print(f"    q01={stats['q01']:.3f}, q99={stats['q99']:.3f}")
        
        print(f"\n✓ Computed statistics for {len(self.layer_stats)} layers")
        print(f"{'='*60}\n")
    
    def _load_cached_patches(self) -> bool:
        """
        Load pre-saved patches from disk.
        
        Returns:
            True if successful, False if cache doesn't exist
        """
        metadata_path = self.cache_dir / f"patches_metadata_{self.split}.csv"
        
        if not metadata_path.exists():
            return False
        
        # Load patch metadata
        metadata_df = pd.read_csv(metadata_path)
        
        # Validate cache matches current configuration
        if not self._validate_cache_config(metadata_df):
            print("⚠ Cached patches configuration mismatch. Regeneration recommended.")
            return False
        
        # Load patch file paths
        self.patches = [
            (row['y'], row['x'], row['region'], row['cache_index'])
            for _, row in metadata_df.iterrows()
        ]
        
        print(f"✓ Loaded {len(self.patches)} patches from cache")
        return True
    
    def _validate_cache_config(self, metadata_df: pd.DataFrame) -> bool:
        """Validate that cached patches match current configuration"""
        if len(metadata_df) == 0:
            return False
        
        # Check patch size
        first_patch_path = self.cache_dir / f"patch_{metadata_df.iloc[0]['cache_index']}.pt"
        if first_patch_path.exists():
            sample_data = torch.load(first_patch_path)
            sample_image = sample_data['image'] if isinstance(sample_data, dict) else sample_data
            
            if sample_image.shape[-1] != self.patch_size or sample_image.shape[-2] != self.patch_size:
                print(f"⚠ Patch size mismatch: cached={sample_image.shape[-2:]}, config={self.patch_size}")
                return False
        
        # Check regions match
        cached_regions = set(metadata_df['region'].unique())
        config_regions = set(self.regions)
        
        if cached_regions != config_regions:
            print(f"⚠ Region mismatch: cached={cached_regions}, config={config_regions}")
            return False
        
        return True
    
    def _load_group_latents(self, group_name: str, reconcile: bool = True) -> Optional[List[str]]:
        """
        Load latents for a specific VAE group.
        
        Args:
            group_name: Name of the VAE group (e.g., 'satellite', 'semantic')
            reconcile: Whether to reconcile latent count with patch count
            
        Returns:
            List of latent file paths, or None if not found/failed
        """
        if group_name not in self.vae_groups:
            print(f"⚠ VAE group '{group_name}' not found in config")
            return None
        
        group_config = self.vae_groups[group_name]
        latents_dir = group_config.get('latents_dir', f'{group_name}_latents')
        stats_dir = group_config.get('stats_dir', f'{group_name}_stats')
        
        # Build full path to latents directory
        results_dir = Path(self.big_data_storage_path) / "results" / self.config['train_params']['task_name']
        latent_path = results_dir / latents_dir
        
        if not latent_path.exists():
            print(f"⚠ Latents directory does not exist: {latent_path}")
            print(f"  Run VAE training for group '{group_name}' first")
            return None
        
        print(f"Loading latents for group '{group_name}' from {latent_path}...")
        latent_files = load_latents(str(latent_path))
        
        if len(latent_files) == 0:
            print(f"⚠ No latent files found for group '{group_name}'")
            return None
        
        print(f"✓ Found {len(latent_files)} latent files for group '{group_name}'")
        
        # Reconcile with patches if requested
        if reconcile and len(latent_files) != len(self.patches):
            print(f"⚠ Latent count mismatch: {len(latent_files)} latents vs {len(self.patches)} patches")
            latent_files = self._reconcile_latents_with_patches(latent_files, group_name, stats_dir)
        
        return latent_files if latent_files and len(latent_files) > 0 else None
    
    def _reconcile_latents_with_patches(self, latent_files: List[str], group_name: str, stats_dir: str) -> Optional[List[str]]:
        """
        Reconcile latent files with current patches using training stats.
        
        Args:
            latent_files: List of latent file paths
            group_name: Name of VAE group (for logging)
            stats_dir: Directory containing training stats
            
        Returns:
            Filtered list of latent files matching patches, or None if failed
        """
        print(f"Attempting to reconcile latents for group '{group_name}' using VAE training stats...")
        
        
        
        results_dir = Path(self.big_data_storage_path) / "results" / self.config['train_params']['task_name']
        stats_csv_path = results_dir / stats_dir / "inpainting_mask_stats_train.csv"
        
        if not stats_csv_path.exists():
            print(f"⚠ Stats file not found: {stats_csv_path}")
            print(f"  Cannot reconcile - patches and latents will be mismatched")
            return None
        
        filtered_patches, filtered_latents, comparison_results = reconcile_patches_with_latents(
            stats_csv_path=stats_csv_path,
            current_patches=self.patches,
            latent_files=latent_files,
            verbose=True
        )
        
        if len(filtered_patches) == 0:
            print(f"⚠ No matching patches found for group '{group_name}'")
            return None
        
        # Update patches (only on first reconciliation)
        if len(self.patches) != len(filtered_patches):
            print(f"✓ Updating dataset patches to match latents: {len(self.patches)} -> {len(filtered_patches)}")
            self.patches = filtered_patches
        
        print(f"✓ Successfully reconciled {len(filtered_latents)} latents for group '{group_name}'")
        return filtered_latents
    
    def _load_diffusion_latents(self):
        """
        Load all required latents for diffusion training.
        
        Loads:
        - Prediction latents (for the prediction group)
        - Conditioning latents (for each latent-space conditioning group)
        """
        if self.mode_type != 'diffusion':
            return
        
        stage_config = self.diffusion_stages[self.mode_target]
        pred_group = stage_config.get('prediction_group')
        conditioning_config = stage_config.get('conditioning', {})
        
        print(f"\n{'='*60}")
        print(f"Loading latents for diffusion stage '{self.mode_target}'")
        print(f"{'='*60}")
        
        # Load prediction latents
        print(f"\nPrediction group: '{pred_group}'")
        pred_latents = self._load_group_latents(pred_group, reconcile=True)
        
        if pred_latents is None:
            raise RuntimeError(
                f"Failed to load prediction latents for group '{pred_group}'. "
                f"Run VAE training for this group first."
            )
        
        self.group_latents[pred_group] = pred_latents
        
        # Load conditioning latents (latent-space conditioning)
        latent_cond_groups = conditioning_config.get('latent_space', [])
        
        if latent_cond_groups:
            print(f"\nLoading {len(latent_cond_groups)} latent-space conditioning groups...")
            
            for cond_spec in latent_cond_groups:
                cond_group = cond_spec['group']
                print(f"\nConditioning group: '{cond_group}'")
                
                cond_latents = self._load_group_latents(cond_group, reconcile=True)
                
                if cond_latents is None:
                    raise RuntimeError(
                        f"Failed to load conditioning latents for group '{cond_group}'. "
                        f"Run VAE training for this group first."
                    )
                
                self.group_latents[cond_group] = cond_latents
        
        print(f"\n✓ Successfully loaded latents for {len(self.group_latents)} VAE groups")
        print(f"{'='*60}\n")
                    
    def prepare_cached_patches(self) -> None:
        """
        Pre-save all patches to disk for faster training.
        
        This method:
        1. Extracts all patches from Xarray
        2. Normalizes and processes them
        3. Saves as individual .pt files
        4. Creates metadata CSV for index mapping
        
        Args:
            num_workers: Number of parallel workers for processing
        """
        if not hasattr(self, 'datasets') or not self.datasets:
            raise RuntimeError("Cannot cache patches: Xarray datasets not loaded. Set use_cached_patches=False first.")
        
        print(f"\n{'='*60}")
        print(f"Preparing cached patches")
        print(f"{'='*60}")
        print(f"Output directory: {self.cache_dir}")
        print(f"Total patches to process: {len(self.patches)}")
        print(f"{'='*60}\n")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata for tracking
        metadata_records = []
        
        # Process and save each patch
        for cache_idx, (y, x, region) in enumerate(tqdm(self.patches, desc="Caching patches")):
            try:
                # Extract patch data (reuse existing logic)
                patch_data = self._extract_patch_from_xarray(y, x, region, cache_idx)
                
                # Save patch
                patch_path = self.cache_dir / f"patch_{self.split}_{cache_idx}.pt"
                torch.save(patch_data, patch_path)
                
                # Record metadata
                metadata_records.append({
                    'cache_index': cache_idx,
                    'y': y,
                    'x': x,
                    'region': region,
                    'patch_file': str(patch_path.name)
                })
                
            except Exception as e:
                print(f"⚠ Failed to cache patch {cache_idx} at (y={y}, x={x}, region={region}): {e}")
                continue
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_records)
        metadata_path = self.cache_dir / f"patches_metadata_{self.split}.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"\n✓ Successfully cached {len(metadata_records)} patches")
        print(f"✓ Metadata saved to: {metadata_path}")
        print(f"✓ Total disk usage: ~{self._estimate_cache_size()} MB\n")
    
    def _extract_patch_from_xarray(
        self, 
        y: int, 
        x: int, 
        region: str, 
        index: int
    ) -> Dict[str, torch.Tensor]:
        """
        Extract and process a single patch from Xarray.
        
        Creates a unified tensor containing ALL layers from the global layer config,
        properly transformed and normalized according to each layer's configuration.
        Includes dynamically generated inpainting mask.
        
        Returns:
            Dictionary with:
            - 'image': [C, H, W] tensor of all layers (including inpainting_mask)
            - 'meta': metadata dict with spatial_names list
        """
        ps = self.patch_size
        data_layers = self.data_layers_per_region[region]
        
        # Extract all layers and stack them
        layer_tensors = []
        layer_names = []  # Track which layer each channel belongs to
        channel_names = []  # Track full channel names (e.g., 'rgb:red', 'buildings')
        
        for layer_name in self.all_layer_names:
            layer_config = self.layers_registry[layer_name]
            source_layer = get_layer_info(self.layers_registry, layer_name).get('layer', layer_name)
            
            # Special handling for inpainting_mask (generated on-the-fly)
            if layer_name == 'inpainting_mask':
                # Get street blocks if available for mask generation
                street_blocks_layer = None
                if 'street_blocks' in data_layers and self.inpainting_config.get('type') == 'street_blocks':
                    street_blocks_layer = data_layers['street_blocks'].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                
                # Create inpainting mask
                patch_info = {
                    'index': index,
                    'region': region,
                    'y': y,
                    'x': x,
                    'split': self.split
                }
                inpaint_mask = self._create_inpainting_mask(ps, ps, street_blocks_layer, patch_info)
                
                # Convert to CHW tensor
                inpaint_mask = self._to_chw(inpaint_mask)
                layer_tensor = torch.from_numpy(inpaint_mask).float()
                layer_tensors.append(layer_tensor)
                
                # Single channel layer
                layer_names.append('inpainting_mask')
                channel_names.append('inpainting_mask')
                continue
            
            if source_layer not in data_layers:
                print(f"⚠ Warning: Layer '{source_layer}' not found in dataset for region {region}")
                continue
            
            # Extract patch
            layer_data = data_layers[source_layer].isel(
                y=slice(y, y+ps),
                x=slice(x, x+ps)
            ).values.astype(np.float32)
            
            # Get global statistics for this layer (if available)
            layer_statistics = self.layer_stats.get(layer_name, None)
            
            # Apply transformations (filtering, normalization with global stats)
            layer_data = apply_layer_transform(layer_data, layer_config, layer_statistics)
            
            # Convert to CHW format
            layer_data = self._to_chw(layer_data)
            
            # Convert to tensor
            layer_tensor = torch.from_numpy(layer_data).float()
            layer_tensors.append(layer_tensor)
            
            # Track channel names and layer names
            formatted_names = get_channel_names(layer_name, layer_config)
            num_channels = len(formatted_names)
            
            # For multi-channel layers, each channel still belongs to the same layer
            layer_names.extend([layer_name] * num_channels)
            channel_names.extend(formatted_names)
        
        # Stack all layers into one tensor
        unified_patch = torch.cat(layer_tensors, dim=0)  # [C_total, H, W]
        
        # Create metadata
        metadata = {
            'y': y,
            'x': x,
            'time': str(data_layers['date']),
            'region': region,
            'layer_names': layer_names,
            'channel_names': channel_names,
            'patch_index': index
        }
        
        return {
            'image': unified_patch,
            'meta': metadata
        }
    
    def _estimate_cache_size(self) -> int:
        """Estimate cache directory size in MB"""
        total_size = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.rglob('*.pt'):
                total_size += f.stat().st_size
        return total_size // (1024 * 1024)
    
    def _load_patches(self):
        """
        Pre-compute valid patch locations from the dataset
        """
        
        rgb_layer = get_layer_info(self.layers_registry, 'rgb')
        
        rgb_layer_name = rgb_layer.get('layer', 'planetscope_sr_4band')
        rgb_layer_channels = rgb_layer.get('channels', ['blue', 'green', 'red'])
        
        all_patches = []
        
        for region in self.regions:
            print(f"\nProcessing region: {region}")
            merged_xs = self.datasets[region]
        
            # Get valid dates with planetscope data
            valid_dates = (
                merged_xs[rgb_layer_name]
                .notnull()
                .sum(dim=['x', 'y']) > 0
            ).any(dim='channel').compute()
        
            valid_dates = merged_xs['time'].where(valid_dates, drop=True).values
        
            if len(valid_dates) == 0:
                print(f"No valid dates found for region {region}")
                continue
        
            # For now, first valid date (can be extended to use multiple dates)
            selected_date = valid_dates[0]
            print(f"Using date: {selected_date} for region {region}")
            
            # Select data for this date 
            date_data = merged_xs.sel(time=selected_date)
            
            # Get satellite image and compute validity mask
            img_da = date_data[rgb_layer_name].sel(channel=rgb_layer_channels)
            valid_mask = (~img_da.isnull()).all(dim='channel').compute()
            
            # Handle reflectance scaling
            if img_da.max() > 20:
                img_da = (img_da / 10000.0).astype(np.float32)
            
            # Store data layers for this region
            data_layers = {
                rgb_layer_name: img_da,
                'valid_mask': valid_mask,
                'date': selected_date
            }
        
            # Add optional layers
            for layer in self.all_layer_names:
                if layer in ['rgb', 'inpainting_mask']:
                    continue  # already handled
                
                layer_info = get_layer_info(self.layers_registry, layer)
                source_layer = layer_info.get('layer', layer)
                layer_channels = layer_info.get('channels', None)
                
                if source_layer in date_data:
                    if layer_channels is not None:
                        layer_da = date_data[source_layer].sel(channel=layer_channels).compute()
                    else:
                        layer_da = date_data[source_layer].compute()
                    data_layers[source_layer] = layer_da
                else:
                    print(f"⚠ Layer '{source_layer}' not found in dataset for region {region}")
                    
            # Store data layers for region
            self.data_layers_per_region[region] = data_layers
        
            # Compute valid patches based on min valid percent of data
            H, W = valid_mask.shape
            min_valid_pixels = int((self.patch_size ** 2) * (self.min_valid_percent / 100))
        
            region_patches = []
            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    valid_count = valid_mask[y:y+self.patch_size, x:x+self.patch_size].sum()
                    if valid_count >= min_valid_pixels:
                        region_patches.append((y, x, region))
            
            print(f"Found {len(region_patches)} valid patches for region {region}")
            all_patches.extend(region_patches)
                    
        print(f"\nTotal {self.split} patches across all regions: {len(all_patches)}")
        return all_patches
    
    def _create_inpainting_mask(self, H, W, street_blocks_layer=None, patch_info=None):
        """
        Create inpainting hole mask (wrapper for pure function).
        """
        return create_inpainting_mask(
            H=H,
            W=W,
            hole_config=self.hole_config,
            street_blocks_layer=street_blocks_layer,
            patch_info=patch_info,
            stats_list=self.stats["inpainting_mask"]
        )
    
    def __len__(self):
        return len(self.patches)
    
    def _to_chw(self, arr):
        """Accepts xarray or numpy. Returns float32 [C,H,W]."""
        # get a small window first, then materialize:
        if hasattr(arr, "values"):   # xarray.DataArray or dask-backed
            arr = arr.values
        arr = np.asarray(arr)

        if arr.ndim == 2:
            # [H,W] -> [1,H,W]
            arr = arr[None, ...]
        elif arr.ndim == 3:
            # assume either [C,H,W] (planetscope) or [H,W,C] --> safe check
            H, W = arr.shape[-2], arr.shape[-1]
            if arr.shape[0] not in (1,3) and arr.shape[-1] in (1,3) and arr.shape[-2] == H:
                # looks like HWC -> CHW
                arr = arr.transpose(2,0,1)
            # else: already CHW
        else:
            raise ValueError(f"Unexpected shape {arr.shape}, need 2D or 3D")

        return arr.astype(np.float32, copy=False)

    def __getitem__(self, index: int):
        """
        Get a single training sample.
        
        Behavior depends on initialization:
        - If use_cached_patches=True: Load from disk
        - If use_cached_patches=False: Extract from Xarray
        - If use_latents=True: Return latent + conditioning
        """
        if self.use_cached_patches:
            return self._getitem_cached(index)
        else:
            return self._getitem_xarray(index)
        
    def _compose_batch_by_mode(self, unified_image: torch.Tensor, patch_data: Dict, index: int):
        """
        Compose dataset output based on mode (shared logic for cached and on-the-fly loading).
        
        Args:
            unified_image: [C_total, H, W] tensor with all layers
            patch_data: Dictionary with 'meta' containing 'layer_names', 'channel_names' and other metadata
            index: Dataset index (for loading latents in diffusion mode)
            
        Returns:
            - default mode: (image, conditioning_dict) or (image, {'image': None, 'meta': ...})
            - vae mode: (image, metadata_dict)
            - diffusion mode: (pred_latent, conditioning_dict)
        """
        layer_names = patch_data['meta']['layer_names']
        channel_names = patch_data['meta']['channel_names']
            
        if self.mode_type == 'default':
            # Default mode: RGB as image, everything else as conditioning
            # Use layer registry to find RGB layer
            rgb_layer_matches = get_layer_channels_from_names(channel_names, 'rgb')
            
            if len(rgb_layer_matches) == 0:
                raise ValueError(
                    f"Default mode requires 'rgb' layer, but it was not found in patch. "
                    f"Available layers: {set(layer_names)}"
                )
            
            # Extract RGB indices and names
            rgb_indices = [idx for idx, _ in rgb_layer_matches]
            rgb_channel_names = [name for _, name in rgb_layer_matches]
            rgb_layer_names = [layer_names[idx] for idx in rgb_indices]
            
            # Get conditioning indices (everything except RGB)
            cond_indices = [idx for idx in range(len(channel_names)) if idx not in rgb_indices]
            cond_channel_names = [channel_names[idx] for idx in cond_indices]
            cond_layer_names = [layer_names[idx] for idx in cond_indices]
            
            # Extract RGB image
            image = unified_image[rgb_indices]  # [C_rgb, H, W]
            
            # Extract conditioning
            if len(cond_indices) > 0:
                conditioning = unified_image[cond_indices]  # [C_cond, H, W]
                cond_meta = patch_data['meta'].copy()
                cond_meta['layer_names'] = cond_layer_names
                cond_meta['channel_names'] = cond_channel_names
                return image, {'image': conditioning, 'meta': cond_meta}
            else:
                # No conditioning channels
                image_meta = patch_data['meta'].copy()
                image_meta['layer_names'] = rgb_layer_names
                image_meta['channel_names'] = rgb_channel_names
                return image, {'image': None, 'meta': image_meta}
        
        elif self.mode_type == 'vae':
            vae_config = self.vae_groups[self.mode_target]
            target_layers = vae_config.get('layers', [])
            
            if len(target_layers) == 0:
                raise ValueError(
                    f"VAE group '{self.mode_target}' has no layers defined"
                )
            
            # Collect all channels for the target layers
            target_indices = []
            target_channel_names = []
            target_layer_names = []
            
            for target_layer in target_layers:
                layer_matches = get_layer_channels_from_names(channel_names, target_layer)
                if len(layer_matches) == 0:
                    raise ValueError(
                        f"VAE group '{self.mode_target}' requires layer '{target_layer}', "
                        f"but it was not found in patch. Available: {set(layer_names)}"
                    )
                
                for idx, name in layer_matches:
                    target_indices.append(idx)
                    target_channel_names.append(name)
                    target_layer_names.append(layer_names[idx])
            
            # Extract target layers
            image = unified_image[target_indices]  # [C_target, H, W]
            
            # Create metadata
            image_meta = patch_data['meta'].copy()
            image_meta['layer_names'] = target_layer_names
            image_meta['channel_names'] = target_channel_names
            
            return image, {'image': None,'meta': image_meta}
        
        else:  # self.mode_type == 'diffusion'
            # Diffusion mode: Load prediction latent + conditioning (pixel + latent space)
            stage_config = self.diffusion_stages[self.mode_target]
            pred_group = stage_config.get('prediction_group')
            conditioning_config = stage_config.get('conditioning', {})
            
            # Load prediction latent from pre-loaded group latents
            if pred_group not in self.group_latents:
                raise RuntimeError(
                    f"Prediction latents for group '{pred_group}' not loaded. "
                    f"This should have been loaded in _load_diffusion_latents()"
                )
            
            pred_latent_path = self.group_latents[pred_group][index]
            pred_latent = load_single_latent(pred_latent_path, device=None)
            
            # Build conditioning dictionary
            cond = {'meta': patch_data['meta'].copy()}
            
            # Use pre-computed latent resolution (all VAEs must have same downsampling)
            latent_h = self.latent_size
            latent_w = self.latent_size
            
            # Add pixel-space conditioning (downsampled to latent resolution)
            pixel_cond_list = []
            pixel_cond_names = []
            
            for cond_spec in conditioning_config.get('pixel_space', []):
                if cond_spec['type'] == 'inpainting_mask':
                    # Find mask in unified patch
                    mask_matches = get_layer_channels_from_names(channel_names, 'inpainting_mask')
                    if len(mask_matches) > 0:
                        mask_idx = mask_matches[0][0]
                        mask_full = unified_image[mask_idx:mask_idx+1, :, :]  # [1, H, W]
                        
                        # Downsample to latent resolution
                        mask_latent = mask_full.unsqueeze(0)  # [1, 1, H, W]
                        mask_latent = torch.nn.functional.interpolate(
                            mask_latent,
                            size=(latent_h, latent_w),
                            mode='nearest'
                        ).squeeze(0)  # [1, H_latent, W_latent]
                        
                        pixel_cond_list.append(mask_latent)
                        pixel_cond_names.append('inpainting_mask')
            
            if pixel_cond_list:
                cond['image'] = torch.cat(pixel_cond_list, dim=0)
                cond['pixel_space_names'] = pixel_cond_names
            
            # Add latent-space conditioning (load from pre-loaded group latents)
            for cond_spec in conditioning_config.get('latent_space', []):
                group_name = cond_spec['group']
                
                if group_name not in self.group_latents:
                    raise RuntimeError(
                        f"Conditioning latents for group '{group_name}' not loaded. "
                        f"This should have been loaded in _load_diffusion_latents()"
                    )
                
                # Load conditioning latent
                cond_latent_path = self.group_latents[group_name][index]
                cond_latent = load_single_latent(cond_latent_path, device=None)
                cond[group_name] = cond_latent
            
            return pred_latent, cond
    
    def _getitem_cached(self, index: int):
        """
        Load pre-saved patch from disk and compose based on mode.
        
        Returns:
            - default mode: (image, conditioning_dict) or just image
            - vae mode: (image, metadata_dict)
            - diffusion mode: (pred_latent, conditioning_dict)
        """
        y, x, region, cache_idx = self.patches[index]
        
        # Load from cache
        patch_path = self.cache_dir / f"patch_{self.split}_{cache_idx}.pt"
        patch_data = torch.load(patch_path)
        
        # Compose using shared logic
        unified_image = patch_data['image']
        return self._compose_batch_by_mode(unified_image, patch_data, index)
    
    def _getitem_xarray(self, index: int):
        """
        Extract patch on-the-fly from Xarray and compose based on mode.
        
        This method mirrors _getitem_cached but extracts data from Xarray instead of loading from disk.
        
        Returns:
            - default mode: (image, conditioning_dict) or just image
            - vae mode: (image, metadata_dict)
            - diffusion mode: (pred_latent, conditioning_dict)
        """
        y, x, region = self.patches[index]
        
        # Extract unified patch from Xarray (same as when caching)
        patch_data = self._extract_patch_from_xarray(y, x, region, index)
        
        # Compose using shared logic
        unified_image = patch_data['image']
        return self._compose_batch_by_mode(unified_image, patch_data, index)
    
    def _print_summary(self):
        """Print dataset configuration summary"""
        print(f"\n{'='*60}")
        print(f"Dataset Configuration Summary")
        print(f"{'='*60}")
        print(f"Mode: {self.mode}")
        if hasattr(self, 'mode_type') and self.mode_type != 'default':
            print(f"  Type: {self.mode_type}")
            print(f"  Target: {self.mode_target}")
        print(f"Split: {self.split}")
        print(f"Loading mode: {'Cached patches' if self.use_cached_patches else 'On-the-fly Xarray'}")
        print(f"Total patches: {len(self.patches)}")
        
        # Print latent information for diffusion mode
        if self.mode_type == 'diffusion' and self.group_latents:
            print(f"\nLoaded latents for {len(self.group_latents)} VAE groups:")
            for group_name, latent_files in self.group_latents.items():
                print(f"  - {group_name}: {len(latent_files)} latents")
        
        print(f"Patch size: {self.patch_size}x{self.patch_size}")
        
        # Print layer information
        print(f"\nUnified patch layers ({self.total_channels} total channels):")
        for layer_name in self.all_layer_names:
            layer_config = self.layers_registry[layer_name]
            num_channels = count_layer_channels(layer_config)
            layer_type = layer_config.get('type', 'continuous')
            print(f"  - {layer_name}: {num_channels} channel(s) [{layer_type}]")
        
        if self.use_cached_patches:
            print(f"\nCache directory: {self.cache_dir}")
        print(f"{'='*60}\n")
    
    def save_stats(self, save_path):
        """
        Save dataset statistics to CSV files
        """
        
        import pandas as pd
        for stat_name, records in self.stats.items():
            save_path = f"{save_path}/{stat_name}_stats_{self.split}.csv"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if records:
                df = pd.DataFrame(records)
                df.to_csv(save_path, index=False)
                print(f"Saved {stat_name} stats to {save_path}")
        
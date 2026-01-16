# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main

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
from model.utils.read_yaml import get_nested
from model.utils.diffusion_utils import load_single_latent
from model.utils.data_utils import apply_layer_transform
from model.utils.config_utils import compute_patch_and_latent_sizes, get_default_configs
from model.utils.layer_config import (
    get_layer_info,
    count_layer_channels,
    get_channel_names,
    get_layer_channels_from_names,
)
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
                 use_latents=False, 
                 latent_path=None,
                 use_cached_patches: bool = True,
                 cache_dir: Optional[str] = None,
                 mode: str = 'default',
        ):
        """
        :param split: 'train' or 'val'
        :param use_latents: whether to use pre-computed latents from autoencoder
        :param latent_path: path to latent files
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

        # Latent space configuration
        self.latent_maps = None
        self.latent_cond_maps = None
        self.latent_path = latent_path
        self.use_latents = bool(use_latents)

        # Compute patch and latent sizes using mode-specific configs
        patch_size, latent_size, vae_downsample_factor, unet_downsample_factor, total_divisor = compute_patch_and_latent_sizes(
            dataset_config,
            vae_config,
            unet_config,
            use_latents=self.use_latents,
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
        
        # store statistics
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
            
        # Load latents if specified
        if use_latents and latent_path is not None:
            self._load_and_reconcile_latents(big_data_storage_path)
        elif use_latents and latent_path is None:
            print('⚠ use_latents=True but no latent_path provided, using raw images')
            self.use_latents = False
            self.latent_maps = None
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
        
        # Load patches
        self.patches = self._load_patches()
        
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
    
    def _load_and_reconcile_latents(self, big_data_storage_path: str):
        """Load VAE latents (both prediction and conditioning) and reconcile with patches"""
        print(f'Loading latents from {self.latent_path}...')
        
        # Load prediction latents
        latent_maps_pred = load_latents(self.latent_path, prefix='pred')
        
        # Load conditioning latents (optional, for two-VAE setup)
        latent_maps_cond = load_latents(self.latent_path, prefix='cond')
        
        # Check if we have conditioning latents
        has_cond_latents = len(latent_maps_cond) > 0
        
        if has_cond_latents:
            print(f'✓ Found {len(latent_maps_pred)} prediction latents and {len(latent_maps_cond)} conditioning latents')
            
            # SAFETY CHECK: Verify indices match
            pred_indices = set([int(Path(p).stem.split('_')[-1]) for p in latent_maps_pred])
            cond_indices = set([int(Path(p).stem.split('_')[-1]) for p in latent_maps_cond])
            
            missing_in_cond = pred_indices - cond_indices
            missing_in_pred = cond_indices - pred_indices
            
            if missing_in_cond:
                print(f'⚠ WARNING: {len(missing_in_cond)} prediction latents have no matching conditioning latent')
                print(f'  Missing indices: {sorted(list(missing_in_cond))[:10]}...')
            
            if missing_in_pred:
                print(f'⚠ WARNING: {len(missing_in_pred)} conditioning latents have no matching prediction latent')
                print(f'  Missing indices: {sorted(list(missing_in_pred))[:10]}...')
            
            # Use only matching indices
            matching_indices = pred_indices & cond_indices
            if len(matching_indices) < len(pred_indices):
                print(f'ℹ️  Using {len(matching_indices)} matching latent pairs (filtered from {len(pred_indices)} prediction latents)')
                latent_maps_pred = [p for p in latent_maps_pred if int(Path(p).stem.split('_')[-1]) in matching_indices]
                latent_maps_cond = [p for p in latent_maps_cond if int(Path(p).stem.split('_')[-1]) in matching_indices]
        else:
            print(f'✓ Found {len(latent_maps_pred)} prediction latents (no separate conditioning latents)')
        
        # Use prediction latents as primary
        latent_maps = latent_maps_pred if len(latent_maps_pred) > 0 else load_latents(self.latent_path)
        
        if len(latent_maps) == len(self.patches):
            # Perfect match
            self.use_latents = True
            self.latent_maps = latent_maps
            self.latent_cond_maps = latent_maps_cond if has_cond_latents else None
            print(f'✓ Latents match {len(self.patches)} patches')
        else:
            # Mismatch - reconcile
            print(f'⚠ Latents size mismatch: found {len(latent_maps)} latents but need {len(self.patches)} patches')
            print(f'⚠ Attempting to reconcile using VAE training stats...')
            
            results_dir = Path(big_data_storage_path) / "results" / self.config['train_params']['task_name']
            stats_csv_path = results_dir / "vae_ddp_stats" / "inpainting_mask_stats_train.csv"
            
            filtered_patches, filtered_latents, comparison_results = reconcile_patches_with_latents(
                stats_csv_path=stats_csv_path,
                current_patches=self.patches,
                latent_files=latent_maps,
                verbose=True
            )
            
            if len(filtered_patches) > 0:
                self.patches = filtered_patches
                self.latent_maps = filtered_latents
                # Filter conditioning latents to match
                if has_cond_latents:
                    # Match indices from filtered prediction latents
                    filtered_indices = set([int(Path(p).stem.split('_')[-1]) for p in filtered_latents])
                    self.latent_cond_maps = [p for p in latent_maps_cond if int(Path(p).stem.split('_')[-1]) in filtered_indices]
                    print(f'✓ Filtered to {len(self.latent_cond_maps)} matching conditioning latents')
                else:
                    self.latent_cond_maps = None
                self.use_latents = True
                print(f'✓ Successfully reconciled {len(self.patches)} patches with matching latents')
            else:
                print('⚠ No matching patches found - falling back to raw images')
                self.use_latents = False
                self.latent_maps = None
                self.latent_cond_maps = None
                    
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
        channel_names = []
        
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
            
            # Apply transformations (filtering, normalization)
            layer_data = apply_layer_transform(layer_data, layer_config)
            
            # Convert to CHW format
            layer_data = self._to_chw(layer_data)
            
            # Convert to tensor
            layer_tensor = torch.from_numpy(layer_data).float()
            layer_tensors.append(layer_tensor)
            
            # Track channel names using proper formatting
            formatted_names = get_channel_names(layer_name, layer_config)
            channel_names.extend(formatted_names)
        
        # Stack all layers into one tensor
        unified_patch = torch.cat(layer_tensors, dim=0)  # [C_total, H, W]
        
        # Create metadata
        metadata = {
            'y': y,
            'x': x,
            'time': str(data_layers['date']),
            'region': region,
            'spatial_names': channel_names,
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
                    
        print(f"\nTotal patches across all regions: {len(all_patches)}")
        return all_patches
    
    def _create_inpainting_mask(self, H, W, street_blocks_layer=None, patch_info=None):
        """
        Create inpainting hole mask
        """
        hole_type = self.hole_config['type']
        hole_size = self.hole_config['size_px']
        
        mask_info = {
            'requested_type': hole_type,
            'actual_type': None,
            'coverage_percent': 0.0,
            'fallback_reason': None
        }
        
        if patch_info:
            mask_info.update(patch_info)
        
        if hole_type == 'street_blocks' and street_blocks_layer is not None:
            # Create binary mask from street blocks
            block_mask = (street_blocks_layer > 0).astype(np.float32)
            
            if block_mask.sum() == 0:
                # Fallback to random square if no street blocks
                hole_type = 'random_square'
                mask_info['fallback_reason'] = 'no_street_blocks'
                mask_info['actual_type'] = 'random_square'
            else:
                # Find connected pixels/street blocks
                from scipy.ndimage import label
                labeled_array, num_features = label(block_mask)
                
                # Select largest connected component
                max_area = 0
                best_mask = np.zeros_like(block_mask)
                for i in range(1, num_features + 1):
                    component = (labeled_array == i).astype(np.float32)
                    area = component.sum()
                    if area > max_area:
                        max_area = area
                        best_mask = component
                
                block_mask = best_mask
                
                # Check if block covers more than 60% of image
                coverage_percent = (block_mask.sum() / (H * W)) * 100
                max_coverage_percent = self.hole_config.get('max_coverage_percent', 25)
                mask_info['coverage_percent'] = coverage_percent
                if coverage_percent > max_coverage_percent:
                    # Fallback to random square if block is too large
                    hole_type = 'random_square'
                    mask_info['fallback_reason'] = 'block_too_large'
                    mask_info['actual_type'] = 'random_square'
                else:
                    mask_info['actual_type'] = 'street_blocks'
                    self.stats["inpainting_mask"].append(mask_info)
                    return block_mask
        if hole_type == 'random_square':
            y0 = np.random.randint(0, max(1, H - hole_size))
            x0 = np.random.randint(0, max(1, W - hole_size))
            mask = np.zeros((H, W), dtype=np.float32)
            mask[y0:y0+hole_size, x0:x0+hole_size] = 1.0
        elif hole_type == 'center_square':
            y0 = (H - hole_size) // 2
            x0 = (W - hole_size) // 2
            mask = np.zeros((H, W), dtype=np.float32)
            mask[y0:y0+hole_size, x0:x0+hole_size] = 1.0
        else:
            raise NotImplementedError(f"Hole type {hole_type} not implemented")
        
        return mask
    
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
        
    def _getitem_cached(self, index: int):
        """
        Load pre-saved patch from disk.
        
        Returns:
            - If use_latents=False: Returns patch_data (dict with 'image' and 'meta')
            - If use_latents=True: Returns latent + conditioning (KEEP AS IS for now)
        """
        y, x, region, cache_idx = self.patches[index]
        
        # Load from cache
        patch_path = self.cache_dir / f"patch_{self.split}_{cache_idx}.pt"
        patch_data = torch.load(patch_path)
        
        if self.use_latents:
            # TODO: Refactor latent handling in next step
            # For now, keep existing latent logic (as instructed)
            latent_path = self.latent_maps[index]
            latent = load_single_latent(latent_path, device=None)
            
            # Check if we have separate conditioning latents (two-VAE mode)
            if self.latent_cond_maps is not None and len(self.latent_cond_maps) > 0:
                # Load conditioning latent
                latent_cond_path = self.latent_cond_maps[index]
                latent_cond = load_single_latent(latent_cond_path, device=None)
                
                # SAFETY CHECK: Verify indices match
                pred_idx = int(Path(latent_path).stem.split('_')[-1])
                cond_idx = int(Path(latent_cond_path).stem.split('_')[-1])
                
                if pred_idx != cond_idx:
                    raise RuntimeError(
                        f"Latent index mismatch at dataset index {index}! "
                        f"Prediction latent: {pred_idx}, Conditioning latent: {cond_idx}. "
                        f"This indicates a data corruption issue. Please regenerate latents."
                    )
                
                # Prepare conditioning with latent_cond (existing logic - keep for now)
                ps = self.patch_size
                latent_h = ps // self.vae_downsample_factor
                latent_w = ps // self.vae_downsample_factor
                
                cond_inputs = {}
                spatial = []
                spatial_names = []
                
                # Extract mask from cached patch and downsample to latent resolution
                if 'meta' in patch_data:
                    cached_spatial_names = patch_data['meta'].get('spatial_names', [])
                    try:
                        mask_idx = cached_spatial_names.index('inpaint_mask')
                        mask_full = patch_data['image'][mask_idx:mask_idx+1, :, :]
                        
                        # Downsample mask to latent resolution
                        mask_latent = mask_full.unsqueeze(0)  # [1, 1, H, W]
                        mask_latent = torch.nn.functional.interpolate(
                            mask_latent,
                            size=(latent_h, latent_w),
                            mode='nearest'
                        ).squeeze(0)  # [1, H_latent, W_latent]
                        
                        spatial.append(mask_latent)
                        spatial_names.append('inpaint_mask')
                    except ValueError:
                        # No mask found, create default
                        mask_latent = torch.ones(1, latent_h, latent_w)
                        spatial.append(mask_latent)
                        spatial_names.append('inpaint_mask')
                
                # Add conditioning latent channels
                for i in range(latent_cond.shape[0]):
                    spatial.append(latent_cond[i:i+1])
                    spatial_names.append(f'latent_cond_{i}')
                
                cond_inputs['image'] = torch.cat(spatial, dim=0)
                cond_inputs['meta'] = {
                    'y': y,
                    'x': x,
                    'time': patch_data['meta'].get('time', ''),
                    'region': region,
                    'spatial_names': spatial_names,
                    'uses_latent_conditioning': True
                }
            else:
                # Fall back to pixel-space interpolation (single-VAE mode)
                cond_inputs = self._prepare_latent_conditioning(patch_data, y, x, region)
            
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Non-latent mode: Compose dataset based on mode
            unified_image = patch_data['image']
            spatial_names = patch_data['meta']['spatial_names']
            
            if self.mode_type == 'default':
                # Default mode: RGB as image, everything else as conditioning
                # Use layer registry to find RGB layer
                rgb_layer_matches = get_layer_channels_from_names(spatial_names, 'rgb')
                
                if len(rgb_layer_matches) == 0:
                    raise ValueError(
                        f"Default mode requires 'rgb' layer, but it was not found in patch. "
                        f"Available layers: {spatial_names}"
                    )
                
                # Extract RGB indices and names
                rgb_indices = [idx for idx, _ in rgb_layer_matches]
                rgb_names = [name for _, name in rgb_layer_matches]
                
                # Get conditioning indices (everything except RGB)
                cond_indices = [idx for idx in range(len(spatial_names)) if idx not in rgb_indices]
                cond_names = [name for idx, name in enumerate(spatial_names) if idx not in rgb_indices]
                
                # Extract RGB image
                image = unified_image[rgb_indices]  # [C_rgb, H, W]
                
                # Extract conditioning
                if len(cond_indices) > 0:
                    conditioning = unified_image[cond_indices]  # [C_cond, H, W]
                    cond_meta = patch_data['meta'].copy()
                    cond_meta['spatial_names'] = cond_names
                    return image, {'image': conditioning, 'meta': cond_meta}
                else:
                    # No conditioning channels
                    image_meta = patch_data['meta'].copy()
                    image_meta['spatial_names'] = rgb_names
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
                target_names = []
                
                for layer_name in target_layers:
                    layer_matches = get_layer_channels_from_names(spatial_names, layer_name)
                    if len(layer_matches) == 0:
                        raise ValueError(
                            f"VAE group '{self.mode_target}' requires layer '{layer_name}', "
                            f"but it was not found in patch. Available: {spatial_names}"
                        )
                    
                    for idx, name in layer_matches:
                        target_indices.append(idx)
                        target_names.append(name)
                
                # Extract target layers
                image = unified_image[target_indices]  # [C_target, H, W]
                
                # Create metadata
                image_meta = patch_data['meta'].copy()
                image_meta['spatial_names'] = target_names
                
                return image, image_meta
            
            else:  # self.mode_type == 'diffusion'
                # Diffusion mode: Load prediction latent + conditioning (pixel + latent space)
                stage_config = self.diffusion_stages[self.mode_target]
                pred_group = stage_config.get('prediction_group')
                conditioning_config = stage_config.get('conditioning', {})
                
                # Get latent directories from VAE group configs
                big_data_storage_path = self.data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
                results_dir = Path(big_data_storage_path) / "results" / self.config['train_params']['task_name']
                
                # Load prediction latent
                pred_latents_dir = self.vae_groups[pred_group].get('latents_dir', f'{pred_group}_latents')
                pred_latent_path = results_dir / pred_latents_dir / f"latent_{cache_idx}.pt"
                
                if not pred_latent_path.exists():
                    raise FileNotFoundError(
                        f"Prediction latent not found: {pred_latent_path}. "
                        f"Run VAE training for group '{pred_group}' first."
                    )
                
                pred_latent = torch.load(pred_latent_path)
                
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
                        mask_matches = get_layer_channels_from_names(spatial_names, 'inpainting_mask')
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
                
                # Add latent-space conditioning (load from disk)
                for cond_spec in conditioning_config.get('latent_space', []):
                    group_name = cond_spec['group']
                    
                    if group_name not in self.vae_groups:
                        raise ValueError(
                            f"Conditioning group '{group_name}' not found in VAE groups"
                        )
                    
                    # Load conditioning latent
                    cond_latents_dir = self.vae_groups[group_name].get('latents_dir', f'{group_name}_latents')
                    cond_latent_path = results_dir / cond_latents_dir / f"latent_{cache_idx}.pt"
                    
                    if not cond_latent_path.exists():
                        raise FileNotFoundError(
                            f"Conditioning latent for group '{group_name}' not found: {cond_latent_path}. "
                            f"Run VAE training for group '{group_name}' first."
                        )
                    
                    cond_latent = torch.load(cond_latent_path)
                    cond[group_name] = cond_latent
                
                return pred_latent, cond
    
    def _getitem_xarray(self, index: int):
        """Extract patch on-the-fly from Xarray (existing logic)"""
        y, x, region = self.patches[index]
        ps = self.patch_size
        
        data_layers = self.data_layers_per_region[region]
        
        ##### Return latents and conditioning #####
        # If using latents, load latent and prepare conditioning inputs
        if self.use_latents:
            # Load prediction latent from file
            latent_path = self.latent_maps[index]
            latent = load_single_latent(latent_path, device=None)  # Load to CPU
            
            # Check if we have separate conditioning latents
            if self.latent_cond_maps is not None and len(self.latent_cond_maps) > 0:
                # Load conditioning latent
                latent_cond_path = self.latent_cond_maps[index]
                latent_cond = load_single_latent(latent_cond_path, device=None)
                
                # SAFETY CHECK: Verify indices match
                pred_idx = int(Path(latent_path).stem.split('_')[-1])
                cond_idx = int(Path(latent_cond_path).stem.split('_')[-1])
                
                if pred_idx != cond_idx:
                    raise RuntimeError(
                        f"Latent index mismatch at dataset index {index}! "
                        f"Prediction latent: {pred_idx}, Conditioning latent: {cond_idx}. "
                        f"This indicates a data corruption issue. Please regenerate latents."
                    )
                
                # Calculate latent space dimensions
                latent_h = ps // self.vae_downsample_factor
                latent_w = ps // self.vae_downsample_factor
                
                # Use conditioning latent directly (no interpolation needed!)
                # Prepare minimal conditioning (just mask, latent_cond has the rest)
                cond_inputs = {}
                spatial = []
                spatial_names = []
                
                # Still need mask in pixel/latent space
                street_blocks_layer = None
                if 'street_blocks' in data_layers and self.hole_config['type'] == 'street_blocks':
                    street_blocks_layer = data_layers['street_blocks'].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                
                patch_info = {
                    'index': index,
                    'region': region,
                    'y': y,
                    'x': x,
                    'split': self.split
                }
                inpaint_mask = self._create_inpainting_mask(ps, ps, street_blocks_layer=street_blocks_layer, patch_info=patch_info)
                
                # Downsample mask to latent resolution
                mask_latent = torch.from_numpy(inpaint_mask).float()
                mask_latent = mask_latent.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                mask_latent = torch.nn.functional.interpolate(
                    mask_latent,
                    size=(latent_h, latent_w),
                    mode='nearest'
                )
                mask_latent = mask_latent.squeeze(0)  # [1,H_latent,W_latent]
                
                # Combine: mask + conditioning latent
                spatial.append(torch.from_numpy(mask_latent.numpy()).float())
                spatial_names.append('inpaint_mask')
                
                # Add conditioning latent channels with proper semantic names
                # The conditioning VAE encodes all OSM + environmental channels
                # We need to maintain the semantic names for auxiliary loss extraction
                conditioning_channel_names = []
                for layer_name in self.osm_layers:
                    layer_idx = self.osm_layers.index(layer_name)
                    layer_config = self.osm_layer_configs[layer_idx]
                    display_name = layer_config.get('key', layer_name)
                    # Apply context suffix if in semantic mode
                    display_name = self._apply_context_suffix(display_name, [layer_config])
                    conditioning_channel_names.append(f'osm:{display_name}')
                
                for layer_name in self.environmental_layers:
                    layer_idx = self.environmental_layers.index(layer_name)
                    layer_config = self.env_layer_configs[layer_idx]
                    display_name = layer_config.get('key', layer_name)
                    # Apply context suffix if in semantic mode
                    display_name = self._apply_context_suffix(display_name, [layer_config])
                    conditioning_channel_names.append(f'env:{display_name}')
                
                # Verify channel count matches
                if len(conditioning_channel_names) != latent_cond.shape[0]:
                    print(f"⚠ WARNING: Conditioning channel mismatch!")
                    print(f"  Expected {len(conditioning_channel_names)} channels: {conditioning_channel_names}")
                    print(f"  Got {latent_cond.shape[0]} latent channels")
                    # Fallback to generic names
                    conditioning_channel_names = [f'latent_cond_{i}' for i in range(latent_cond.shape[0])]
                
                for i in range(latent_cond.shape[0]):
                    spatial.append(latent_cond[i:i+1])
                    spatial_names.append(conditioning_channel_names[i])
                
                cond_inputs['image'] = torch.cat(spatial, dim=0)
                cond_inputs['meta'] = {
                    'y': y,
                    'x': x,
                    'time': str(data_layers['date']),
                    'region': region,
                    'spatial_names': spatial_names,
                    'uses_latent_conditioning': True
                }
                
                if len(self.condition_types) == 0:
                    return latent
                else:
                    return latent, cond_inputs
            
            # Fall back to pixel-space interpolation if no conditioning latents
            latent_h = ps // self.vae_downsample_factor
            
            # Prepare conditioning inputs
            cond_inputs = {}
            
            # Create inpainting mask in original resolution
            street_blocks_layer = None
            if 'street_blocks' in data_layers and self.hole_config['type'] == 'street_blocks':
                street_blocks_layer = data_layers['street_blocks'].isel(
                    y=slice(y, y+ps),
                    x=slice(x, x+ps)
                ).values
            
            patch_info = {
                'index': index,
                'region': region,
                'y': y,
                'x': x,
                'split': self.split
            }
            inpaint_mask = self._create_inpainting_mask(ps, ps, street_blocks_layer=street_blocks_layer, patch_info=patch_info)
            
            # Prepare spatial conditioning
            spatial = []
            spatial_names = []
            
            if 'inpainting' in self.condition_types:
                # Downsample mask to latent resolution
                mask_latent = torch.from_numpy(inpaint_mask).float()
                mask_latent = mask_latent.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                mask_latent = torch.nn.functional.interpolate(
                    mask_latent,
                    size=(latent_h, latent_w),
                    mode='nearest'
                )
                mask_latent = mask_latent.squeeze(0)  # [1,H_latent,W_latent]
                self._append_spatial(spatial, spatial_names, mask_latent.numpy(), 'inpaint_mask')
            
            if 'osm_features' in self.condition_types:
                osm_layers = []
                osm_layer_names = []  # Track which layers are included
                
                for idx, layer_name in enumerate(self.osm_layers):
                    if layer_name in data_layers and data_layers[layer_name] is not None:
                        layer_patch = data_layers[layer_name].isel(
                            y=slice(y, y+ps),
                            x=slice(x, x+ps)
                        ).values
                        
                        # Apply normalization and filters
                        layer_patch = self._apply_layer_transform(layer_patch, layer_name, idx, 'osm')
                        layer_patch = self._to_chw(layer_patch)
                        osm_layers.append(layer_patch)
                        
                        # Use custom key from config
                        layer_config = self.osm_layer_configs[idx]
                        display_name = layer_config.get('key', layer_name)
                        osm_layer_names.append(display_name)
                
                if osm_layers:
                    osm_features = np.concatenate(osm_layers, axis=0)
                    # Downsample to latent resolution
                    osm_features = torch.from_numpy(osm_features).float().unsqueeze(0)
                    osm_features = torch.nn.functional.interpolate(
                        osm_features,
                        size=(latent_h, latent_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    self._append_spatial(spatial, spatial_names, osm_features.numpy(), 'osm', channel_names=osm_layer_names, layer_configs=self.osm_layer_configs)
            
            if 'environmental' in self.condition_types:
                env_layers = []
                env_layer_names = []  # Track which layers are included
                
                for idx, layer_name in enumerate(self.environmental_layers):
                    if layer_name in data_layers and data_layers[layer_name] is not None:
                        layer_patch = data_layers[layer_name].isel(
                            y=slice(y, y+ps),
                            x=slice(x, x+ps)
                        ).values
                        
                        # Apply normalization and filters
                        layer_patch = self._apply_layer_transform(layer_patch, layer_name, idx, 'env')
                        layer_patch = self._to_chw(layer_patch)
                        env_layers.append(layer_patch)
                        
                        # Use custom key from config
                        layer_config = self.env_layer_configs[idx]
                        display_name = layer_config.get('key', layer_name)
                        env_layer_names.append(display_name)
                
                if env_layers:
                    env_features = np.concatenate(env_layers, axis=0)
                    # Downsample to latent resolution
                    env_features = torch.from_numpy(env_features).float().unsqueeze(0)
                    env_features = torch.nn.functional.interpolate(
                        env_features,
                        size=(latent_h, latent_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    self._append_spatial(spatial, spatial_names, env_features.numpy(), 'env', channel_names=env_layer_names, layer_configs=self.env_layer_configs)
            
            if spatial:
                cond_inputs['image'] = torch.cat(spatial, dim=0)
            
            # Add meta information
            cond_inputs['meta'] = {
                'y': y,
                'x': x,
                'time': str(data_layers['date']),
                'region': region,
                'spatial_names': spatial_names
            }
            
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        
        ##### Return raw satellite image and conditioning #####
        # Extract satellite image patch (main input)
        img_patch = img_patch = data_layers['satellite'].isel(
            y=slice(y, y+ps), 
            x=slice(x, x+ps)
        ).values.astype(np.float32)
        
        # convert to CHW and normalize
        img_patch = self._to_chw(img_patch)
        img_patch = self._normalize_layer(img_patch, 'satellite')
        
        # street blocks mask
        street_blocks_layer = None
        if 'street_blocks' in data_layers and self.hole_config['type'] == 'street_blocks':
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
        inpaint_mask = self._create_inpainting_mask(ps, ps, street_blocks_layer=street_blocks_layer, patch_info=patch_info)
        
        # Prepare conditioning inputs
        cond_inputs = {}
        
        # put spatial conditions together into one image tensor
        spatial = []
        spatial_names = []

        # inpainting context
        if 'inpainting' in self.condition_types:
            # Only include masked RGB if explicitly requested
            if 'masked_rgb' in self.condition_types:
                masked_image = img_patch * (1.0 - inpaint_mask)
                rgb_names = ['blue', 'green', 'red']
                self._append_spatial(spatial, spatial_names, masked_image, 'masked_image', channel_names=rgb_names)
            
            # Always include mask for inpainting
            self._append_spatial(spatial, spatial_names, inpaint_mask, 'inpaint_mask')
        
        if 'osm_features' in self.condition_types:
            osm_layers = []
            osm_layer_names = []  # Track display names
            for idx, layer_name in enumerate(self.osm_layers):
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    # Apply normalization and filters
                    layer_patch = self._apply_layer_transform(layer_patch, layer_name, idx, 'osm')
                    layer_patch = self._to_chw(layer_patch)
                    osm_layers.append(layer_patch)
                    
                    # Extract display name
                    layer_config = self.osm_layer_configs[idx]
                    display_name = layer_config.get('key', layer_name)
                    osm_layer_names.append(display_name)
            
            if osm_layers:
                osm_features = np.concatenate(osm_layers, axis=0)
                self._append_spatial(spatial, spatial_names, osm_features, 'osm', channel_names=osm_layer_names, layer_configs=self.osm_layer_configs)
        
        if 'environmental' in self.condition_types:
            # Environmental data (NDVI, LST)
            env_layers = []
            env_layer_names = []  # Track display names
            for idx, layer_name in enumerate(self.environmental_layers):
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    # Apply normalization and filters
                    layer_patch = self._apply_layer_transform(layer_patch, layer_name, idx, 'env')
                    layer_patch = self._to_chw(layer_patch)
                    env_layers.append(layer_patch)
                    
                    # Extract display name
                    layer_config = self.env_layer_configs[idx]
                    display_name = layer_config.get('key', layer_name)
                    env_layer_names.append(display_name)
            
            if env_layers:
                env_features = np.concatenate(env_layers, axis=0)
                self._append_spatial(spatial, spatial_names, env_features, 'env', channel_names=env_layer_names, layer_configs=self.env_layer_configs)
        
        if 'temperature_threshold' in self.condition_types:
            # Temperature optimization target (scalar or spatially varying)
            if 'landsat_surface_temp_b10_masked' in data_layers and data_layers['landsat_surface_temp_b10_masked'] is not None:
                lst_patch = data_layers['landsat_surface_temp_b10_masked'].isel(
                    y=slice(y, y+ps),
                    x=slice(x, x+ps)
                ).values
                lst_patch = self._normalize_layer(lst_patch, 'landsat_surface_temp_b10_masked')
                lst_patch = self._to_chw(lst_patch)
                # Store as target for optimization
                cond_inputs['temperature_target'] = torch.from_numpy(lst_patch).float()
        
        if spatial:
            cond_inputs['image'] = torch.cat(spatial, dim=0)   # [C_total,H,W]


        # Add meta information
        cond_inputs['meta'] = {
            'y': y, 
            'x': x, 
            'time': str(data_layers['date']), 
            'region': region, 
            'spatial_names': spatial_names
        }

        
        # Convert target image to tensor
        im_tensor = torch.from_numpy(img_patch).float()
        if len(self.condition_types) == 0:
            return im_tensor
        else:
            return im_tensor, cond_inputs


    def _prepare_latent_conditioning(
        self,
        patch_data: Dict[str, torch.Tensor],
        y: int,
        x: int,
        region: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare conditioning inputs for latent-based training.
        
        Downsamples spatial conditioning to latent resolution, using appropriate
        interpolation for each channel type (nearest for mask, bilinear for features).
        
        Args:
            patch_data: Cached patch data with 'image' and 'conditioning'
            y, x: Patch coordinates
            region: Region name
        
        Returns:
            Conditioning dict with downsampled spatial features
        """
        ps = self.patch_size
        latent_h = ps // self.vae_downsample_factor
        latent_w = ps // self.vae_downsample_factor
        
        cond_inputs = {}
        
        if patch_data['conditioning'] is None or 'image' not in patch_data['conditioning']:
            return cond_inputs
        
        full_cond = patch_data['conditioning']['image']  # [C, H, W]
        spatial_names = patch_data['conditioning']['meta'].get('spatial_names', [])
        
        # Separate mask from other features for appropriate downsampling
        downsampled_channels = []
        
        for idx, channel_name in enumerate(spatial_names):
            if not 'masked_rgb' in self.condition_types and 'masked_image' in channel_name.lower():
                continue
            
            channel = full_cond[idx:idx+1, :, :]  # [1, H, W]
            
            # Use nearest interpolation for mask to preserve binary values,
            # bilinear for continuous features
            if 'mask' in channel_name.lower():
                mode = 'nearest'
            else:
                mode = 'bilinear'
            
            channel_resized = torch.nn.functional.interpolate(
                channel.unsqueeze(0),  # [1, 1, H, W]
                size=(latent_h, latent_w),
                mode=mode,
                align_corners=False if mode == 'bilinear' else None
            ).squeeze(0)  # [1, H_latent, W_latent]
            
            downsampled_channels.append(channel_resized)
        
        # Concatenate all channels
        cond_inputs['image'] = torch.cat(downsampled_channels, dim=0)  # [C, H_latent, W_latent]
        cond_inputs['meta'] = patch_data['conditioning']['meta'].copy()
        
        return cond_inputs
    
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
        print(f"Using latents: {self.use_latents}")
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
        
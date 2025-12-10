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
                 cache_dir: Optional[str] = None
        ):
        """
        :param split: 'train' or 'val'
        :param use_latents: whether to use pre-computed latents from autoencoder
        :param latent_path: path to latent files
        :param use_cached_patches: whether to use cached patches
        :param cache_dir: directory for cached patches
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
        
        # Basic parameters
        big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
        im_res = dataset_config.get('res', 3)  # in meters
        im_channels = dataset_config.get('im_channels', 3)
        min_valid_percent = dataset_config.get('min_valid_percent', 90)
        pixel_size = dataset_config.get('patch_size_m', 650)  # in pixels
        patch_size = int(pixel_size/im_res)  # compute patch size in pixels
        patch_size = patch_size - (patch_size % 8) # make patch size divisible by 8

        # Latent space configuration
        self.latent_maps = None
        self.latent_path = latent_path
        self.use_latents = bool(use_latents)
        # If using latents, need to account for both VAE and U-Net downsampling
        if use_latents:
            # Calculate downsampling factor for latent space
            autoencoder_config = config.get('autoencoder_params', {})
            down_sample = autoencoder_config.get('down_sample', [True, True, True])
            self.latent_downsample_factor = 2 ** sum([1 for ds in down_sample if ds])
        else:
            self.latent_downsample_factor = 1
        
        # Get U-Net downsampling factor
        ldm_config = config.get('ldm_params', {})
        num_down_layers = len(ldm_config.get('down_channels', [64, 128, 256, 512]))
        unet_downsample_factor = 2 ** num_down_layers
        
        # Total required divisibility
        total_divisor = self.latent_downsample_factor * unet_downsample_factor
        
        # Make patch size divisible by total factor
        patch_size = patch_size - (patch_size % total_divisor)
        
        print(f"Using patch size: {patch_size} pixels ({patch_size*im_res} m at {im_res} m resolution)")
        print(f"  VAE downsample factor: {self.latent_downsample_factor}")
        print(f"  U-Net downsample factor: {unet_downsample_factor}")
        print(f"  Total divisor: {total_divisor}")

        # Store parameters
        self.split = split
        self.patch_size = patch_size
        self.stride_overlap = dataset_config.get('stride_overlap', 2)
        self.stride = int(patch_size // self.stride_overlap)  # compute stride based on overlap
        self.im_channels = im_channels
        self.min_valid_percent = min_valid_percent

        # Conditioning configuration
        ldm_config = config.get('ldm_params', None)
        if ldm_config is None:
            raise ValueError("LDM configuration not found in config file")
        
        condition_config = get_nested(config, ['ldm_params', 'condition_config'])
        if condition_config is None:
            raise ValueError("Conditioning configuration not found in config file")

        self.condition_types = condition_config.get('condition_types', [])
        self.hole_config = condition_config.get('hole_config', {
            'type': 'random_square',
            'size_px': 64
        })
        self.osm_layers = get_nested(condition_config, ['osm_layers'], ['buildings', 'streets', 'water'])
        self.environmental_layers = get_nested(condition_config, ['environmental_layers'], ['ndvi', 'landsat_surface_temp_b10_masked'])
        
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
            cache_dir = Path(big_data_storage_path) / "processed" / task_name
            
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
        """Load VAE latents and reconcile with patches"""
        print(f'Loading latents from {self.latent_path}...')
        latent_maps = load_latents(self.latent_path)
        
        if len(latent_maps) == len(self.patches):
            # Perfect match
            self.use_latents = True
            self.latent_maps = latent_maps
            print(f'✓ Found {len(self.latent_maps)} latents matching {len(self.patches)} patches')
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
                self.use_latents = True
                print(f'✓ Successfully reconciled {len(self.patches)} patches with matching latents')
            else:
                print('⚠ No matching patches found - falling back to raw images')
                self.use_latents = False
                self.latent_maps = None
                    
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
        Extract and process a single patch from Xarray (existing __getitem__ logic).
        
        Returns:
            Dictionary with 'image' and optional 'conditioning' data
        """
        ps = self.patch_size
        data_layers = self.data_layers_per_region[region]
        
        # Extract satellite image
        img_patch = data_layers['satellite'].isel(
            y=slice(y, y+ps),
            x=slice(x, x+ps)
        ).values.astype(np.float32)
        
        img_patch = self._to_chw(img_patch)
        img_patch = self._normalize_layer(img_patch, 'satellite')
        
        # Prepare conditioning
        patch_info = {
            'index': index,
            'region': region,
            'y': y,
            'x': x,
            'split': self.split
        }
        
        # Street blocks for inpainting mask
        street_blocks_layer = None
        if 'street_blocks' in data_layers and self.hole_config['type'] == 'street_blocks':
            street_blocks_layer = data_layers['street_blocks'].isel(
                y=slice(y, y+ps),
                x=slice(x, x+ps)
            ).values
        
        inpaint_mask = self._create_inpainting_mask(ps, ps, street_blocks_layer, patch_info)
        
        # Build conditioning dict
        cond_inputs = {}
        spatial = []
        spatial_names = []
        
        if 'inpainting' in self.condition_types:
            masked_image = img_patch * (1.0 - inpaint_mask)
            rgb_names = ['blue', 'green', 'red']
            self._append_spatial(spatial, spatial_names, masked_image, 'masked_image', channel_names=rgb_names)
            self._append_spatial(spatial, spatial_names, inpaint_mask, 'inpaint_mask')
            # Also store mask separately for sampling compatibility
            cond_inputs['mask'] = torch.from_numpy(self._to_chw(inpaint_mask)).float()
        
        if 'osm_features' in self.condition_types:
            osm_layers = []
            for layer_name in self.osm_layers:
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    layer_patch = self._normalize_layer(layer_patch, layer_name)
                    layer_patch = self._to_chw(layer_patch)
                    osm_layers.append(layer_patch)
            
            if osm_layers:
                osm_features = np.concatenate(osm_layers, axis=0)
                self._append_spatial(spatial, spatial_names, osm_features, 'osm', channel_names=self.osm_layers)
        
        if 'environmental' in self.condition_types:
            env_layers = []
            for layer_name in self.environmental_layers:
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    layer_patch = self._normalize_layer(layer_patch, layer_name)
                    layer_patch = self._to_chw(layer_patch)
                    env_layers.append(layer_patch)
            
            if env_layers:
                env_features = np.concatenate(env_layers, axis=0)
                self._append_spatial(spatial, spatial_names, env_features, 'env', channel_names=self.environmental_layers)
        
        if spatial:
            cond_inputs['image'] = torch.cat(spatial, dim=0)
        
        cond_inputs['meta'] = {
            'y': y,
            'x': x,
            'time': str(data_layers['date']),
            'region': region,
            'spatial_names': spatial_names,
            'mask_channel_idx': spatial_names.index('inpaint_mask') if 'inpaint_mask' in spatial_names else None
        }
        
        # Return as dict for easy serialization
        return {
            'image': torch.from_numpy(img_patch).float(),
            'conditioning': cond_inputs if len(self.condition_types) > 0 else None
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
        
        all_patches = []
        
        for region in self.regions:
            print(f"\nProcessing region: {region}")
            merged_xs = self.datasets[region]
        
            # Get valid dates with planetscope data
            valid_planet_dates = (
                merged_xs['planetscope_sr_4band']
                .notnull()
                .sum(dim=['x', 'y']) > 0
            ).any(dim='channel').compute()
        
            valid_dates = merged_xs['time'].where(valid_planet_dates, drop=True).values
        
            if len(valid_dates) == 0:
                print(f"No valid dates found for region {region}")
                continue
        
            # For now, first valid date (can be extended to use multiple dates)
            selected_date = valid_dates[0]
            print(f"Using date: {selected_date} for region {region}")
            
            # Select data for this date 
            date_data = merged_xs.sel(time=selected_date)
            
            # Get satellite image and compute validity mask
            img_da = date_data['planetscope_sr_4band'].sel(channel=['blue', 'green', 'red'])
            valid_mask = (~img_da.isnull()).all(dim='channel').compute()
            
            # Handle reflectance scaling
            if img_da.max() > 20:
                img_da = (img_da / 10000.0).astype(np.float32)
            
            # Store data layers for this region
            data_layers = {
                'satellite': img_da,
                'valid_mask': valid_mask,
                'date': selected_date
            }
        
            # Add optional layers
            for layer in self.osm_layers:
                if layer in date_data:
                    data_layers[layer] = date_data[layer]
        
            # Add environmental data if available
            for layer in self.environmental_layers:
                if layer in date_data:
                    # layer_data = date_data[layer].sel(time=self.selected_date, method='nearest')
                    # data_layers[layer] = layer_data.values
                    data_layers[layer] = date_data[layer]
                    
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
                    
        print(f"\nTotal patches across all {self.split} regions: {len(all_patches)}")
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
    
    def _normalize_layer(self, data, layer_name):
        """
        Normalize data layer to [-1, 1] range
        """
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Different normalization strategies per layer type
        if layer_name in ['satellite']:
            # Clip to reasonable range and normalize
            data = np.clip(data, 0, 1)  # ensure in [0, 1]
            data = data* 2.0 - 1.0
        elif layer_name in ['ndvi']:
            # (NDVI is already in [-1, 1] range typically)
            data = np.clip(data, -1, 1)
        elif layer_name in ['landsat_surface_temp_b10_masked']:
            # Temperature - normalize to reasonable range
            data = np.clip(data, 250, 350)  # Kelvin
            data = ((data - 250) / 100.0) * 2.0 - 1.0
        else:
            # Binary or categorical layers
            if data.max() <= 1.0:
                data = data * 2.0 - 1.0 # normalize to [-1, 1]
            else:
                # Normalize to [0, 1] then to [-1, 1]
                data = (data - data.min()) / (data.max() - data.min() + 1e-6)
                data = data * 2.0 - 1.0
        
        return data
    
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

    def _append_spatial(self, stack, names, arr, base_name, channel_names=None):
        """Adds array to the stack and records channel names.
        
        Args:
            stack: list to append tensors to
            names: list to append channel names to
            arr: numpy array to add
            base_name: base name for the layer
            channel_names: optional list of specific channel names (e.g., ['blue', 'green', 'red'])
        """
        arr = self._to_chw(arr)
        stack.append(torch.from_numpy(arr).float())
        
        if arr.shape[0] == 1:
            names.append(base_name)
        else:
            if channel_names is not None and len(channel_names) == arr.shape[0]:
                # Use provided channel names
                names.extend([f"{base_name}:{ch}" for ch in channel_names])
            else:
                # Fall back to numeric indices
                names.extend([f"{base_name}:{i}" for i in range(arr.shape[0])])


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
        """Load pre-saved patch from disk"""
        y, x, region, cache_idx = self.patches[index]
        
        # Load from cache
        patch_path = self.cache_dir / f"patch_{self.split}_{cache_idx}.pt"
        patch_data = torch.load(patch_path)
        
        if self.use_latents:
            # Load corresponding latent
            latent_path = self.latent_maps[index]
            latent = load_single_latent(latent_path, device=None)
            
            # Prepare conditioning for latent space
            cond_inputs = self._prepare_latent_conditioning(patch_data, y, x, region)
            
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Return raw image + conditioning
            if len(self.condition_types) == 0:
                return patch_data['image']
            else:
                return patch_data['image'], patch_data['conditioning']
    
    def _getitem_xarray(self, index: int):
        """Extract patch on-the-fly from Xarray (existing logic)"""
        y, x, region = self.patches[index]
        ps = self.patch_size
        
        data_layers = self.data_layers_per_region[region]
        
        ##### Return latents and conditioning #####
        # If using latents, load latent and prepare conditioning inputs
        if self.use_latents:
            # Load latent from file
            latent_path = self.latent_maps[index]
            latent = load_single_latent(latent_path, device=None)  # Load to CPU
            
            # Still need to prepare conditioning for latent-based training
            # Calculate latent space dimensions
            latent_h = ps // self.latent_downsample_factor
            latent_w = ps // self.latent_downsample_factor
            
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
                for layer_name in self.osm_layers:
                    if layer_name in data_layers and data_layers[layer_name] is not None:
                        layer_patch = data_layers[layer_name].isel(
                            y=slice(y, y+ps),
                            x=slice(x, x+ps)
                        ).values
                        layer_patch = self._normalize_layer(layer_patch, layer_name)
                        layer_patch = self._to_chw(layer_patch)
                        osm_layers.append(layer_patch)
                
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
                    self._append_spatial(spatial, spatial_names, osm_features.numpy(), 'osm', channel_names=self.osm_layers)
            
            if 'environmental' in self.condition_types:
                env_layers = []
                for layer_name in self.environmental_layers:
                    if layer_name in data_layers and data_layers[layer_name] is not None:
                        layer_patch = data_layers[layer_name].isel(
                            y=slice(y, y+ps),
                            x=slice(x, x+ps)
                        ).values
                        layer_patch = self._normalize_layer(layer_patch, layer_name)
                        layer_patch = self._to_chw(layer_patch)
                        env_layers.append(layer_patch)
                
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
                    self._append_spatial(spatial, spatial_names, env_features.numpy(), 'env', channel_names=self.environmental_layers)
            
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
        
        if 'inpainting' in self.condition_types:
            # Masked image for inpainting context
            masked_image = img_patch * (1.0 - inpaint_mask)
            cond_inputs['masked_image'] = torch.from_numpy(masked_image).float()
            cond_inputs['mask'] = torch.from_numpy(self._to_chw(inpaint_mask)).float()  # [1,H,W]
        
        if 'osm_features' in self.condition_types:
            # Stack OSM layers as control signal
            osm_layers = []
            for layer_name in self.osm_layers:
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    # layer_patch = self.data_layers[layer_name][y:y+ps, x:x+ps]
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    layer_patch = self._normalize_layer(layer_patch, layer_name)
                    layer_patch = self._to_chw(layer_patch)
                    osm_layers.append(layer_patch)
            
            if osm_layers:
                osm_features = np.concatenate(osm_layers, axis=0)
                cond_inputs['osm_features'] = torch.from_numpy(osm_features).float()
        
        if 'environmental' in self.condition_types:
            # Environmental data (NDVI, LST)
            env_layers = []
            for layer_name in self.environmental_layers:
                if layer_name in data_layers and data_layers[layer_name] is not None:
                    # layer_patch = self.data_layers[layer_name][y:y+ps, x:x+ps]
                    layer_patch = data_layers[layer_name].isel(
                        y=slice(y, y+ps),
                        x=slice(x, x+ps)
                    ).values
                    layer_patch = self._normalize_layer(layer_patch, layer_name)
                    layer_patch = self._to_chw(layer_patch)
                    env_layers.append(layer_patch)
            
            if env_layers:
                # env_features = np.stack(env_layers, axis=0).astype(np.float32)
                env_features = np.concatenate(env_layers, axis=0)
                cond_inputs['environmental'] = torch.from_numpy(env_features).float()
        
        if 'temperature_threshold' in self.condition_types:
            # Temperature optimization target (scalar or spatially varying)
            if 'landsat_surface_temp_b10_masked' in data_layers and data_layers['landsat_surface_temp_b10_masked'] is not None:
                # lst_patch = self.data_layers['landsat_surface_temp_b10_masked'][y:y+ps, x:x+ps]
                lst_patch = data_layers['landsat_surface_temp_b10_masked'].isel(
                    y=slice(y, y+ps),
                    x=slice(x, x+ps)
                ).values
                lst_patch = self._normalize_layer(lst_patch, 'landsat_surface_temp_b10_masked')
                lst_patch = self._to_chw(lst_patch)
                # Store as target for optimization
                cond_inputs['temperature_target'] = torch.from_numpy(lst_patch).float()
        
        
        # put spatial conditions together into one image tensor
        spatial = []
        spatial_names = []

        # inpainting context
        if 'inpainting' in self.condition_types:
            # RGB channels for masked image
            rgb_names = ['blue', 'green', 'red']
            self._append_spatial(spatial, spatial_names, masked_image, 'masked_image', channel_names=rgb_names)
            self._append_spatial(spatial, spatial_names, inpaint_mask, 'inpaint_mask')

        # OSM
        if 'osm_features' in self.condition_types and 'osm_features' in cond_inputs:
            osm_data = cond_inputs.pop('osm_features').numpy()
            # Use actual OSM layer names
            self._append_spatial(spatial, spatial_names, osm_data, 'osm', channel_names=self.osm_layers)

        # environmental
        if 'environmental' in self.condition_types and 'environmental' in cond_inputs:
            env_data = cond_inputs.pop('environmental').numpy()
            # Use actual environmental layer names
            self._append_spatial(spatial, spatial_names, env_data, 'env', channel_names=self.environmental_layers)

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
        
        # # Return based on whether using latents
        # if self.use_latents:
        #     # Placeholder - implement latent loading logic
        #     latent = self.latent_maps[index]
        #     if len(self.condition_types) == 0:
        #         return latent
        #     else:
        #         return latent, cond_inputs
        # else:
        #     if len(self.condition_types) == 0:
        #         return im_tensor
        #     else:
        #         return im_tensor, cond_inputs
    
    def _prepare_latent_conditioning(
        self,
        patch_data: Dict[str, torch.Tensor],
        y: int,
        x: int,
        region: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare conditioning inputs for latent-based training"""
        ps = self.patch_size
        latent_h = ps // self.latent_downsample_factor
        latent_w = ps // self.latent_downsample_factor
        
        cond_inputs = {}
        spatial = []
        spatial_names = []
        
        # Extract conditioning from cached patch data
        if patch_data['conditioning'] is not None and 'image' in patch_data['conditioning']:
            full_cond = patch_data['conditioning']['image']
            
            # Downsample to latent resolution
            full_cond_resized = torch.nn.functional.interpolate(
                full_cond.unsqueeze(0),
                size=(latent_h, latent_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            cond_inputs['image'] = full_cond_resized
            cond_inputs['meta'] = patch_data['conditioning']['meta']
        
        return cond_inputs
    
    def _print_summary(self):
        """Print dataset configuration summary"""
        print(f"\n{'='*60}")
        print(f"Dataset Configuration Summary")
        print(f"{'='*60}")
        print(f"Split: {self.split}")
        print(f"Mode: {'Cached patches' if self.use_cached_patches else 'On-the-fly Xarray'}")
        print(f"Total patches: {len(self.patches)}")
        print(f"Using latents: {self.use_latents}")
        print(f"Patch size: {self.patch_size}x{self.patch_size}")
        print(f"Conditioning types: {self.condition_types}")
        if self.use_cached_patches:
            print(f"Cache directory: {self.cache_dir}")
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
        
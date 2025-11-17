# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main
# import relevant libraries
import os
import yaml
import numpy as np
import torch
import xarray as xr
from torch.utils.data.dataset import Dataset
from utils.diffusion_utils import load_latents
from utils.read_yaml import get_nested

# Dataset class
class UrbanInpaintingDataset(Dataset):
    """
    Dataset for urban layout inpainting with multiple conditioning types:
    - Spatial context (surrounding areas via inpainting mask)
    - OSM features (buildings, streets, water, etc.)
    - Environmental data (NDVI, LST)
    - Satellite imagery
    """
    
    def __init__(self, split, use_latents=False, 
                 latent_path=None):
        """
        :param split: 'train' or 'test'
        :param zarr_path: path to the zarr dataset
        :param patch_size: size of patches to extract
        :param stride: stride for patch extraction
        :param min_valid_percent: minimum percentage of valid pixels in a patch
        :param im_channels: number of image channels
        :param use_latents: whether to use pre-computed latents from autoencoder
        :param latent_path: path to latent files
        :param condition_config: dict with conditioning configuration
        """
        
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
        
        # if not config, raise error
        dataset_config = config.get('dataset_params', None)
        if dataset_config is None:
            raise ValueError("Dataset configuration not found in config file")
        
        im_res = dataset_config.get('res', 3)  # in meters
        pixel_size = dataset_config.get('patch_size_m', 650)  # in pixels
        patch_size = int(pixel_size/im_res)  # compute patch size in pixels
        patch_size = patch_size - (patch_size % 8) # make patch size divisible by 8
        print(f"Using patch size: {patch_size} pixels ({patch_size*im_res} m at {im_res} m resolution)")

        im_channels = dataset_config.get('im_channels', 3)
        min_valid_percent = dataset_config.get('min_valid_percent', 90)
        
        self.split = split
        self.patch_size = patch_size
        self.stride_overlap = dataset_config.get('stride_overlap', 2)
        self.stride = int(patch_size // self.stride_overlap)  # compute stride based on overlap
        self.im_channels = im_channels
        self.min_valid_percent = min_valid_percent

        
        ldm_config = config.get('ldm_params', None)
        if ldm_config is None:
            raise ValueError("LDM configuration not found in config file")
        
        condition_config = get_nested(config, ['ldm_params', 'condition_config'])
        if condition_config is None:
            raise ValueError("Conditioning configuration not found in config file")
        
        # Conditioning configuration
        self.condition_types = condition_config.get('condition_types', [])
        self.hole_config = condition_config.get('hole_config', {
            'type': 'random_square',
            'size_px': 64
        })
        self.osm_layers = get_nested(condition_config, ['osm_layers'], ['buildings', 'streets', 'water'])
        self.environmental_layers = get_nested(condition_config, ['environmental_layers'], ['ndvi', 'landsat_surface_temp_b10_masked'])


        # Latent configuration
        self.latent_maps = None
        self.latent_path = latent_path
        self.use_latents = bool(use_latents)
        
        # Load xarray dataset
        train_regions = dataset_config.get('train_regions', ['Dresden', 'Hamburg', 'Stuttgart'])
        eval_regions = dataset_config.get('eval_regions', ['Leipzig'])
        
        region = train_regions[0]  # for now, use first region --> to do: extend to multiple regions
        ## to do: implement train regions and eval region(s) split to select based on self.split
        processed_data_path = f"{big_data_storage_path}/processed/{region.lower()}"
        zarr_name = dataset_config.get('zarr_name', 'input_data.zarr')
        zarr_path = os.path.join(processed_data_path, zarr_name)
        
        print(f"Loading zarr dataset from {zarr_path}...")
        self.merged_xs = xr.open_zarr(zarr_path, consolidated=True)
        
        # Load patches
        self.patches = self._load_patches(region=region)
        
        # Load latents if specified
        if self.use_latents and self.latent_path is not None:
            latent_maps = load_latents(self.latent_path)
            if len(latent_maps) == len(self.patches):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'Found {len(self.latent_maps)} latents')
            else:
                print('Latents not found or size mismatch')
    
    def _load_patches(self, region='Leipzig'):
        """
        Pre-compute valid patch locations from the dataset
        """
        # Get valid dates with planetscope data
        valid_planet_dates = (
            self.merged_xs['planetscope_sr_4band']
            .notnull()
            .sum(dim=['x', 'y']) > 0
        ).any(dim='channel').compute()
        
        valid_dates = self.merged_xs['time'].where(valid_planet_dates, drop=True).values
        
        if len(valid_dates) == 0:
            raise ValueError("No valid dates found in dataset")
        
        # For now, use first valid date (can be extended to use multiple dates)
        self.selected_date = valid_dates[0]
        print(f"Using date: {self.selected_date}")
        
        # Select data for this date
        date_data = self.merged_xs.sel(time=self.selected_date)
        
        # Get satellite image and compute validity mask
        self.img_da = date_data['planetscope_sr_4band'].sel(channel=['blue', 'green', 'red'])
        valid_mask = (~self.img_da.isnull()).all(dim='channel').compute()
        
        # close dataset for io efficiency
        # self.merged_xs.close()
        
        
        if self.img_da.max() > 20:  # reflectance scaled
            self.data_layers['satellite'] = (self.data_layers['satellite'] / 10000.0).astype(np.float32)

        
        # Store all layers we need
        self.data_layers = {
            'satellite': self.img_da,  # (3, H, W)
            'buildings': date_data['buildings'].values if 'buildings' in date_data else None,
        }
        
        # Add optional layers based on what's available
        for layer in self.osm_layers:
            if layer in date_data:
                self.data_layers[layer] = date_data[layer]
        
        # Add environmental data if available
        for layer in self.environmental_layers:
            if layer in date_data:
                # layer_data = date_data[layer].sel(time=self.selected_date, method='nearest')
                # self.data_layers[layer] = layer_data.values
                self.data_layers[layer] = date_data[layer]
        
        # Compute valid patches based on min valid percent of data
        H, W = valid_mask.shape
        min_valid_pixels = int((self.patch_size ** 2) * (self.min_valid_percent / 100))
        
        patches = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                valid_count = valid_mask[y:y+self.patch_size, x:x+self.patch_size].sum()
                if valid_count >= min_valid_pixels:
                    patches.append((y, x, region))
                    
        print(f"Found {len(patches)} valid patches for region {region} with split {self.split} and min valid percent {self.min_valid_percent}%")
        return patches
    
    def _create_inpainting_mask(self, H, W):
        """
        Create inpainting hole mask
        """
        hole_type = self.hole_config['type']
        hole_size = self.hole_config['size_px']
        
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

    
    def __getitem__(self, index):
        y, x, region = self.patches[index]
        ps = self.patch_size
        
        # Extract satellite image patch (main input)
        img_patch = img_patch = self.data_layers['satellite'].isel(
            y=slice(y, y+ps), 
            x=slice(x, x+ps)
        ).values.astype(np.float32)
        
        # convert to CHW and normalize
        img_patch = self._to_chw(img_patch)
        img_patch = self._normalize_layer(img_patch, 'satellite')
        
        # Create inpainting mask
        inpaint_mask = self._create_inpainting_mask(ps, ps)
        
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
                if layer_name in self.data_layers and self.data_layers[layer_name] is not None:
                    # layer_patch = self.data_layers[layer_name][y:y+ps, x:x+ps]
                    layer_patch = self.data_layers[layer_name].isel(
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
                if layer_name in self.data_layers and self.data_layers[layer_name] is not None:
                    # layer_patch = self.data_layers[layer_name][y:y+ps, x:x+ps]
                    layer_patch = self.data_layers[layer_name].isel(
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
            if 'landsat_surface_temp_b10_masked' in self.data_layers and self.data_layers['landsat_surface_temp_b10_masked'] is not None:
                # lst_patch = self.data_layers['landsat_surface_temp_b10_masked'][y:y+ps, x:x+ps]
                lst_patch = self.data_layers['landsat_surface_temp_b10_masked'].isel(
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
        cond_inputs['meta'] = {'y': y, 'x': x, 'time': str(self.selected_date), 'region': region, 'spatial_names': spatial_names}

        
        # Convert target image to tensor
        im_tensor = torch.from_numpy(img_patch).float()
        
        # Return based on whether using latents
        if self.use_latents:
            # Placeholder - implement latent loading logic
            latent = self.latent_maps[index]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
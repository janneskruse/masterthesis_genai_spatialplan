# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils
def validate_class_config(condition_config):
    assert 'class_condition_config' in condition_config, \
        "Class conditioning desired but class condition config missing"
    assert 'num_classes' in condition_config['class_condition_config'], \
        "num_class missing in class condition config"


def validate_text_config(condition_config):
    assert 'text_condition_config' in condition_config, \
        "Text conditioning desired but text condition config missing"
    assert 'text_embed_dim' in condition_config['text_condition_config'], \
        "text_embed_dim missing in text condition config"
    

def validate_image_config(condition_config):
    assert 'image_condition_config' in condition_config, \
        "Image conditioning desired but image condition config missing"
    assert 'image_condition_input_channels' in condition_config['image_condition_config'], \
        "image_condition_input_channels missing in image condition config"
    assert 'image_condition_output_channels' in condition_config['image_condition_config'], \
        "image_condition_output_channels missing in image condition config"
    

def validate_image_conditional_input(cond_input, x):
    assert 'image' in cond_input, \
        "Model initialized with image conditioning but cond_input has no image information"
    assert cond_input['image'].shape[0] == x.shape[0], \
        "Batch size mismatch of image condition and input"
    assert cond_input['image'].shape[2] % x.shape[2] == 0, \
        "Height/Width of image condition must be divisible by latent input"


def validate_class_conditional_input(cond_input, x, num_classes):
    assert 'class' in cond_input, \
        "Model initialized with class conditioning but cond_input has no class information"
    assert cond_input['class'].shape == (x.shape[0], num_classes), \
        "Shape of class condition input must match (Batch Size, )"
def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value


def get_prediction_channels(condition_config):
    """
    Extract prediction channel names from condition config.
    
    Handles two formats:
    1. Simple string: ['buildings', 'streets'] -> ['osm:buildings', 'osm:streets']
    2. Dict with predict flag: {vegetation: {predict: True, layer: 'ndvi', gte: 0.2}}
       -> ['env:vegetation'] (only if predict=True)
    
    Args:
        condition_config: Configuration dict containing osm_layers and environmental_layers
        
    Returns:
        List of prediction channel names with prefixes (osm: or env:)
    """
    prediction_channels = []
    
    # Process OSM layers (simple format)
    osm_layers = condition_config.get('osm_layers', [])
    for layer in osm_layers:
        if isinstance(layer, str):
            # Simple string format - add with osm: prefix
            prediction_channels.append(f'osm:{layer}')
        elif isinstance(layer, dict):
            # Dict format with predict flag
            for layer_name, layer_config in layer.items():
                if isinstance(layer_config, dict) and layer_config.get('predict', True):
                    prediction_channels.append(f'osm:{layer_name}')
    
    # Process environmental layers (can be dict format with filters)
    env_layers = condition_config.get('environmental_layers', [])
    for layer in env_layers:
        if isinstance(layer, str):
            # Simple string format - add with env: prefix
            prediction_channels.append(f'env:{layer}')
        elif isinstance(layer, dict):
            # Dict format with predict flag and optional filters
            for layer_name, layer_config in layer.items():
                if isinstance(layer_config, dict) and layer_config.get('predict', True):
                    prediction_channels.append(f'env:{layer_name}')
    
    return prediction_channels


def get_all_channels(condition_config, include_mask=False):
    """
    Extract ALL channel names from condition config (prediction + conditioning).
    
    Used when condition_latents=True to encode all channels through VAE.
    
    Args:
        condition_config: Configuration dict containing osm_layers and environmental_layers
        include_mask: Whether to include 'inpaint_mask' channel
        
    Returns:
        List of all channel names with prefixes (osm: or env:)
    """
    all_channels = []
    
    # Add inpainting mask if requested
    if include_mask and 'inpainting' in condition_config.get('condition_types', []):
        all_channels.append('inpaint_mask')
    
    # Process OSM layers (all layers, regardless of predict flag)
    osm_layers = condition_config.get('osm_layers', [])
    for layer in osm_layers:
        if isinstance(layer, str):
            # Simple string format
            all_channels.append(f'osm:{layer}')
        elif isinstance(layer, dict):
            # Dict format - include all layers
            for layer_name, layer_config in layer.items():
                if isinstance(layer_config, dict):
                    # Use custom key if specified, otherwise use layer name
                    display_name = layer_config.get('key', layer_name)
                    all_channels.append(f'osm:{display_name}')
                else:
                    all_channels.append(f'osm:{layer_name}')
    
    # Process environmental layers (all layers, regardless of predict flag)
    env_layers = condition_config.get('environmental_layers', [])
    for layer in env_layers:
        if isinstance(layer, str):
            # Simple string format
            all_channels.append(f'env:{layer}')
        elif isinstance(layer, dict):
            # Dict format - include all layers
            for layer_name, layer_config in layer.items():
                if isinstance(layer_config, dict):
                    # Use custom key if specified, otherwise use layer name
                    display_name = layer_config.get('key', layer_name)
                    all_channels.append(f'env:{display_name}')
                else:
                    all_channels.append(f'env:{layer_name}')
    
    return all_channels

def compute_patch_and_latent_sizes(
    dataset_config: dict,
    autoencoder_config: dict,
    ldm_config: dict,
    use_latents: bool = False,
    self=None
) -> tuple[int, int, int, int, int]:
    """
    Compute properly aligned patch and latent sizes.
    
    Ensures patch size is divisible by both VAE and U-Net downsampling factors
    to prevent dimension mismatches in skip connections.
    
    Args:
        dataset_config: Dataset configuration with 'patch_size_m' and 'res'
        autoencoder_config: VAE config with 'down_sample'
        ldm_config: U-Net config with 'down_channels',
        self: optional UrbanInpaintingDataset instance for setting latent_downsample_factor
    
    Returns:
        tuple: (patch_size, latent_size, vae_factor, unet_factor, total_divisor)
    """
    # Initial calculation
    pixel_size = dataset_config.get('patch_size_m', 650)
    im_res = dataset_config.get('res', 3)
    patch_size = int(pixel_size / im_res) # compute patch size in pixels
    patch_size = patch_size - (patch_size % 8) # make patch size divisible by 8
    
    # VAE downsampling factor
    if use_latents:
        vae_downsample_factor = 2 ** sum([1 for ds in autoencoder_config.get('down_sample', [True, True, True]) if ds])
    else:
        vae_downsample_factor = 1
        
    patch_size = patch_size - (patch_size % vae_downsample_factor)
    
    # U-Net downsampling factor
    num_down_layers = len(ldm_config.get('down_channels', [64, 128, 256, 512]))
    unet_downsample_factor = 2 ** num_down_layers
    
    # Total divisibility requirement
    total_divisor = vae_downsample_factor * unet_downsample_factor
    patch_size = patch_size - (patch_size % total_divisor)
    
    # Latent size
    latent_size = patch_size // vae_downsample_factor
    
    if self is not None:
        self.vae_downsample_factor = vae_downsample_factor
    
    print(f"Using patch size: {patch_size} pixels ({patch_size*im_res} m at {im_res} m resolution)")
    print(f"  VAE downsample factor: {vae_downsample_factor}")
    print(f"  U-Net downsample factor: {unet_downsample_factor}")
    print(f"  Total divisor: {total_divisor}")
    
    return (
        patch_size,
        latent_size,
        vae_downsample_factor,
        unet_downsample_factor,
        total_divisor
    )
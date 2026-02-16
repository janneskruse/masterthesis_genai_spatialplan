"""
==============================================================================
Utilities for building model configs from global config and stage-specific settings.
==============================================================================
"""

# partially adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/utils

# Local imports
from model.utils.scalar_controls import parse_scalar_controls_config

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


def compute_cvae_cond_channels(cvae_config, vae_groups_config):
    """
    Auto-compute CVAE decoder conditioning channels from config.
    
    cond_channels = 1 (mask) + sum(z_channels for each latent conditioning group)
    cond_projected_channels = int(cond_channels * cond_channel_scale)
    
    Args:
        cvae_config: CVAE stage config dict (e.g. config['cvae_inpainting']['semantic'])
        vae_groups_config: VAE groups configuration dict (config['vae_groups'])
        
    Returns:
        Tuple of (cond_channels, cond_projected_channels)
    """
    # Always start with 1 for the binary inpainting mask
    cond_channels = 1
    
    # Add z_channels for each latent-space conditioning group
    cond_latent_groups = cvae_config.get('conditioning', {}).get('latent_space', [])
    for spec in cond_latent_groups:
        group_name = spec.get('group')
        if group_name in vae_groups_config:
            z_channels = vae_groups_config[group_name].get('z_channels', 0)
            cond_channels += z_channels
    
    # Scale to get projected channels
    cond_channel_scale = cvae_config.get('cond_channel_scale', 2.0)
    cond_projected_channels = max(int(cond_channels * cond_channel_scale), cond_channels)
    
    return cond_channels, cond_projected_channels


def build_unet_condition_config(stage_config, vae_groups_config, global_config=None):
    """
    Build U-Net condition_config from diffusion stage conditioning configuration.
    
    Args:
        stage_config: Diffusion stage config dict with 'conditioning' key
        vae_groups_config: VAE groups configuration dict
        global_config: Full global config dict (for scalar controls)
        
    Returns:
        condition_config dict for U-Net initialization
    """
    conditioning = stage_config.get('conditioning', {})
    
    condition_config = {
        'condition_types': ['image'],  # Always use image conditioning
        'image_condition_config': {}
    }
    
    # Compute pixel-space conditioning channels
    pixel_space_specs = conditioning.get('pixel_space') or []
    pixel_channels = len(pixel_space_specs)  # Each spec = 1 channel (e.g., inpainting_mask)
    
    # Compute latent-space conditioning channels
    latent_space_specs = conditioning.get('latent_space') or []
    latent_channels = 0
    for spec in latent_space_specs:
        group_name = spec.get('group')
        if group_name in vae_groups_config:
            z_channels = vae_groups_config[group_name].get('z_channels', 0)
            latent_channels += z_channels
    
    total_cond_channels = pixel_channels + latent_channels
    
    condition_config['image_condition_config'] = {
        'image_condition_input_channels': total_cond_channels,
        'image_condition_output_channels': min(total_cond_channels * 2, 128),  # Project to reasonable size
        'pixel_space_count': pixel_channels,
        'latent_space_count': latent_channels,
        'latent_space_specs': latent_space_specs  # Store for forward pass
    }
    
    # Generic scalar controls config
    # Supports multiple scalar controls: temperature, vegetation, building heights, etc.
    # Can be enabled per-stage with list of control names: scalar_controls: ["temperature", "building_coverage"]
    stage_scalar_controls = stage_config.get('scalar_controls', None)
    
    # Check if scalar controls are enabled (list of names or legacy boolean)
    scalar_controls_enabled = (
        isinstance(stage_scalar_controls, list) and len(stage_scalar_controls) > 0
    ) or (
        isinstance(stage_scalar_controls, bool) and stage_scalar_controls
    )
    
    if scalar_controls_enabled and global_config is not None:
        # Parse enabled scalar controls for this stage
        stage_control_names = stage_scalar_controls if isinstance(stage_scalar_controls, list) else None
        control_specs = parse_scalar_controls_config(global_config, stage_control_names=stage_control_names)
        
        if len(control_specs) > 0:
            # Build scalar conditioning config
            scalar_config = {}
            
            for spec in control_specs:
                control_name = spec['name']
                scalar_keys = spec['keys']
                training_cfg = spec.get('training', {})
                conditioning_cfg = spec.get('conditioning', {})
                
                # Extract per-key config
                for key in scalar_keys:
                    scalar_config[key] = {
                        'control_name': control_name,
                        'mlp_hidden': conditioning_cfg.get('mlp_hidden', 128),
                        'drop_prob': training_cfg.get('drop_prob', 0.1),
                        'unconditional_value': training_cfg.get('unconditional_value', 0.0)
                    }
            
            condition_config['scalar_condition_config'] = {
                'enabled': True,
                'scalars': scalar_config  # Dict: key -> config
            }
    
    return condition_config



def get_default_configs(vae_groups: dict, diffusion_stages: dict) -> tuple[dict, dict]:
    """
    Extract default VAE and U-Net configs from the first available groups/stages.
    
    Used as fallback when mode-specific configs are not available.
    
    Args:
        vae_groups: Dictionary of VAE group configurations
        diffusion_stages: Dictionary of diffusion stage configurations
        
    Returns:
        tuple: (vae_config, unet_config) dictionaries
    """
    # Get first VAE group config (or empty dict)
    first_vae_group = list(vae_groups.keys())[0] if vae_groups else None
    vae_config = vae_groups[first_vae_group] if first_vae_group else {}
    
    # Get first diffusion stage U-Net config (or empty dict)
    first_diffusion_stage = list(diffusion_stages.keys())[0] if diffusion_stages else None
    if first_diffusion_stage:
        unet_config = diffusion_stages[first_diffusion_stage].get('unet_config', {})
    else:
        unet_config = {}
    
    return vae_config, unet_config


def compute_patch_and_latent_sizes(
    dataset_config: dict,
    autoencoder_config: dict,
    ldm_config: dict = None,
    self=None
) -> tuple[int, int, int, int, int]:
    """
    Compute properly aligned patch and latent sizes.
    
    Ensures patch size is divisible by the VAE spatial downsample factor and,
    when applicable, by the U-Net downsample factor as well.
    
    Supports three standalone modes (and combinations):
      - VAE-only:  patch divisible by VAE factor
      - LDM mode:  patch divisible by VAE factor × U-Net factor
      - CVAE mode: patch divisible by VAE factor (no U-Net)
      - LDM + CVAE: patch divisible by VAE factor × U-Net factor
    
    Args:
        dataset_config: Dataset configuration with 'patch_size_m' and 'res'
        autoencoder_config: VAE config with 'down_sample'
        ldm_config: U-Net / diffusion config with 'down_channels' (optional)
        self: optional UrbanInpaintingDataset instance for setting vae_downsample_factor
    
    Returns:
        tuple: (patch_size, latent_size, vae_factor, unet_factor, total_divisor)
    """
    
    # Initial calculation
    pixel_size = dataset_config.get('patch_size_m', 650)
    im_res = dataset_config.get('res', 3)
    patch_size = int(pixel_size / im_res) # compute patch size in pixels
    patch_size = patch_size - (patch_size % 8) # make patch size divisible by 8
    
    # VAE spatial downsample factor (from encoder architecture)
    # Always computed from the autoencoder config — the CVAE/VAE encoder
    # needs the input to be spatially divisible by this factor.
    vae_downsample_factor = 2 ** sum(
        1 for ds in autoencoder_config.get('down_sample', [True, True, True]) if ds
    )
    
    # Ensure patch is divisible by VAE factor (needed for all modes)
    patch_size = patch_size - (patch_size % vae_downsample_factor)
    
    # U-Net / LDM additional downsample factor
    unet_downsample_factor = 1
    if ldm_config:
        num_down_layers = len(ldm_config.get('down_channels', [64, 128, 256, 512]))
        unet_downsample_factor = 2 ** num_down_layers
    
    # CVAE mode adds no extra spatial constraints beyond the VAE factor
    # (the CVAE operates at full resolution; its internal encoder/decoder
    # share the same architecture as the base VAE)
    
    # Combined divisor — must satisfy all components
    total_divisor = vae_downsample_factor * unet_downsample_factor
    patch_size = patch_size - (patch_size % total_divisor)
    
    # Latent size
    latent_size = patch_size // vae_downsample_factor
    
    if self is not None:
        self.vae_downsample_factor = vae_downsample_factor
    
    # Summary
    print(f"Using patch size: {patch_size} pixels ({patch_size * im_res} m at {im_res} m resolution)")
    print(f"  VAE downsample factor: {vae_downsample_factor}")
    if ldm_config:
        print(f"  U-Net downsample factor: {unet_downsample_factor}")
    print(f"  Total divisor: {total_divisor}")
    
    return (
        patch_size,
        latent_size,
        vae_downsample_factor,
        unet_downsample_factor,
        total_divisor
    )
    
def build_scalar_specs(config: dict, cvae_config: dict) -> dict:
    """
    Build scalar_specs dict for ConditionalVAE from config.
    
    Args:
        config: Full config dict
        cvae_config: CVAE inpainting config section for target group
        
    Returns:
        Dict mapping scalar keys to spec dicts with 'mlp_hidden'
    """
    scalar_control_names = cvae_config.get('scalar_controls', [])
    if not scalar_control_names:
        return {}
    
    control_specs = parse_scalar_controls_config(config, stage_control_names=scalar_control_names)
    
    scalar_specs = {}
    for spec in control_specs:
        conditioning_cfg = spec.get('conditioning', {})
        for key in spec['keys']:
            scalar_specs[key] = {
                'mlp_hidden': conditioning_cfg.get('mlp_hidden', 128)
            }
    
    return scalar_specs
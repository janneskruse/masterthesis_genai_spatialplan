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
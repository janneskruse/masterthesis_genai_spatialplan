import torch
import numpy as np

def apply_range_filter(data, filter_config):
    """
    Apply range filters to data array based on configuration.
    
    Supports:
    - gte (>=): Greater than or equal
    - lte (<=): Less than or equal
    - gt (>): Greater than
    - lt (<): Less than
    - eq (==): Equal to
    
    Args:
        data: numpy array or torch tensor
        filter_config: dict with filter specifications (e.g., {'gte': 0.2, 'lte': 0.8})
        
    Returns:
        Binary mask (0 or 1) where conditions are met
    """
    if filter_config is None or not isinstance(filter_config, dict):
        return data
    
    # Convert to numpy for easier manipulation
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        device = data.device
        data = data.cpu().numpy()
    
    # Create mask starting with all True
    mask = np.ones_like(data, dtype=bool)
    
    # Apply filters
    if 'gte' in filter_config:
        mask &= (data >= filter_config['gte'])
    if 'lte' in filter_config:
        mask &= (data <= filter_config['lte'])
    if 'gt' in filter_config:
        mask &= (data > filter_config['gt'])
    if 'lt' in filter_config:
        mask &= (data < filter_config['lt'])
    if 'eq' in filter_config:
        mask &= (data == filter_config['eq'])
    
    # Convert mask to binary (0 or 1)
    result = mask.astype(np.float32)
    
    # Convert back to torch if needed
    if is_torch:
        result = torch.from_numpy(result).to(device)
    
    return result


def collate_fn(batch):
    """
    Custom collate function to make sure conditioning inputs are properly batched.
    """
    if isinstance(batch[0], tuple):
        # With conditioning
        images = torch.stack([item[0] for item in batch])
        
        # Get first sample to check keys
        sample_cond = batch[0][1]
        
        cond_inputs = {}
        for key in sample_cond.keys():
            if key == 'meta':
                # Meta is a dict, just collect as list, don't stack
                cond_inputs[key] = [item[1][key] for item in batch]
            else:
                # Stack tensors
                cond_inputs[key] = torch.stack([item[1][key] for item in batch])
        
        return images, cond_inputs
    else:
        # Just images, no conditioning
        return torch.stack(batch)
import torch

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
"""
==============================================================================
Utilities for computing statistics
==============================================================================
"""
 

###### import libraries ######
# Standard libraries

# Data Science/ML libraries
import torch

def compute_temperature_statistic(
    lst_tensor: torch.Tensor,
    statistic: str = 'p95',
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute LST statistic from full-resolution tensor.
    
    Args:
        lst_tensor: LST tensor [B, 1, H, W] in [0, 1] range
        statistic: 'p95', 'mean', 'max', 'p99'
        mask: Optional mask [B, 1, H, W] where 1 = inside region
        
    Returns:
        Statistic tensor [B, 1]
    """
    B = lst_tensor.shape[0]
    results = []
    
    for b in range(B):
        lst_flat = lst_tensor[b].flatten()  # [H*W]
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask[b].flatten().bool()
            lst_flat = lst_flat[mask_flat]
        
        # Remove NaN/invalid values
        lst_flat = lst_flat[~torch.isnan(lst_flat)]
        lst_flat = lst_flat[lst_flat > 0]  # Temperature should be positive
        
        if len(lst_flat) == 0:
            # Fallback to 0 if no valid pixels
            results.append(torch.tensor(0.0, device=lst_tensor.device))
            continue
        
        if statistic == 'p95':
            val = torch.quantile(lst_flat, 0.95)
        elif statistic == 'p99':
            val = torch.quantile(lst_flat, 0.99)
        elif statistic == 'mean':
            val = lst_flat.mean()
        elif statistic == 'max':
            val = lst_flat.max()
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        results.append(val)
    
    return torch.stack(results).unsqueeze(1)  # [B, 1]
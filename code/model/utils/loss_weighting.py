"""
=============================================================================
Loss Weighting Strategies for Diffusion Training.

Implements timestep-dependent loss weighting to improve convergence:
- Simple: Uniform weights (baseline)
- SNR: Signal-to-Noise Ratio weighting
- Min-SNR: Clipped SNR weighting (recommended)
- V-Loss: V-prediction objective

Min-SNR is particularly effective at preventing noisy timesteps from
dominating training loss, leading to faster convergence and better quality.
==============================================================================
"""

###### import libraries ######
# Standard libraries
from typing import Optional

# Data Science/ML
import torch


def compute_loss_weights(
    timesteps: torch.Tensor,
    scheduler,
    loss_type: str = 'simple',
    min_snr_gamma: float = 5.0
) -> torch.Tensor:
    """
    Compute loss weights based on timestep and weighting strategy.
    
    Different timesteps have different difficulty levels:
    - Low noise (t=0): Easy to denoise, high SNR
    - High noise (t=999): Hard to denoise, low SNR
    
    Problem: Without weighting, high-noise timesteps dominate loss
    because they have larger prediction errors. This slows convergence.
    
    Solution: Weight loss by timestep difficulty using SNR.
    
    **Weighting Strategies:**
    
    1. **'simple'** (baseline):
       - Uniform weights = 1.0 for all timesteps
       
    2. **'snr'**:
       - weight = SNR(t) = alpha_t / (1 - alpha_t)
       - Upweights clean images, downweights noisy
       - Can over-focus on low-noise timesteps
       
    3. **'min_snr'** (recommended):
       - weight = min(SNR(t), gamma) / SNR(t)
       - Clips very high SNR to prevent over-focus on easy timesteps
       - Balances all timesteps more evenly
       - used in Stable Diffusion, Imagen, DALL-E
       - Typical gamma: 5.0
       
    4. **'v_loss'**:
       - weight = alpha_t * (1 - alpha_t)
       - V-prediction parameterization
       - Alternative to epsilon prediction
    
    Args:
        timesteps: Tensor of timestep indices [B] (0 to num_timesteps-1)
        scheduler: Noise scheduler with alpha_cum_prod attribute
        loss_type: One of ['simple', 'snr', 'min_snr', 'v_loss']
        min_snr_gamma: Clipping value for min_snr (5.0 is standard)
        
    Returns:
        Loss weights tensor [B], multiply with per-sample loss
        
    Example:
        >>> # In training loop:
        >>> t = torch.randint(0, 1000, (batch_size,))
        >>> noise_pred = model(noisy_x, t, cond)
        >>> 
        >>> # Compute base loss (per sample)
        >>> loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none')
        >>> loss_per_sample = loss_per_sample.mean(dim=[1,2,3])  # [B]
        >>> 
        >>> # Apply timestep weighting
        >>> weights = compute_loss_weights(t, scheduler, 'min_snr', min_snr_gamma=5.0)
        >>> weighted_loss = (loss_per_sample * weights).mean()
        
    References:
        - Min-SNR: "Efficient Diffusion Training via Min-SNR Weighting"
          https://arxiv.org/abs/2303.09556
        - Used in Stable Diffusion, Imagen, DALL-E
    """
    device = timesteps.device
    
    # Get cumulative alpha product for each timestep
    # alpha_cum_prod[t] = product of (1 - beta) from 0 to t
    alpha_cum_prod = scheduler.alpha_cum_prod.to(device)[timesteps]  # [B]
    
    # Validate loss type
    valid_types = ['simple', 'snr', 'min_snr', 'v_loss']
    if loss_type not in valid_types:
        raise ValueError(
            f"Invalid loss_type: '{loss_type}'. Must be one of {valid_types}"
        )
    
    # Strategy 1: Simple (uniform weights)
    if loss_type == 'simple':
        return torch.ones_like(timesteps, dtype=torch.float32)
    
    # Compute SNR (Signal-to-Noise Ratio)
    # SNR(t) = alpha_t / (1 - alpha_t)
    # High alpha (clean) → high SNR
    # Low alpha (noisy) → low SNR
    snr = alpha_cum_prod / (1.0 - alpha_cum_prod)
    
    # Strategy 2: SNR weighting
    if loss_type == 'snr':
        # Upweight clean images (high SNR), downweight noisy (low SNR)
        return snr
    
    # Strategy 3: Min-SNR weighting
    if loss_type == 'min_snr':
        # Clip SNR to gamma to prevent over-focus on easy timesteps
        # weight = min(SNR, gamma) / SNR
        #        = 1.0 when SNR <= gamma (no change)
        #        < 1.0 when SNR > gamma (downweight easy timesteps)
        
        snr_clamped = torch.clamp(snr, max=min_snr_gamma)
        weights = snr_clamped / snr
        
        return weights
    
    # Strategy 4: V-loss weighting
    if loss_type == 'v_loss':
        weights = alpha_cum_prod * (1.0 - alpha_cum_prod)
        return weights
    
    # Should never reach here due to validation above
    raise ValueError(f"Unhandled loss_type: {loss_type}")


def compute_snr_curve(scheduler, device: str = 'cpu') -> torch.Tensor:
    """
    Compute SNR values for all timesteps in scheduler.
    
    Useful for visualization and analysis of noise schedule.
    
    Args:
        scheduler: Noise scheduler with alpha_cum_prod
        device: Device to compute on
        
    Returns:
        SNR values [num_timesteps]
        
    Example:
        >>> snr = compute_snr_curve(scheduler)
        >>> plt.plot(snr.numpy())
        >>> plt.xlabel('Timestep')
        >>> plt.ylabel('SNR')
        >>> plt.yscale('log')
        >>> plt.show()
    """
    alpha_cum_prod = scheduler.alpha_cum_prod.to(device)
    snr = alpha_cum_prod / (1.0 - alpha_cum_prod)
    return snr


def plot_loss_weights(
    scheduler,
    min_snr_gamma: float = 5.0,
    num_samples: int = 1000
) -> None:
    """
    Plot loss weight curves for different strategies.
    
    Helps visualize how different weighting schemes affect training.
    Requires matplotlib.
    
    Args:
        scheduler: Noise scheduler
        min_snr_gamma: Gamma value for min_snr
        num_samples: Number of timesteps to sample
        
    Example:
        >>> from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
        >>> scheduler = LinearNoiseScheduler(1000, 0.0001, 0.02)
        >>> plot_loss_weights(scheduler, min_snr_gamma=5.0)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")
        return
    
    # Sample timesteps uniformly
    timesteps = torch.linspace(0, scheduler.num_timesteps - 1, num_samples).long()
    
    # Compute weights for each strategy
    strategies = {
        'Simple (uniform)': 'simple',
        'SNR': 'snr',
        f'Min-SNR (γ={min_snr_gamma})': 'min_snr',
        'V-Loss': 'v_loss'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, loss_type) in enumerate(strategies.items()):
        weights = compute_loss_weights(
            timesteps,
            scheduler,
            loss_type=loss_type,
            min_snr_gamma=min_snr_gamma
        ).numpy()
        
        ax = axes[idx]
        ax.plot(timesteps.numpy(), weights, linewidth=2)
        ax.set_xlabel('Timestep (0=clean, 999=noisy)')
        ax.set_ylabel('Loss Weight')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at weight=1.0 for reference
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Baseline')
        ax.legend()
    
    plt.suptitle('Loss Weighting Strategies Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Also plot SNR curve
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    snr = compute_snr_curve(scheduler).numpy()
    ax2.plot(snr, linewidth=2)
    ax2.set_xlabel('Timestep (0=clean, 999=noisy)')
    ax2.set_ylabel('SNR (Signal-to-Noise Ratio)')
    ax2.set_title('SNR Curve Across Diffusion Timesteps')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=min_snr_gamma, color='r', linestyle='--', alpha=0.5, 
                label=f'Min-SNR γ={min_snr_gamma}')
    ax2.legend()
    plt.tight_layout()
    plt.show()


def apply_loss_weighting(
    loss_per_sample: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    loss_type: str = 'simple',
    min_snr_gamma: float = 5.0
) -> torch.Tensor:
    """
    Convenience function to compute and apply loss weights in one step.
    
    Args:
        loss_per_sample: Unreduced loss per sample [B]
        timesteps: Timestep indices [B]
        scheduler: Noise scheduler
        loss_type: Weighting strategy
        min_snr_gamma: Gamma for min_snr
        
    Returns:
        Weighted scalar loss (reduced)
        
    Example:
        >>> # Instead of:
        >>> # weights = compute_loss_weights(t, scheduler, 'min_snr')
        >>> # loss = (loss_per_sample * weights).mean()
        >>> 
        >>> # Use:
        >>> loss = apply_loss_weighting(loss_per_sample, t, scheduler, 'min_snr')
    """
    weights = compute_loss_weights(timesteps, scheduler, loss_type, min_snr_gamma)
    
    # Apply weights and reduce
    weighted_loss = (loss_per_sample * weights).mean()
    
    return weighted_loss

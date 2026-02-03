"""
Latent-space guidance utilities for diffusion sampling.
Applies gradient-based guidance using LatentLSTPredictor directly on VAE latents.
"""
###### import libraries ######
# Data Science/ML libraries
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


def apply_latent_lst_guidance(
    x: torch.Tensor,
    t: torch.Tensor,
    noise_pred: torch.Tensor,
    model: torch.nn.Module,
    scheduler: Any,
    cond_input: Dict[str, torch.Tensor],
    latent_predictor: torch.nn.Module,
    target_p95: float,
    guidance_scale: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    clamp_x0: float = 3.0,
) -> torch.Tensor:
    """
    Apply LST predictor guidance directly in latent space (no VAE decode needed).
    
    Uses classifier guidance approach: modify noise prediction based on gradient
    of the latent LST predictor w.r.t. the predicted x0.
    
    Phase 2 implementation: Soft guidance that steers generation toward target temperature.
    
    Args:
        x: Current noisy latent [B, C, H, W]
        t: Current timestep tensor [B,]
        noise_pred: Base noise prediction from diffusion model [B, C, H, W]
        model: Diffusion U-Net model (for forward pass with gradients)
        scheduler: Noise scheduler (for alpha values)
        cond_input: Conditioning input dict
        latent_predictor: Trained LatentLSTPredictor model
        target_p95: Target temperature p95 in normalized [0, 1] range
        guidance_scale: Strength of guidance (0.0 = no guidance)
        mask: Optional inpainting mask [B, 1, H, W] (guidance only in masked region)
        clamp_x0: Clamp value for predicted x0 (stability)
        
    Returns:
        Guided noise prediction [B, C, H, W]
    """
    if latent_predictor is None or guidance_scale == 0.0:
        return noise_pred
    
    device = x.device
    batch_size = x.shape[0]
    
    # Target tensor
    target = torch.full((batch_size, 1), target_p95, device=device, dtype=torch.float32)
    
    # Enable gradients for guidance computation
    x_guide = x.detach().clone().requires_grad_(True)
    
    # Forward pass with gradients
    noise_pred_grad = model(x_guide, t, cond_input=cond_input)
    
    # Get alpha values for x0 prediction
    # Handle both scalar timesteps and batched timesteps
    t_indices = t.long().view(-1)
    alpha_cum_prod = scheduler.alpha_cum_prod.to(device)
    alpha_t = alpha_cum_prod[t_indices]
    
    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1, 1)
    
    # Predict x0 from current noisy x using noise prediction
    # x0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
    x0_pred = (x_guide - sqrt_one_minus_alpha_t * noise_pred_grad) / (sqrt_alpha_t + 1e-8)
    x0_pred = torch.clamp(x0_pred, -clamp_x0, clamp_x0)
    
    # Predict LST p95 from x0 latent (no VAE decode!)
    lst_pred = latent_predictor(x0_pred)  # [B, 1]
    
    # Compute guidance loss
    # We want to minimize |predicted_lst - target_lst|
    # Gradient descent on this loss → steer toward target
    lst_loss = F.mse_loss(lst_pred, target)
    
    # Compute gradient of loss w.r.t. x
    grad = torch.autograd.grad(lst_loss, x_guide, retain_graph=False)[0]
    
    # Apply mask if provided (only guide inside inpainting region)
    if mask is not None:
        # Ensure mask matches latent spatial dimensions
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
        grad = grad * mask
    
    # Apply guidance: modify noise prediction
    # noise_pred_guided = noise_pred - scale * grad
    # Negative gradient moves toward lower loss (closer to target)
    noise_pred_guided = noise_pred - guidance_scale * grad.detach()
    
    return noise_pred_guided


def compute_latent_lst_prediction(
    x: torch.Tensor,
    latent_predictor: torch.nn.Module,
    lst_max: float = 80.0,
) -> Dict[str, float]:
    """
    Compute LST prediction from latent for Phase 3 hard check.
    
    Args:
        x: Clean latent [B, C, H, W] (after sampling completes)
        latent_predictor: Trained LatentLSTPredictor model
        lst_max: Maximum LST value for denormalization
        
    Returns:
        Dict with 'p95_normalized' and 'p95_celsius' values
    """
    with torch.no_grad():
        p95_normalized = latent_predictor(x)  # [B, 1]
        p95_celsius = p95_normalized * lst_max
    
    return {
        'p95_normalized': p95_normalized.mean().item(),
        'p95_celsius': p95_celsius.mean().item(),
    }


def should_apply_guidance(
    step_idx: int,
    num_steps: int,
    eval_every_n_steps: int = 1,
    warmup_fraction: float = 0.0,
    cooldown_fraction: float = 0.0,
) -> bool:
    """
    Determine if guidance should be applied at current step.
    
    Supports:
    - Periodic guidance (every N steps)
    - Warmup: Skip early steps when noise is too high
    - Cooldown: Skip final steps for quality
    
    Args:
        step_idx: Current step index (0 = noisiest, num_steps-1 = cleanest)
        num_steps: Total number of denoising steps
        eval_every_n_steps: Apply guidance every N steps
        warmup_fraction: Skip first X% of steps (e.g., 0.1 = skip first 10%)
        cooldown_fraction: Skip last X% of steps
        
    Returns:
        True if guidance should be applied
    """
    # Periodic check
    if step_idx % eval_every_n_steps != 0:
        return False
    
    # Progress through sampling (0.0 = start, 1.0 = end)
    progress = (num_steps - step_idx - 1) / num_steps
    
    # Warmup: skip early steps (high noise)
    if warmup_fraction > 0 and progress < warmup_fraction:
        return False
    
    # Cooldown: skip final steps
    if cooldown_fraction > 0 and progress > (1.0 - cooldown_fraction):
        return False
    
    return True


class LatentGuidanceConfig:
    """Configuration container for latent guidance during sampling."""
    
    def __init__(
        self,
        enabled: bool = False,
        scale: float = 1.0,
        target_p95_celsius: float = 35.0,
        eval_every_n_steps: int = 1,
        warmup_fraction: float = 0.1,
        cooldown_fraction: float = 0.1,
        lst_max: float = 80.0,
    ):
        """
        Args:
            enabled: Whether to apply guidance
            scale: Guidance strength
            target_p95_celsius: Target temperature in Celsius
            eval_every_n_steps: Apply guidance every N steps
            warmup_fraction: Skip first X% of steps
            cooldown_fraction: Skip last X% of steps
            lst_max: Max LST for normalization
        """
        self.enabled = enabled
        self.scale = scale
        self.target_p95_celsius = target_p95_celsius
        self.target_p95_normalized = target_p95_celsius / lst_max
        self.eval_every_n_steps = eval_every_n_steps
        self.warmup_fraction = warmup_fraction
        self.cooldown_fraction = cooldown_fraction
        self.lst_max = lst_max
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LatentGuidanceConfig':
        """Create from config dict."""
        guidance_cfg = config.get('temperature_control', {}).get('guidance', {})
        return cls(
            enabled=guidance_cfg.get('enabled', False),
            scale=guidance_cfg.get('scale', 1.0),
            target_p95_celsius=guidance_cfg.get('target_p95_celsius', 35.0),
            eval_every_n_steps=guidance_cfg.get('eval_every_n_steps', 1),
            warmup_fraction=guidance_cfg.get('warmup_fraction', 0.1),
            cooldown_fraction=guidance_cfg.get('cooldown_fraction', 0.1),
            lst_max=guidance_cfg.get('lst_max', 80.0),
        )
    
    def __repr__(self) -> str:
        return (
            f"LatentGuidanceConfig(enabled={self.enabled}, scale={self.scale}, "
            f"target={self.target_p95_celsius}°C, every={self.eval_every_n_steps} steps)"
        )

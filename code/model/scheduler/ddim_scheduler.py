"""
DDIM (Denoising Diffusion Implicit Models) Scheduler.

Fast sampling algorithm that produces high-quality samples in 50-100 steps
instead of 1000 steps. Up to 20x faster than DDPM with similar quality.

Key features:
- Deterministic sampling (eta=0) for reproducibility
- Stochastic sampling (eta=1) similar to DDPM
- Flexible timestep scheduling (skip steps)
- Compatible with inpainting

Reference:
    "Denoising Diffusion Implicit Models" (Song et al., 2020)
    https://arxiv.org/abs/2010.02502
"""

###### import libraries ######
# Standard libraries
import numpy as np

# Data Science/ML
import torch


class DDIMScheduler:
    """
    DDIM scheduler for fast diffusion sampling.
    
    DDIM achieves similar quality to DDPM in much fewer steps by:
    1. Using implicit (non-Markovian) process instead of explicit
    2. Deterministic or controllably stochastic sampling
    3. Skipping timesteps via sub-sequence sampling
    
    Typical usage: 50 steps instead of 1000 (20x speedup)
    
    Args:
        num_timesteps: Total diffusion timesteps (usually 1000)
        beta_start: Starting beta value (0.0001)
        beta_end: Ending beta value (0.02)
        ddim_steps: Number of actual sampling steps (50-100 recommended)
        ddim_eta: Stochasticity parameter
                  0.0 = fully deterministic (recommended)
                  1.0 = stochastic like DDPM
                  
    Example:
        >>> # Training (same as DDPM)
        >>> scheduler = DDIMScheduler(1000, 0.0001, 0.02, ddim_steps=50)
        >>> noisy = scheduler.add_noise(x, noise, t)
        >>> 
        >>> # Sampling (20x faster than DDPM)
        >>> x = torch.randn_like(image)
        >>> for i in reversed(range(scheduler.ddim_steps)):
        >>>     t = scheduler.ddim_timesteps[i]
        >>>     noise_pred = model(x, t, cond)
        >>>     x, x0 = scheduler.sample_prev_timestep(x, noise_pred, i)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        ddim_steps: int = 50,
        ddim_eta: float = 0.0
    ):
        """
        Initialize DDIM scheduler.
        
        Args:
            num_timesteps: Total timesteps in diffusion process
            beta_start: Initial beta value
            beta_end: Final beta value
            beta_schedule: Schedule type - 'linear' (default) or 'cosine'
            ddim_steps: Number of sampling steps (< num_timesteps)
            ddim_eta: Stochasticity (0=deterministic, 1=DDPM-like)
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        
        # Create beta schedule (same options as LinearNoiseScheduler)
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'linear':
            # Scaled-linear schedule (original DDPM)
            self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
            )
        else:
            raise ValueError(
                f"Unknown beta_schedule: '{beta_schedule}'. "
                f"Supported: 'linear', 'cosine'"
            )
        
        # Pre-compute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)
        
        # Create DDIM timestep schedule
        # Evenly spaced indices from full timestep range
        # e.g., [0, 20, 40, ..., 980] for ddim_steps=50, num_timesteps=1000
        self.ddim_timesteps = self._create_ddim_timesteps()
        
        # Pre-compute DDIM alpha values for each sampling step
        self.ddim_alpha_cum_prod = self.alpha_cum_prod[self.ddim_timesteps]
        self.ddim_alpha_cum_prod_prev = self._get_prev_alpha_cum_prod()
        
        print(f"✓ DDIM Scheduler initialized:")
        print(f"  Beta schedule: {beta_schedule}")
        print(f"  Total timesteps: {num_timesteps}")
        print(f"  Sampling steps: {ddim_steps} ({ddim_steps/num_timesteps*100:.1f}% of total)")
        print(f"  Eta (stochasticity): {ddim_eta:.2f}")
        print(f"  Speedup: {num_timesteps/ddim_steps:.1f}x faster than DDPM")
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule (same as LinearNoiseScheduler).
        
        Args:
            timesteps: Number of diffusion steps
            s: Offset parameter (default: 0.008)
            
        Returns:
            Tensor of beta values [timesteps]
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _create_ddim_timesteps(self) -> torch.Tensor:
        """
        Create evenly-spaced timestep schedule for DDIM sampling.
        
        Returns:
            Timestep indices [ddim_steps] ranging from 0 to num_timesteps-1
        """
        # Create evenly spaced indices
        # Example: num_timesteps=1000, ddim_steps=50
        # → [0, 20, 40, 60, ..., 960, 980]
        step_ratio = self.num_timesteps // self.ddim_steps
        ddim_timesteps = (np.arange(0, self.ddim_steps) * step_ratio).astype(np.int64)
        
        # Add 1 to start from timestep 1 instead of 0 (common practice)
        ddim_timesteps = ddim_timesteps + 1
        
        return torch.from_numpy(ddim_timesteps).long()
    
    def _get_prev_alpha_cum_prod(self) -> torch.Tensor:
        """
        Get alpha_cum_prod for previous timestep in DDIM schedule.
        
        Returns:
            Alpha values for t-1 in DDIM schedule [ddim_steps]
        """
        # Shift timesteps by 1 to get previous values
        # First timestep uses alpha=1.0 (clean image)
        prev_timesteps = torch.cat([
            torch.tensor([0]),  # t-1 for first step is 0 (clean)
            self.ddim_timesteps[:-1]
        ])
        
        alpha_cum_prod_prev = self.alpha_cum_prod[prev_timesteps]
        
        return alpha_cum_prod_prev
    
    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean image.
        
        Same as DDPM - this doesn't change between schedulers.
        
        Args:
            original: Clean image [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]
            t: Timestep indices [B] (0 to num_timesteps-1)
            
        Returns:
            Noisy image [B, C, H, W]
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape to match image dimensions
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # q(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
        return (sqrt_alpha_cum_prod.to(original.device) * original +
                sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        ddim_step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM reverse step: denoise from x_t to x_{t-1}.
        
        Uses DDIM formula instead of DDPM:
        - Predicts x0 from current xt and noise prediction
        - Computes x_{t-1} using implicit process
        - Optionally adds noise controlled by eta
        
        Args:
            xt: Current noisy sample [B, C, H, W]
            noise_pred: Model's noise prediction [B, C, H, W]
            ddim_step: Current step index (0 to ddim_steps-1)
                      NOT the full timestep! Use this for indexing.
                      
        Returns:
            Tuple of (x_{t-1}, x0_pred):
                - x_{t-1}: Denoised sample [B, C, H, W]
                - x0_pred: Predicted clean image [B, C, H, W]
        """
        device = xt.device
        
        # Get alpha values for current and previous step
        alpha_cum_prod_t = self.ddim_alpha_cum_prod[ddim_step].to(device)
        alpha_cum_prod_t_prev = self.ddim_alpha_cum_prod_prev[ddim_step].to(device)
        
        # Predict x0 from xt and noise
        # x0 = (xt - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_cum_prod_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_cum_prod_t)
        
        x0_pred = (xt - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Clamp x0 prediction for stability (latents may be outside [-1,1])
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        
        # Last step: return clean prediction
        if ddim_step == 0:
            return x0_pred, x0_pred
        
        # Compute variance (sigma) controlled by eta
        # eta=0: deterministic (no noise)
        # eta=1: stochastic like DDPM
        sqrt_alpha_t_prev = torch.sqrt(alpha_cum_prod_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_cum_prod_t_prev)
        
        # Variance formula
        variance = (
            (1.0 - alpha_cum_prod_t_prev) /
            (1.0 - alpha_cum_prod_t) *
            (1.0 - alpha_cum_prod_t / alpha_cum_prod_t_prev)
        )
        sigma = self.ddim_eta * torch.sqrt(variance)
        
        # Compute direction pointing to xt
        # This is the "denoising direction"
        pred_dir = torch.sqrt(1.0 - alpha_cum_prod_t_prev - sigma**2) * noise_pred
        
        # Compute x_{t-1}
        # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + pred_dir + noise
        x_prev = sqrt_alpha_t_prev * x0_pred + pred_dir
        
        # Add noise if eta > 0
        if self.ddim_eta > 0:
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma * noise
        
        return x_prev, x0_pred
    
    def sample_prev_timestep_inpainting(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        ddim_step: int,
        x_context: torch.Tensor,
        mask: torch.Tensor,
        noise_context: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM reverse step with hard inpainting.
        
        Clamps known regions to context after each denoising step.
        Uses FIXED noise_context for temporal consistency.
        
        Args:
            xt: Current noisy sample [B, C, H, W]
            noise_pred: Model's noise prediction [B, C, H, W]
            ddim_step: Current DDIM step index (0 to ddim_steps-1)
            x_context: Known context (ground truth) [B, C, H, W]
            mask: Inpainting mask [B, 1, H, W], 1=regenerate, 0=keep
            noise_context: FIXED noise tensor [B, C, H, W], sampled once per generation
            
        Returns:
            Tuple of (x_{t-1}, x0_pred)
        """
        # Standard DDIM denoising step
        xt_prev, x0_pred = self.sample_prev_timestep(xt, noise_pred, ddim_step)
        
        # Last step: use clean context
        if ddim_step == 0:
            xt_prev = mask * xt_prev + (1 - mask) * x_context
            return xt_prev, x0_pred
        
        # Clamp known region to properly noised context
        if noise_context is None:
            raise ValueError(
                "noise_context is required for inpainting when ddim_step > 0. "
                "Sample it ONCE per generation: noise_context = torch.randn_like(x_context)"
            )
        
        # Get previous timestep from DDIM schedule
        # ddim_timesteps contains full timestep indices (e.g., [1, 21, 41, ...])
        # So we can use them directly with add_noise()
        t_prev = self.ddim_timesteps[ddim_step - 1] if ddim_step > 0 else 0
        t_batch = torch.full((x_context.shape[0],), t_prev, device=x_context.device, dtype=torch.long)
        
        # Noise context to match previous DDIM timestep
        x_context_noisy = self.add_noise(x_context, noise_context, t_batch)
        
        # Blend: regenerate masked region, keep context elsewhere
        xt_prev = mask * xt_prev + (1 - mask) * x_context_noisy
        
        return xt_prev, x0_pred

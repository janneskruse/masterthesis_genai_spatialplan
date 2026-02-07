"""
==============================================================================
Inpainting Sampling Strategies for Latent Diffusion Models

Implements different inpainting strategies that can be combined with
any base noise sampler (DDPM or DDIM).

- Noise Sampler (ddpm/ddim): Controls HOW to denoise (x_t -> x_{t-1})
- Inpainting Sampler (standard/repaint/lanpaint): Controls boundary harmonization strategy

References:
- RePaint: https://arxiv.org/abs/2201.09865 (Lugmayr et al., CVPR 2022)
  "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
  
  Implementation adapted from:
  https://github.com/andreas128/RePaint
  Original code: Copyright (c) 2022 Huawei Technologies Co., Ltd.
  Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
  
- LanPaint: https://arxiv.org/abs/2502.03491 (Zheng et al., 2025)
  "LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling"
==============================================================================
"""

###### import libraries ######
# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

# Data handling
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
from model.utils.lanpaint import StochasticHarmonicOscillator


@dataclass
class InpaintingSamplerConfig:
    """Configuration for inpainting samplers."""
    sampler_type: str = 'standard'  # 'standard', 'repaint', 'lanpaint'
    
    # Base sampler config
    num_steps: int = 50  # Number of denoising steps
    
    # RePaint parameters
    jump_length: int = 10  # Timesteps to jump back (official default: 10)
    jump_n_sample: int = 10  # Number of times to resample at each jump (official default: 10)
    start_resampling: int = 100000000  # Only resample when t <= this (official default: very large = always)
    
    # LanPaint parameters
    lanpaint_num_steps: int = 5       # K: Inner Langevin iterations per denoising step ("thinking depth")
    lanpaint_lambda: float = 16.0     # λ: BiG score alignment strength (higher = stricter boundary matching)
    lanpaint_step_size: float = 0.2   # η: Step size for Langevin dynamics
    lanpaint_beta: float = 1.0        # β: Step size ratio masked/unmasked (lower compensates high λ)
    lanpaint_friction: float = 15.0   # γ: Friction for Fast Langevin Dynamics (higher = more stable)
    lanpaint_early_stop: int = 1      # Stop LanPaint iterations before final step (prevents artifacts)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'InpaintingSamplerConfig':
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            sampler_type=config.get('type', 'standard'),
            num_steps=config.get('num_steps', 50),
            # RePaint
            jump_length=config.get('jump_length', 10),
            jump_n_sample=config.get('jump_n_sample', 10),
            start_resampling=config.get('start_resampling', 100000000),
            # LanPaint
            lanpaint_num_steps=config.get('lanpaint_num_steps', 5),
            lanpaint_lambda=config.get('lanpaint_lambda', 16.0),
            lanpaint_step_size=config.get('lanpaint_step_size', 0.2),
            lanpaint_beta=config.get('lanpaint_beta', 1.0),
            lanpaint_friction=config.get('lanpaint_friction', 15.0),
            lanpaint_early_stop=config.get('lanpaint_early_stop', 1),
        )


class InpaintingSamplerBase(ABC):
    """
    Abstract base class for inpainting sampling strategies.
    
    All samplers share the same interface but implement different strategies
    for harmonizing generated content with known regions.
    """
    
    def __init__(
        self,
        scheduler,
        config: InpaintingSamplerConfig,
        device: torch.device = None
    ):
        """
        Initialize the inpainting sampler.
        
        Args:
            scheduler: Base noise scheduler (DDPM or DDIM)
            config: Sampler configuration
            device: Computation device
        """
        self.scheduler = scheduler
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine if using DDIM (has ddim_steps attribute)
        self.is_ddim = hasattr(scheduler, 'ddim_steps')
        self.num_steps = scheduler.ddim_steps if self.is_ddim else scheduler.num_timesteps
    
    @abstractmethod
    def sample(
        self,
        model: Callable,
        x_init: torch.Tensor,
        x_context: torch.Tensor,
        mask: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 0.0,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Run the inpainting sampling process.
        
        Args:
            model: Diffusion U-Net model
            x_init: Initial noisy latent [B, C, H, W]
            x_context: Clean context latent (known region) [B, C, H, W]
            mask: Binary mask [B, 1, H, W] (1=inpaint/generate, 0=keep/known)
            cond_input: Conditioning dictionary for model
            uncond_input: Unconditional input for CFG (optional)
            guidance_scale: Classifier-free guidance scale (0=no guidance)
            show_progress: Whether to show progress bar
            
        Returns:
            Inpainted latent [B, C, H, W]
        """
        pass # To be implemented by subclasses
    
    def _get_timestep_schedule(self) -> list:
        """
        Get step indices for denoising loop.
        
        Returns indices [num_steps-1, num_steps-2, ..., 0] reversed for denoising.
        Note: self.num_steps is set in __init__ based on scheduler type:
        - DDIM: num_steps = scheduler.ddim_steps (e.g., 50)
        - DDPM: num_steps = scheduler.num_timesteps (e.g., 1000)
        """
        return list(reversed(range(self.num_steps)))
    
    def _get_timestep_value(self, step_idx: int) -> int:
        """Convert step index to actual timestep value for model."""
        if self.is_ddim:
            return self.scheduler.ddim_timesteps[step_idx].item()
        else:
            return step_idx
    
    def _predict_noise(
        self,
        model: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """Predict noise with optional classifier-free guidance."""
        if guidance_scale > 0 and uncond_input is not None:
            # Classifier-free guidance
            noise_pred_cond = model(x, t, cond_input=cond_input)
            noise_pred_uncond = model(x, t, cond_input=uncond_input)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model(x, t, cond_input=cond_input)
        return noise_pred
    
    def _denoise_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        step_idx: int,
        x_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        noise_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one denoising step with optional inpainting.
        
        Returns:
            Tuple of (x_{t-1}, x0_pred)
        """
        if self.is_ddim:
            if x_context is not None and mask is not None:
                return self.scheduler.sample_prev_timestep_inpainting(
                    x, noise_pred, step_idx, x_context, mask, noise_context=noise_context
                )
            else:
                return self.scheduler.sample_prev_timestep(x, noise_pred, step_idx)
        else:
            # DDPM
            t_tensor = torch.tensor([step_idx], device=self.device)
            if x_context is not None and mask is not None:
                return self.scheduler.sample_prev_timestep_inpainting(
                    x, noise_pred, t_tensor, x_context, mask, noise_context=noise_context
                )
            else:
                return self.scheduler.sample_prev_timestep(x, noise_pred, t_tensor)
    
    def _add_noise(self, x: torch.Tensor, noise: torch.Tensor, timestep: int) -> torch.Tensor:
        """Add noise to a clean sample at given timestep."""
        t_tensor = torch.full((x.shape[0],), timestep, device=self.device, dtype=torch.long)
        return self.scheduler.add_noise(x, noise, t_tensor)


class StandardInpaintingSampler(InpaintingSamplerBase):
    """
    Standard inpainting sampler using the Replace method.
    
    At each step:
    1. Denoise the full latent
    2. Replace known region with properly noised ground truth
    """
    
    def sample(
        self,
        model: Callable,
        x_init: torch.Tensor,
        x_context: torch.Tensor,
        mask: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 0.0,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Standard Replace-method inpainting."""
        x = x_init.clone()
        
        # Fixed noise for temporal consistency
        noise_context = torch.randn_like(x_context)
        
        # Get timestep schedule
        timesteps = self._get_timestep_schedule()
        
        # Progress bar
        iterator = tqdm(timesteps, desc="Standard Inpainting", disable=not show_progress)
        
        model.eval()
        with torch.no_grad():
            for step_idx in iterator:
                # Get timestep tensor for model
                t_value = self._get_timestep_value(step_idx)
                t = torch.full((x.shape[0],), t_value, device=self.device, dtype=torch.long)
                
                # Predict noise (with optional CFG)
                noise_pred = self._predict_noise(
                    model, x, t, cond_input, uncond_input, guidance_scale
                )
                
                # Denoise step with inpainting
                x, x0_pred = self._denoise_step(
                    x, noise_pred, step_idx, x_context, mask, noise_context
                )
        
        return x


class RePaintSampler(InpaintingSamplerBase):
    """
    RePaint inpainting sampler with time-travel mechanism.
    
    Adapted from https://github.com/andreas128/RePaint
    
    Algorithm (from paper Section 4.2):
    1. Generate schedule with jumps: e.g., [249, 248, 249, 248, 247, ...]
    2. For each (t_last, t_cur) pair:
       - If t_cur < t_last: denoise step (reverse diffusion)
       - If t_cur > t_last: undo step (forward diffusion to jump back)
    3. At each denoise step, replace known region with noised ground truth
    
    The schedule creates regular jumps every `jump_length` timesteps,
    repeating `jump_n_sample` times at each jump point.
    
    Parameters:
        jump_length: How many timesteps to jump back (default: 10)
        jump_n_sample: How many times to resample at each jump point (default: 10)
        start_resampling: Only resample when t <= this value (default: inf, meaning always)
                         Note: This is different from our previous implementation
                         which counted from the end. Official counts from t=0.
    """
    
    def sample(
        self,
        model: Callable,
        x_init: torch.Tensor,
        x_context: torch.Tensor,
        mask: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 0.0,
        show_progress: bool = True
    ) -> torch.Tensor:

        x = x_init.clone()
        
        # Fixed noise for temporal consistency of context injection
        noise_context = torch.randn_like(x_context)
        
        # Generate the jump schedule (ported from official get_schedule_jump)
        # Returns list of timesteps like [249, 248, 249, 248, 247, ...]
        times = self._get_schedule_jump()
        
        # Create time pairs for iteration
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # Progress bar
        iterator = tqdm(
            time_pairs, 
            desc=f"RePaint (jump={self.config.jump_length}, n={self.config.jump_n_sample})",
            disable=not show_progress
        )
        
        pred_xstart = None
        
        model.eval()
        with torch.no_grad():
            for t_last, t_cur in iterator:
                t_last_tensor = torch.full(
                    (x.shape[0],), t_last, device=self.device, dtype=torch.long
                )
                
                if t_cur < t_last:
                    # Reverse step: denoise from x_t to x_{t-1}
                    # This is p_sample in official code
                    
                    # Inject known region with noised context BEFORE denoising
                    # (This is conf.inpa_inj_sched_prev in official code)
                    if pred_xstart is not None:
                        alpha_cumprod = self._get_alpha_bar(t_last)
                        x_context_noised = self._noise_context(
                            x_context, noise_context, alpha_cumprod
                        )
                        # mask=1 for inpaint (unknown), mask=0 for known
                        x = mask * x + (1 - mask) * x_context_noised
                    
                    # Predict noise
                    noise_pred = self._predict_noise(
                        model, x, t_last_tensor, cond_input, uncond_input, guidance_scale
                    )
                    
                    # Denoise step
                    x, pred_xstart = self._denoise_step(x, noise_pred, self._timestep_to_step_idx(t_last))
                    
                else:
                    # Forward step (undo): add noise to jump back in time
                    # This is the _undo function in official code
                    # Official formula: x_t = sqrt(1-beta) * x_{t-1} + sqrt(beta) * noise
                    x = self._undo(x, t_last_tensor)
        
        # Final composite with clean context
        x = mask * x + (1 - mask) * x_context
        return x
    
    def _get_schedule_jump(self) -> list:
        """
        Returns:
            List of timesteps like [t_T-1, t_T-2, t_T-1, t_T-2, t_T-3, ...]
            where jumps back occur at regular intervals.
        """
        # Get total timesteps (t_T in official code)
        t_T = self.num_steps
        
        # Parameters from config
        jump_length = self.config.jump_length
        jump_n_sample = self.config.jump_n_sample
        start_resampling = self.config.start_resampling
        
        # n_sample parameter (single-step resampling, usually 1)
        n_sample = 1
        
        # Build jump dictionary: at which timesteps to jump, and how many times
        jumps = {}
        for j in range(0, t_T - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1
        
        t = t_T
        ts = []
        
        while t >= 1:
            t = t - 1
            ts.append(t)
            
            # Single-step resampling (n_sample > 1 case, usually disabled)
            if t + 1 < t_T - 1 and t <= start_resampling:
                for _ in range(n_sample - 1):
                    t = t + 1
                    ts.append(t)
                    if t >= 0:
                        t = t - 1
                        ts.append(t)
            
            # Jump resampling
            if jumps.get(t, 0) > 0 and t <= start_resampling - jump_length:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    ts.append(t)
        
        ts.append(-1)  # Sentinel value
        
        return ts
    
    def _timestep_to_step_idx(self, timestep: int) -> int:
        """Convert timestep value to step index for scheduler."""
        if self.is_ddim:
            # Find the index in ddim_timesteps that matches this timestep
            # or the closest one
            timesteps_np = self.scheduler.ddim_timesteps.cpu().numpy()
            idx = (timesteps_np <= timestep).sum() - 1
            return max(0, min(idx, len(timesteps_np) - 1))
        else:
            return timestep
    
    def _undo(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Undo one denoising step (forward diffusion by one step).
        
        This is NOT the same as adding noise to timestep t using alpha_bar.
        Instead, it's a single-step forward diffusion:
            x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * noise
        
        Args:
            x: Image at timestep t-1 (less noisy)
            t: Target timestep t (we want to go back to this noisier state)
            
        Returns:
            Image at timestep t (more noisy)
        """
        # Get beta for this timestep
        if hasattr(self.scheduler, 'betas'):
            betas = self.scheduler.betas
        else:
            # Fallback: compute from alphas
            betas = 1 - self.scheduler.alphas
        
        # Extract beta for timestep t
        beta = betas[t.cpu()].to(x.device)
        while len(beta.shape) < len(x.shape):
            beta = beta[..., None]
        
        # Official formula: sqrt(1-beta) * x + sqrt(beta) * noise
        noise = torch.randn_like(x)
        x_noisy = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
        
        return x_noisy
    
    def _get_alpha_bar(self, t: int) -> torch.Tensor:
        """Get a_t (cumulative product of alphas) for timestep t."""
        if hasattr(self.scheduler, 'alpha_cum_prod'):
            return self.scheduler.alpha_cum_prod[t].to(self.device)
        elif hasattr(self.scheduler, 'alphas_cumprod'):
            return self.scheduler.alphas_cumprod[t].to(self.device)
        else:
            betas = self.scheduler.betas[:t+1].to(self.device)
            alphas = 1 - betas
            return torch.prod(alphas)
    
    def _noise_context(
        self, 
        x_context: torch.Tensor, 
        noise: torch.Tensor,
        alpha_bar_t: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to context for timestep t (VP formulation)."""
        sqrt_alpha = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_bar_t).sqrt()
        return sqrt_alpha * x_context + sqrt_one_minus_alpha * noise


class LanPaintSampler(InpaintingSamplerBase):
    """
    LanPaint inpainting sampler with Stochastic Harmonic Oscillator dynamics.
    
    Adapted from official LanPaint implementations:
    https://github.com/scraed/LanPaint (ComfyUI extension)
    https://github.com/scraed/LanPaintBench (Python benchmark)
    
    Key innovations per https://arxiv.org/abs/2502.03491:
    1. BiG Score: Bidirectional Guided score function
       - For unknown region (mask=1): standard score
       - For known region (mask=0): score_y = -(1+λ)(x_t - y) + λ·e_t
       
    2. Stochastic Harmonic Oscillator (underdamped Langevin with momentum)
       - Uses exact analytical integration with special functions
       - Multivariate normal sampling for correlated position-velocity
       
    3. Coefficient C Formula:
       C = (x̂₀ - x_t) / (1 - a_t) + A · x_t
       
    4. Leapfrog integration:
       - First step: full advance_time with initial C
       - Subsequent: half step → velocity kick → half step
    
    Parameters:
        lanpaint_num_steps: K - Inner Langevin iterations. Default: 5
        lanpaint_lambda: λ - BiG score alignment strength. Default: 16.0
        lanpaint_step_size: η - Langevin step size. Default: 0.2
        lanpaint_beta: β - Step size ratio (not used in current version). Default: 1.0
        lanpaint_friction: y - Friction for stability. Default: 15.0
        lanpaint_early_stop: Stop iterations early at final steps. Default: 1
        
    Note on mask convention:
        This implementation uses mask=1 for unknown (inpaint) region.
        Official LanPaint uses mask=1 for known region, so we swap internally.
    """
    
    def __init__(self, scheduler, config: InpaintingSamplerConfig, device: torch.device = None):
        super().__init__(scheduler, config, device)
        
        # LanPaint-specific state (reset each sample call)
        self._langevin_args = None  # (velocity, C) tuple for leapfrog
        
    def sample(
        self,
        model: Callable,
        x_init: torch.Tensor,
        x_context: torch.Tensor,
        mask: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 0.0,
        show_progress: bool = True
    ) -> torch.Tensor:
        """LanPaint inpainting with Stochastic Harmonic Oscillator dynamics."""
        
        # Fixed noise for context injection consistency
        noise_context = torch.randn_like(x_context)
        if torch.mean(torch.abs(noise_context)) < 1e-8:
            noise_context = torch.randn_like(noise_context)
        
        # Get timestep schedule
        timesteps = self._get_timestep_schedule()
        total_steps = len(timesteps)
        
        # Initialize x
        x = x_init.clone()
        
        # Progress bar
        K = self.config.lanpaint_num_steps
        iterator = tqdm(
            enumerate(timesteps),
            desc=f"LanPaint (K={K}, λ={self.config.lanpaint_lambda})",
            total=total_steps,
            disable=not show_progress
        )
        
        # Reset Langevin state
        self._langevin_args = None
        
        model.eval()
        for step_i, step_idx in iterator:
            t_value = self._get_timestep_value(step_idx)
            t = torch.full((x.shape[0],), t_value, device=self.device, dtype=torch.long)
            
            # Get noise schedule parameters
            alpha_bar_t = self._get_alpha_bar(t_value)
            sigma = ((1 - alpha_bar_t) / alpha_bar_t).sqrt()  # σ² = (1-α̅)/α̅
            
            # Replace known region with properly noised context
            x_context_noised = self._noise_context(x_context, noise_context, alpha_bar_t)
            # mask=1 means inpaint/generate, mask=0 means keep known
            x = mask * x + (1 - mask) * x_context_noised
            
            # LanPaint iterations (skip at final steps if early_stop > 0)
            steps_remaining = total_steps - step_i
            n_lanpaint_steps = K if steps_remaining > self.config.lanpaint_early_stop else 0
            
            if n_lanpaint_steps > 0 and t_value > 0:
                x, self._langevin_args = self._langevin_dynamics(
                    model, x, x_context, mask, t, sigma, alpha_bar_t,
                    cond_input, uncond_input, guidance_scale,
                    n_steps=n_lanpaint_steps,
                    args=self._langevin_args
                )
            
            # Standard denoising step to get x_{t-1}
            with torch.no_grad():
                noise_pred = self._predict_noise(
                    model, x, t, cond_input, uncond_input, guidance_scale
                )
                x, x0_pred = self._denoise_step(x, noise_pred, step_idx)
            
            # Composite for next step
            if step_i + 1 < total_steps:
                next_t_value = self._get_timestep_value(timesteps[step_i + 1])
                if next_t_value > 0:
                    next_alpha_bar = self._get_alpha_bar(next_t_value)
                    x_context_next = self._noise_context(x_context, noise_context, next_alpha_bar)
                    x = mask * x + (1 - mask) * x_context_next
        
        # Final output: composite with clean context
        x = mask * x + (1 - mask) * x_context
        return x
    
    def _get_alpha_bar(self, t: int) -> torch.Tensor:
        """Get a_t (cumulative product of alphas) for timestep t."""
        if hasattr(self.scheduler, 'alpha_cum_prod'):
            return self.scheduler.alpha_cum_prod[t].to(self.device)
        elif hasattr(self.scheduler, 'alphas_cumprod'):
            return self.scheduler.alphas_cumprod[t].to(self.device)
        else:
            betas = self.scheduler.betas[:t+1].to(self.device)
            alphas = 1 - betas
            return torch.prod(alphas)
    
    def _noise_context(
        self, 
        x_context: torch.Tensor, 
        noise: torch.Tensor,
        alpha_bar_t: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to context for timestep t (VP formulation)."""
        sqrt_alpha = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_bar_t).sqrt()
        return sqrt_alpha * x_context + sqrt_one_minus_alpha * noise
    
    def _prepare_step_size(
        self,
        sigma: torch.Tensor,
        abt: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare step sizes and parameters for Langevin dynamics.

        """
        step_size = self.config.lanpaint_step_size
        lamb = self.config.lanpaint_lambda
        friction = self.config.lanpaint_friction
        alpha = 0.0  # Official default for ULD
        
        # Step sizes for x (unknown) and y (known) branches
        sigma_x = (1 - abt + abt * alpha)  # For unknown region
        sigma_y = torch.zeros_like(sigma_x)  # Known region (forward diffused)
        
        dtx = step_size * sigma_x
        dty = step_size * sigma_y
        
        # Spring constants A
        A_x_T = 1.0
        A_y_T = 1.0 + lamb
        A_x = A_x_T / (1 - abt + abt * alpha + 1e-8)
        A_y = A_y_T / (1 - abt + 1e-8)
        
        # Friction (Gamma)
        Gamma_x = friction**2 * A_x
        Gamma_y = friction**2 * A_y
        
        # Diffusion coefficients
        D_x = (2 * (1 + sigma**2))**0.5
        D_y = (2 * (1 + sigma**2))**0.5
        
        return sigma, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y
    
    def _x0_evaluation(
        self,
        model: Callable,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        abt: torch.Tensor,
        t: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]],
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Evaluate x̂₀ from x_t using the model.

        """
        with torch.no_grad():
            # Scale input for model (official uses sigma scaling)
            x_scaled = x_t / (1 + sigma**2)**0.5
            
            # Get noise prediction
            eps = self._predict_noise(model, x_scaled, t, cond_input, uncond_input, guidance_scale)
            
            # x0 = x_t - sigma * eps
            x0 = x_t - sigma * eps
        
        return x0
    
    def _langevin_dynamics(
        self,
        model: Callable,
        x_t: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        abt: torch.Tensor,
        cond_input: Dict[str, Any],
        uncond_input: Optional[Dict[str, Any]],
        guidance_scale: float,
        n_steps: int,
        args: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Run Langevin dynamics with Stochastic Harmonic Oscillator.
        
        Adapted from official LanPaintBench:
        ULDCharaPipeline_Sampler.langevin_dynamics()
        
        Args:
            x_t: Current latent (in x-space)
            y: Context/known region values
            mask: mask=1 for unknown (inpaint), mask=0 for known
            t: Timestep tensor
            sigma: Noise level o
            abt: a_t
            args: (velocity, C) from previous iteration, or None
            
        Returns:
            (x_new, (v, C)) - updated latent and state for next iteration
        """
        # Swap mask: our convention is mask=1 for inpaint, official uses mask=1 for known
        # In official code: (1-mask) is applied to x branch, mask to y branch
        # So we need to invert: our mask=1 (unknown) -> official (1-mask)
        mask_x = mask      # Unknown region (inpaint)
        mask_y = 1 - mask  # Known region
        
        # Scale x_t to sigma-space (official formulation)
        x_t_scaled = x_t * (1 + sigma**2)**0.5
        
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            # Prepare step sizes
            step_sizes = self._prepare_step_size(sigma, abt, x_t)
            sigma_s, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = step_sizes
        
        # Check if step size is valid
        if torch.mean(dtx) <= 0.:
            return x_t, args
        
        # Combine parameters based on mask
        A = A_x * mask_x + A_y * mask_y
        D = D_x * mask_x + D_y * mask_y
        dt = dtx * mask_x + dty * mask_y
        Gamma = Gamma_x * mask_x + Gamma_y * mask_y
        
        # Define Coef_C computation (ported from official)
        def Coef_C(x_t_inner):
            x0 = self._x0_evaluation(
                model, x_t_inner, sigma, abt, t,
                cond_input, uncond_input, guidance_scale
            )
            # Official formula: C = (x0 - x_t) / (1-abt) + A * x_t
            C = (x0 - x_t_inner) / (1 - abt + 1e-8) + A * x_t_inner
            return C
        
        # Define advance_time using SHO
        def advance_time(x, v, dt_step, Gamma_step, A_step, C_step, D_step):
            dtype = x.dtype
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                osc = StochasticHarmonicOscillator(Gamma_step, A_step, C_step, D_step)
                x_new, v_new = osc.dynamics(x, v, dt_step)
            return x_new.to(dtype), v_new.to(dtype)
        
        x = x_t_scaled
        
        # Leapfrog integration (ported exactly from official)
        if args is None:
            # First iteration: initialize and do full step
            v = None
            C = Coef_C(x)
            x, v = advance_time(x, v, dt, Gamma, A, C, D)
        else:
            v, C = args
            
            # Half step with old C
            x, v = advance_time(x, v, dt/2, Gamma, A, C, D)
            
            # Recompute C at new position
            C_new = Coef_C(x)
            
            # Velocity kick (key leapfrog step)
            v = v + Gamma**0.5 * (C_new - C) * dt
            
            # Half step with new C
            x, v = advance_time(x, v, dt/2, Gamma, A, C_new, D)
            
            C = C_new
        
        # Scale back to x-space
        x_out = x / (1 + sigma**2)**0.5
        
        return x_out, (v, C)


def get_inpainting_sampler(
    sampler_type: str,
    scheduler,
    config: Optional[Dict[str, Any]] = None,
    device: torch.device = None
) -> InpaintingSamplerBase:
    """
    Factory function to create inpainting samplers.
    
    Args:
        sampler_type: 'standard', 'repaint', or 'lanpaint'
        scheduler: Base noise scheduler (DDPM or DDIM)
        config: Optional configuration dictionary
        device: Computation device
        
    Returns:
        InpaintingSamplerBase instance
        
    Example:
        >>> config = {'type': 'repaint', 'jump_length': 10, 'jump_n_sample': 10}
        >>> sampler = get_inpainting_sampler('repaint', scheduler, config)
        >>> result = sampler.sample(model, x_init, x_context, mask, cond_input)
    """
    config = config or {}
    sampler_config = InpaintingSamplerConfig.from_config(config)
    
    samplers = {
        'standard': StandardInpaintingSampler,
        'repaint': RePaintSampler,
        'lanpaint': LanPaintSampler,
    }
    
    if sampler_type not in samplers:
        raise ValueError(
            f"Unknown inpainting sampler: {sampler_type}. "
            f"Available: {list(samplers.keys())}"
        )
    
    return samplers[sampler_type](scheduler, sampler_config, device)

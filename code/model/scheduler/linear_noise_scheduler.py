# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/scheduler/linear_noise_scheduler.py
import torch
import numpy as np


class LinearNoiseScheduler:
    r"""
    Noise scheduler for DDPM with multiple beta schedule options and optional Velocity schedule.
    
    Supports:
    - 'linear' (scaled-linear): Original DDPM schedule with sqrt scaling
    - 'cosine': Improved DDPM schedule (from "Improved Denoising Diffusion Probabilistic Models")
    - 'laplace': Laplace noise schedule (from "Improved Noise Schedule for Diffusion Training")
    
    Cosine schedule advantages:
    - More gradual noise addition at early timesteps
    - Better preservation of signal in mid-range timesteps
    
    Laplace schedule advantages:
    - Concentrates training compute around log(SNR)=0 (medium noise levels)
    - Faster convergence than cosine + min-SNR loss weighting
    
    References:
        Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
        arXiv:2102.09672. https://arxiv.org/abs/2102.09672
        
        Hang, T., et al. (2024). Improved Noise Schedule for Diffusion Training.
        arXiv:2407.03297. https://arxiv.org/abs/2407.03297
        
        Velocity prediction:
            Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of 
            Diffusion Models. ICLR 2022. arXiv:2202.00512.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end, beta_schedule='linear', 
                 clamp_range=(-10.0, 10.0), laplace_mu=0.0, laplace_b=0.5):
        """
        Args:
            num_timesteps: Total diffusion timesteps (typically 1000)
            beta_start: Starting beta value (e.g., 0.0001) - used for linear schedule
            beta_end: Ending beta value (e.g., 0.02) - used for linear schedule
            beta_schedule: Schedule type - 'linear', 'cosine', or 'laplace'
            clamp_range: (min, max) range for clamping x0 predictions during sampling
                        Prevents numerical instability while preserving latent detail
            laplace_mu: Location parameter for Laplace schedule (default: 0.0, center at log(SNR)=0)
            laplace_b: Scale parameter for Laplace schedule (default: 0.5 for 256px, use 0.75 for 512px)
                      Smaller b = sharper peak = more compute at medium noise
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clamp_min, self.clamp_max = clamp_range
        self.laplace_mu = laplace_mu
        self.laplace_b = laplace_b
        
        # Compute schedule based on type
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
            # Precompute alpha values from betas
            self.alphas = 1. - self.betas
            self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        elif beta_schedule == 'linear':
            # Scaled-linear schedule (original DDPM)
            self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
            )
            # Precompute alpha values from betas
            self.alphas = 1. - self.betas
            self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        elif beta_schedule == 'laplace':
            # Laplace schedule: directly computes alpha_cum_prod from log(SNR)
            self.alpha_cum_prod = self._laplace_schedule(num_timesteps, laplace_mu, laplace_b)
            # Derive betas and alphas from alpha_cum_prod for compatibility
            self.alphas = torch.ones_like(self.alpha_cum_prod)
            self.alphas[1:] = self.alpha_cum_prod[1:] / self.alpha_cum_prod[:-1]
            self.alphas[0] = self.alpha_cum_prod[0]
            self.betas = 1. - self.alphas
            # Clip betas for numerical stability
            self.betas = torch.clamp(self.betas, min=1e-8, max=0.999)
        else:
            raise ValueError(
                f"Unknown beta_schedule: '{beta_schedule}'. "
                f"Supported: 'linear', 'cosine', 'laplace'"
            )
        
        # Precompute sqrt values (used by all methods)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def get_velocity(self, x0, noise, t):
        """
        Compute velocity (v-prediction target) from x0 and noise.
        
        v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0
        
        This is the target for v-prediction training. V-prediction provides more balanced
        gradients across timesteps compared to epsilon prediction.
        
        Args:
            x0: Clean latent [B, C, H, W]
            noise: Sampled noise ε [B, C, H, W]
            t: Timestep indices [B]
            
        Returns:
            Velocity v [B, C, H, W]
            
        Reference:
            Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of 
            Diffusion Models. ICLR 2022. arXiv:2202.00512.
            https://arxiv.org/abs/2202.00512
        """
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(x0.device)[t]
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(x0.device)[t]
        
        # Reshape to [B, 1, 1, 1] for broadcasting
        for _ in range(len(x0.shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0
        v = sqrt_alpha_cum_prod * noise - sqrt_one_minus_alpha_cum_prod * x0
        return v
    
    def velocity_to_epsilon(self, v_pred, xt, t):
        """
        Convert velocity prediction to noise prediction adapted from Salimans & Ho (2022).
        
        From v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0 and x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,
        we can solve for ε:
        ε = √ᾱ_t · v + √(1-ᾱ_t) · x_t
        
        This conversion is needed during sampling when using v-prediction mode,
        as the DDPM/DDIM sampling formulas expect epsilon (noise) predictions.
        
        Args:
            v_pred: Predicted velocity [B, C, H, W]
            xt: Noisy latent at timestep t [B, C, H, W]
            t: Timestep index
            
        Returns:
            Predicted noise ε [B, C, H, W]
            
        Reference:
            Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of 
            Diffusion Models. ICLR 2022. arXiv:2202.00512.
            https://arxiv.org/abs/2202.00512
        """
        sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod.to(xt.device)[t])
        sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod.to(xt.device)[t])
        
        # ε = √ᾱ_t · v + √(1-ᾱ_t) · x_t
        epsilon = sqrt_alpha_cum_prod * v_pred + sqrt_one_minus_alpha_cum_prod * xt
        return epsilon
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in Nichol & Dhariwal (2021).
        
        Creates a more gradual noise schedule that:
        - Preserves more signal at early timesteps
        - Avoids too much noise at final timesteps
        - Results in more balanced SNR across timesteps
        
        Args:
            timesteps: Number of diffusion steps
            s: Offset parameter to prevent beta from being too small (default: 0.008)
            
        Returns:
            Tensor of beta values [timesteps]
            
        Reference:
            Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
            arXiv:2102.09672. https://arxiv.org/abs/2102.09672
        """
        # Compute alpha_bar values using cosine schedule
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        
        # Cosine schedule: alpha_bar(t) = cos((t/T + s)/(1+s) * pi/2)^2
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
        
        # Compute betas from alpha_bar values
        # beta_t = 1 - alpha_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip to reasonable range (prevent numerical issues)
        # Cosine schedule naturally produces smaller betas than linear
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _laplace_schedule(self, timesteps, mu=0.0, b=0.5):
        """
        Laplace noise schedule adapted from Hang et al. (2024).
        
        Key insight from Hang et al. (2024): By defining log(SNR) via the Laplace distribution inverse CDF,
        sampling timesteps uniformly naturally concentrates more training compute
        around log(SNR)=0 (medium noise levels). This is more effective than 
        adjusting loss weights alone.
        
        The schedule defines:
            λ(t) = μ - b · sign(0.5-t) · log(1 - 2|0.5-t|)
        
        where λ = log(SNR), and then:
            SNR = exp(λ)
            a_bar = SNR / (1 + SNR)
        
        Args:
            timesteps: Number of diffusion steps
            mu: Location parameter (default: 0.0 = center distribution at log(SNR)=0)
            b: Scale parameter (default: 0.5 for ~256px images)
               - Smaller b = sharper peak = more concentrated around medium noise
               - Use b=0.5 for 256×256, b=0.75 for 512×512
               
        Returns:
            Tensor of alpha_cum_prod values [timesteps]
            
        Reference:
            Hang, T., et al. (2024). Improved Noise Schedule for Diffusion Training.
            arXiv:2407.03297. https://arxiv.org/abs/2407.03297
        """
        # Map discrete timesteps to continuous t ∈ (0, 1)
        # t=0 → high SNR (low noise), t=1 → low SNR (high noise)
        # Use small epsilon to avoid log(0) at boundaries
        eps = 1e-4
        t = torch.linspace(eps, 1 - eps, timesteps)
        
        # Compute log(SNR) using Laplace distribution inverse CDF
        # λ(t) = μ - b · sign(0.5-t) · log(1 - 2|0.5-t|)
        t_shifted = 0.5 - t
        sign_t = torch.sign(t_shifted)
        abs_t = torch.abs(t_shifted)
        
        # Clamp argument to avoid log(0) near t=0.5
        log_arg = torch.clamp(1 - 2 * abs_t, min=1e-8)
        log_snr = mu - b * sign_t * torch.log(log_arg)
        
        # Convert log(SNR) to SNR
        snr = torch.exp(log_snr)
        
        # Compute alpha_bar = SNR / (1 + SNR) (variance-preserving formulation)
        # At t=0: high SNR → alpha_bar ≈ 1 (low noise)
        # At t=1: low SNR → alpha_bar ≈ 0 (high noise)
        alphas_cumprod = snr / (1 + snr)
        
        # Ensure values are in valid range and monotonically decreasing
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1 - 1e-8)
        
        # Enforce monotonic decrease (alpha_bar should decrease as t increases)
        # This handles any numerical edge cases
        for i in range(1, len(alphas_cumprod)):
            if alphas_cumprod[i] >= alphas_cumprod[i-1]:
                alphas_cumprod[i] = alphas_cumprod[i-1] - 1e-8
        
        return alphas_cumprod
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        # Clamp x0 prediction for stability (configurable for different latent ranges)
        x0 = torch.clamp(x0, self.clamp_min, self.clamp_max)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
    
    def sample_prev_timestep_inpainting(self, xt, noise_pred, t, x_context, mask, noise_context=None):
        r"""
        Sample previous timestep for inpainting.
        Clamps known regions to context after each denoising step using FIXED noise.
        
        noise_context must be sampled ONCE per sample and reused for all timesteps
        to prevent temporal inconsistency and seam artifacts.
        
        :param xt: current timestep sample [B, C, H, W]
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :param x_context: known context latent (masked region) [B, C, H, W]
        :param mask: inpainting mask [B, 1, H, W], 1=regenerate, 0=keep
        :param noise_context: FIXED noise tensor [B, C, H, W] sampled once per sample (required for t>0)
        :return: (xt-1, x0_pred)
        """
        # Standard denoising step
        xt_minus_1, x0 = self.sample_prev_timestep(xt, noise_pred, t)
        
        # Clamp known region to properly noised context distribution
        if t > 0:
            if noise_context is None:
                raise ValueError(
                    "noise_context is required for inpainting when t > 0. "
                    "Sample it ONCE per generation: noise_context = torch.randn_like(x_context)"
                )
            
            # Clamp outside region to q(x_{t-1} | x0_context) using fixed noise_context
            t_context = t - 1
            t_batch = torch.full((x_context.shape[0],), t_context, device=x_context.device, dtype=torch.long)
            x_context_noisy = self.add_noise(x_context, noise_context, t_batch)
            
            # Blend: regenerate masked region, keep (noisy) context elsewhere
            xt_minus_1 = mask * xt_minus_1 + (1 - mask) * x_context_noisy
        else:
            # Final step: use clean context
            xt_minus_1 = mask * xt_minus_1 + (1 - mask) * x_context
        
        return xt_minus_1, x0
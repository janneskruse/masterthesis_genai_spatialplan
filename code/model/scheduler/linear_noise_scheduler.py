# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/scheduler/linear_noise_scheduler.py
import torch
import numpy as np


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
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
        # Widen clamp range to avoid artifacts (latents may not be in [-1,1])
        x0 = torch.clamp(x0, -3., 3.)
        
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
        
        CRITICAL: noise_context must be sampled ONCE per sample and reused for all timesteps
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
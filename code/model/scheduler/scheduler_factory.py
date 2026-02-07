"""
==============================================================================
Scheduler Factory for Diffusion Sampling that centralizes scheduler creation.

Supports:
- DDPM (LinearNoiseScheduler): Standard diffusion, 1000 steps
- DDIM (DDIMScheduler): Fast deterministic sampling, 50 steps (20x faster)
- Inpainting Samplers: Standard, RePaint, LanPaint for boundary harmonization
==
"""

###### import libraries ######
from typing import Optional, Dict, Any
import torch

from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.scheduler.ddim_scheduler import DDIMScheduler
from model.scheduler.inpainting_samplers import get_inpainting_sampler


def get_scheduler(diffusion_config: dict):
    """
    Create scheduler from diffusion configuration.
    
    Args:
        diffusion_config: Diffusion parameters from config dict
                         Required keys:
                         - num_timesteps (int): Total diffusion steps (usually 1000)
                         - beta_start (float): Starting beta value (0.0001)
                         - beta_end (float): Ending beta value (0.02)
                         Optional keys:
                         - beta_schedule (str): 'linear' | 'cosine' (default: 'linear')
                         - sampler (str): 'ddpm' | 'ddim' (default: 'ddpm')
                         - ddim_steps (int): DDIM sampling steps (default: 50)
                         - ddim_eta (float): DDIM stochasticity (default: 0.0)
        
    Returns:
        Scheduler instance (LinearNoiseScheduler or DDIMScheduler)
        
    Raises:
        ValueError: If sampler type is unknown
        
    Example:
        >>> config = {
        ...     'num_timesteps': 1000,
        ...     'beta_start': 0.0001,
        ...     'beta_end': 0.02,
        ...     'beta_schedule': 'cosine',
        ...     'sampler': 'ddim',
        ...     'ddim_steps': 50,
        ...     'ddim_eta': 0.0
        ... }
        >>> scheduler = get_scheduler(config)
        >>> # Use for training or sampling
    """
    sampler_type = diffusion_config.get('sampler', 'ddpm')
    beta_schedule = diffusion_config.get('beta_schedule', 'linear')
    
    if sampler_type == 'ddim':
        # DDIM: Fast deterministic sampling
        clamp_range = diffusion_config.get('clamp_range', [-10.0, 10.0])  # Default for semantic
        scheduler = DDIMScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            beta_schedule=beta_schedule,
            ddim_steps=diffusion_config.get('ddim_steps', 50),
            ddim_eta=diffusion_config.get('ddim_eta', 0.0),
            clamp_range=tuple(clamp_range)
        )
        return scheduler
        
    elif sampler_type == 'ddpm':
        # DDPM: Standard diffusion sampling
        scheduler = LinearNoiseScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            beta_schedule=beta_schedule
        )
        return scheduler
        
    else:
        raise ValueError(
            f"Unknown sampler: '{sampler_type}'. "
            f"Supported samplers: 'ddpm', 'ddim'. "
            f"Check config['diffusion_params']['sampler']."
        )




def get_inpainting_sampler_for_stage(
    config: Dict[str, Any],
    stage_name: str,
    device: Optional[torch.device] = None
):
    """
    Create inpainting sampler for a specific diffusion stage.
    
    Args:
        config: Full model config dict
        stage_name: Name of the diffusion stage ('semantic' or 'satellite')
        device: Computation device (default: cuda if available)
        
    Returns:
        InpaintingSamplerBase instance (StandardInpaintingSampler, RePaintSampler, or LanPaintSampler)
        
    Example:
        >>> config = yaml.safe_load(open('two_stage_8.yml'))
        >>> # Get sampler for semantic stage
        >>> semantic_sampler = get_inpainting_sampler_for_stage(config, 'semantic')
        >>> # Get sampler for satellite stage
        >>> satellite_sampler = get_inpainting_sampler_for_stage(config, 'satellite')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get base scheduler (DDPM or DDIM)
    scheduler = get_scheduler(config['diffusion_params'])
    
    # Get stage config
    diffusion_stages = config.get('diffusion_stages', {})
    if stage_name not in diffusion_stages:
        raise ValueError(
            f"Stage '{stage_name}' not found in diffusion_stages. "
            f"Available: {list(diffusion_stages.keys())}"
        )
    
    stage_config = diffusion_stages[stage_name]
    inpainting_cfg = stage_config.get('inpainting', {})
    sampler_config = inpainting_cfg.get('sampler', {'type': 'standard'})
    
    return _build_inpainting_sampler(scheduler, sampler_config, device)


def _build_inpainting_sampler(
    scheduler,
    sampler_config: Dict[str, Any],
    device: torch.device
):
    """
    Internal helper to build inpainting sampler from config dict.
    
    Args:
        scheduler: Base noise scheduler (DDPM or DDIM)
        sampler_config: Sampler configuration dict with 'type' and type-specific params
        device: Computation device
        
    Returns:
        InpaintingSamplerBase instance
    """
    sampler_type = sampler_config.get('type', 'standard')
    
    # Build config dict for sampler
    if sampler_type == 'repaint':
        repaint_config = sampler_config.get('repaint', {})
        flat_config = {
            'type': 'repaint',
            'jump_length': repaint_config.get('jump_length', 10),
            'jump_n_sample': repaint_config.get('jump_n_sample', 10),
            'start_resampling': repaint_config.get('start_resampling', 100000000),
        }
    elif sampler_type == 'lanpaint':
        lanpaint_config = sampler_config.get('lanpaint', {})
        flat_config = {
            'type': 'lanpaint',
            'lanpaint_num_steps': lanpaint_config.get('num_steps', 5),
            'lanpaint_lambda': lanpaint_config.get('lambda', 16.0),
            'lanpaint_step_size': lanpaint_config.get('step_size', 0.2),
            'lanpaint_beta': lanpaint_config.get('beta', 1.0),
            'lanpaint_friction': lanpaint_config.get('friction', 15.0),
            'lanpaint_early_stop': lanpaint_config.get('early_stop', 1),
        }
    else:
        # Standard sampler
        flat_config = {'type': 'standard'}
    
    # Create inpainting sampler
    return get_inpainting_sampler(
        sampler_type=sampler_type,
        scheduler=scheduler,
        config=flat_config,
        device=device
    )

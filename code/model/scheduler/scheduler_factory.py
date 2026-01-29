"""
Scheduler Factory for Diffusion Sampling.

Centralizes scheduler creation logic to eliminate code duplication
between training and sampling scripts.

Supports:
- DDPM (LinearNoiseScheduler): Standard diffusion, 1000 steps
- DDIM (DDIMScheduler): Fast deterministic sampling, 50 steps (20x faster)

Future extensions (easy to add):
- DPM-Solver++: Even faster than DDIM
- UniPC: Unified predictor-corrector
- Euler-A: Ancestral sampling
"""

###### import libraries ######
from model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from model.scheduler.ddim_scheduler import DDIMScheduler


def get_scheduler(diffusion_config: dict):
    """
    Create scheduler from diffusion configuration.
    
    Single source of truth for scheduler creation - ensures training
    and sampling use identical scheduler configuration.
    
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
        scheduler = DDIMScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            beta_schedule=beta_schedule,
            ddim_steps=diffusion_config.get('ddim_steps', 50),
            ddim_eta=diffusion_config.get('ddim_eta', 0.0)
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

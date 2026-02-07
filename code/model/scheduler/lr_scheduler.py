"""
==============================================================================
Learning Rate Scheduler with Warmup Support.

Provides flexible LR scheduling for diffusion model training:
- Cosine annealing for smooth decay
- Linear decay for simple schedules
- Constant LR with optional warmup
==============================================================================
"""

###### import libraries ######
# Standard libraries
from typing import Optional, Dict, Any

# Data Science/ML
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
    SequentialLR,
    LambdaLR
)


def get_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler with optional warmup.
    
    Supports three main schedules:
    1. 'cosine': Cosine annealing - smooth decay from initial LR to 0
       Best for: Long training runs, stable convergence
       
    2. 'linear': Linear decay - gradual linear decrease to 0
       Best for: Fine-tuning, shorter training
       
    3. 'constant': Fixed LR with optional warmup
       Best for: Small datasets, when LR is already optimized
    
    All schedules support warmup: LR linearly increases from 0 to target
    over first N steps to prevent early training instability.
    
    Args:
        optimizer: PyTorch optimizer (Adam, AdamW, etc.)
        config: Dictionary with keys:
            - lr_scheduler: str, one of ['cosine', 'linear', 'constant']
            - lr_warmup_steps: int, number of warmup steps (0 = no warmup)
            - num_epochs: int, total training epochs
            - steps_per_epoch: int, batches per epoch
            
    Returns:
        LR scheduler or None if lr_scheduler='constant' and no warmup
        
    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.0001)
        >>> config = {
        >>>     'lr_scheduler': 'cosine',
        >>>     'lr_warmup_steps': 1000,
        >>>     'num_epochs': 300,
        >>>     'steps_per_epoch': 500
        >>> }
        >>> scheduler = get_lr_scheduler(optimizer, config)
        >>> 
        >>> # Training loop
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         optimizer.step()
        >>>         scheduler.step()  # Call after each batch
    """
    # Extract config
    scheduler_type = config.get('lr_scheduler', 'constant').lower()
    warmup_steps = config.get('lr_warmup_steps', 0)
    num_epochs = config.get('num_epochs', 100)
    steps_per_epoch = config.get('steps_per_epoch', 100)
    
    total_steps = num_epochs * steps_per_epoch
    
    # Validate config
    if scheduler_type not in ['cosine', 'linear', 'constant']:
        raise ValueError(
            f"Invalid lr_scheduler: '{scheduler_type}'. "
            f"Must be one of: 'cosine', 'linear', 'constant'"
        )
    
    if warmup_steps < 0:
        raise ValueError(f"lr_warmup_steps must be >= 0, got {warmup_steps}")
    
    if warmup_steps >= total_steps:
        print(f"⚠ Warning: warmup_steps ({warmup_steps}) >= total_steps ({total_steps})")
        print(f"  Setting warmup to 10% of total steps")
        warmup_steps = int(0.1 * total_steps)
    
    # Case 1: Constant LR with no warmup
    if scheduler_type == 'constant' and warmup_steps == 0:
        print(f"✓ LR Scheduler: constant (no warmup)")
        return None  # No scheduler needed
    
    # Case 2: Constant LR with warmup only
    if scheduler_type == 'constant' and warmup_steps > 0:
        print(f"✓ LR Scheduler: constant with {warmup_steps} warmup steps")
        
        # Warmup: 0 → 1.0 over warmup_steps
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-10,  # Start from near-zero
            end_factor=1.0,      # End at full LR
            total_iters=warmup_steps
        )
        
        # After warmup: constant at 1.0
        constant_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps - warmup_steps
        )
        
        # Combine: warmup → constant
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    # Case 3: Cosine annealing with optional warmup
    if scheduler_type == 'cosine':
        main_steps = total_steps - warmup_steps
        
        if warmup_steps > 0:
            print(f"✓ LR Scheduler: cosine annealing with {warmup_steps} warmup steps")
            print(f"  Warmup: 0 → 1.0 over {warmup_steps} steps")
            print(f"  Cosine: 1.0 → 0 over {main_steps} steps (T_max={main_steps})")
            
            # Warmup phase
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-10,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Cosine annealing phase
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=0  # Decay to 0
            )
            
            # Combine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            print(f"✓ LR Scheduler: cosine annealing (no warmup)")
            print(f"  Cosine: 1.0 → 0 over {total_steps} steps")
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=0
            )
        
        return scheduler
    
    # Case 4: Linear decay with optional warmup
    if scheduler_type == 'linear':
        main_steps = total_steps - warmup_steps
        
        if warmup_steps > 0:
            print(f"✓ LR Scheduler: linear decay with {warmup_steps} warmup steps")
            print(f"  Warmup: 0 → 1.0 over {warmup_steps} steps")
            print(f"  Linear: 1.0 → 0 over {main_steps} steps")
            
            # Warmup phase
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-10,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Linear decay phase
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=main_steps
            )
            
            # Combine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
        else:
            print(f"✓ LR Scheduler: linear decay (no warmup)")
            print(f"  Linear: 1.0 → 0 over {total_steps} steps")
            
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps
            )
        
        return scheduler


def get_current_lr(optimizer: Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    """
    return optimizer.param_groups[0]['lr']


def plot_lr_schedule(
    scheduler_type: str,
    initial_lr: float,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_steps: int = 0
) -> None:
    """
    Plot learning rate schedule for visualization.
    
    Args:
        scheduler_type: 'cosine', 'linear', or 'constant'
        initial_lr: Starting learning rate
        num_epochs: Total training epochs
        steps_per_epoch: Steps per epoch
        warmup_steps: Warmup steps
        
    Example:
        >>> plot_lr_schedule('cosine', 0.0001, 300, 500, 1000)
        >>> # Shows cosine curve with warmup
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")
        return
    
    # Create dummy optimizer and scheduler
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=initial_lr)
    
    config = {
        'lr_scheduler': scheduler_type,
        'lr_warmup_steps': warmup_steps,
        'num_epochs': num_epochs,
        'steps_per_epoch': steps_per_epoch
    }
    
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Record LR at each step
    total_steps = num_epochs * steps_per_epoch
    lrs = []
    
    for step in range(total_steps):
        lrs.append(get_current_lr(optimizer))
        if scheduler is not None:
            scheduler.step()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title(f'LR Schedule: {scheduler_type} (warmup={warmup_steps})')
    plt.grid(True, alpha=0.3)
    
    # Mark warmup region
    if warmup_steps > 0:
        plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5, label='End of warmup')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

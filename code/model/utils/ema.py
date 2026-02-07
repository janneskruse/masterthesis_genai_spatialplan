"""
==============================================================================
Exponential Moving Average (EMA) for model weights.

Maintains a slow-moving average of model parameters for stable, high-quality inference.
Critical for diffusion models to reduce sampling artifacts and improve consistency.
==============================================================================
"""

###### import libraries ######
# Standard libraries
from typing import Optional
from copy import deepcopy

# Data Science/ML
import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Exponential Moving Average wrapper for PyTorch models.
    
    Maintains shadow parameters that update slowly via:
        ema_param = decay * ema_param + (1 - decay) * model_param
    
    Usage:
        >>> model = MyModel()
        >>> ema = ExponentialMovingAverage(model, decay=0.9999)
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        >>>     loss = train_step(model, batch)
        >>>     optimizer.step()
        >>>     ema.update(model)  # Update EMA after each step
        >>> 
        >>> # Inference with EMA weights
        >>> ema.store(model)      # Temporarily save training weights
        >>> ema.copy_to(model)    # Copy EMA weights to model
        >>> samples = model.generate()
        >>> ema.restore(model)    # Restore training weights
    
    Args:
        model: PyTorch model to track
        decay: EMA decay rate (0.9999 is standard for diffusion models)
               Higher = slower updates, more smoothing
        device: Device to store shadow parameters on
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA with model's current parameters.
        
        Args:
            model: Model to track
            decay: EMA decay rate (0.9999 recommended)
            device: Device for shadow params (None = same as model)
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow parameters (deep copy of model parameters)
        self.shadow_params = {}
        self.collected_params = {}  # For temporary storage during inference
        
        # Initialize shadow with current model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)
        
        print(f"✓ EMA initialized with decay={decay:.5f}, tracking {len(self.shadow_params)} parameters")
    
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with current model parameters.
        
        Call after each optimizer step during training.
        
        Args:
            model: Model with updated parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in self.shadow_params:
                        # EMA update: shadow = decay * shadow + (1 - decay) * current
                        self.shadow_params[name].mul_(self.decay).add_(
                            param.data.to(self.device),
                            alpha=1.0 - self.decay
                        )
                    else:
                        # New parameter appeared (e.g., model architecture changed)
                        self.shadow_params[name] = param.data.clone().to(self.device)
    
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA shadow parameters to model.
        
        Use before inference to get stable, high-quality predictions.
        Remember to call store() first to save training weights.
        
        Args:
            model: Model to copy EMA weights into
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name].to(param.device))
    
    def store(self, model: nn.Module) -> None:
        """
        Temporarily store current model parameters.
        
        Call before copy_to() to save training weights for later restoration.
        
        Args:
            model: Model whose weights to store
        """
        self.collected_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collected_params[name] = param.data.clone()
    
    def restore(self, model: nn.Module) -> None:
        """
        Restore model parameters from temporary storage.
        
        Call after inference to resume training with original weights.
        
        Args:
            model: Model to restore weights into
        """
        if not self.collected_params:
            raise RuntimeError(
                "No stored parameters to restore. Call store() before restore()."
            )
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.collected_params:
                    param.data.copy_(self.collected_params[name])
        
        # Clear stored params to prevent accidental reuse
        self.collected_params = {}
    
    def state_dict(self) -> dict:
        """
        Get EMA state dictionary for checkpointing.
        
        Returns:
            Dict containing shadow parameters and decay rate
        """
        return {
            'decay': self.decay,
            'shadow_params': {name: param.cpu() for name, param in self.shadow_params.items()}
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load EMA state from checkpoint.
        
        Args:
            state_dict: Dict from state_dict() method
        """
        self.decay = state_dict['decay']
        self.shadow_params = {
            name: param.to(self.device)
            for name, param in state_dict['shadow_params'].items()
        }
        print(f"✓ EMA loaded from checkpoint: decay={self.decay:.5f}, {len(self.shadow_params)} parameters")
    
    def to(self, device: torch.device) -> 'ExponentialMovingAverage':
        """
        Move EMA shadow parameters to different device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.shadow_params = {
            name: param.to(device)
            for name, param in self.shadow_params.items()
        }
        return self


def apply_ema_to_model(model: nn.Module, ema: ExponentialMovingAverage) -> None:
    """
    Context-safe wrapper to apply EMA weights for inference.
    
    Usage:
        >>> with torch.no_grad():
        >>>     ema.store(model)
        >>>     ema.copy_to(model)
        >>>     samples = model.generate()
        >>>     ema.restore(model)
    
    Or use as standalone:
        >>> apply_ema_to_model(model, ema)
        >>> samples = model.generate()
        >>> # (EMA weights remain in model after this call)
    
    Args:
        model: Model to apply EMA weights to
        ema: EMA instance
    """
    ema.copy_to(model)

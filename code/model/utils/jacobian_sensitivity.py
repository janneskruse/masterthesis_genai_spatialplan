"""
==============================================================================
Jacobian Sensitivity Analysis for VAE Latent Space
Computes how each latent channel contributes to each decoded layer
Used for class-balanced loss weighting in latent diffusion training

Inspired by Jacobian-based relevance scoring from:
    Saha, S., Joshi, S., & Whitaker, R. (2025). ARD-VAE: A Statistical Formulation 
    to Find the Relevant Latent Dimensions of Variational Autoencoders. 
    IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025.
    arXiv:2501.10901v2. https://arxiv.org/abs/2501.10901
=======================================================================
"""

###### import libraries ######
# Standard libraries
import os
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

# Data Science/ML libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


def compute_jacobian_sensitivity(
    vae_or_decoder: Union[nn.Module, callable],
    z: torch.Tensor,
    layer_channel_ranges: Dict[str, Tuple[int, int]],
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobian-based sensitivity matrix for a single latent sample.
    
    Measures how much each latent channel contributes to each decoded layer
    using ||∂decoded_layer/∂latent_channel||_F (Frobenius norm).
    
    Args:
        vae_or_decoder: Either full VAE model or decoder function
        z: Latent tensor [1, C_latent, H', W'] - single sample
        layer_channel_ranges: Dict mapping layer names to (start_ch, end_ch) in decoded output
            e.g., {'buildings': (0, 1), 'streets': (1, 2), 'vegetation': (2, 3)}
        device: Device for computation (defaults to z.device)
        
    Returns:
        S: Sensitivity matrix [n_layers, n_latent] - how much each latent affects each layer
        z_stats: Statistics tensor [n_latent * 4] - [means, stds, mins, maxs] per channel
    """
    if device is None:
        device = z.device
    
    # Handle both VAE model and decoder function
    if isinstance(vae_or_decoder, nn.Module):
        # It's a full model - set to eval and get decode method
        vae_or_decoder.eval()
        if hasattr(vae_or_decoder, 'module'):
            decode_fn = vae_or_decoder.module.decode
        elif hasattr(vae_or_decoder, 'decode'):
            decode_fn = vae_or_decoder.decode
        else:
            # Assume it's the decoder itself
            decode_fn = vae_or_decoder
    else:
        # It's already a function
        decode_fn = vae_or_decoder
    
    n_latent = z.shape[1]  # Number of latent channels
    n_layers = len(layer_channel_ranges)
    layer_names = list(layer_channel_ranges.keys())
    
    # Ensure z requires grad for Jacobian computation
    z = z.clone().detach().requires_grad_(True)
    
    # Forward pass through decoder
    decoded = decode_fn(z)  # [1, C_decoded, H, W]
    
    # Initialize sensitivity matrix
    S = torch.zeros(n_layers, n_latent, device=device)
    
    # Compute sensitivity per (layer, latent_channel) pair
    for layer_idx, layer_name in enumerate(layer_names):
        start_ch, end_ch = layer_channel_ranges[layer_name]
        
        for latent_ch in range(n_latent):
            # Create grad_output mask: 1 for the layer channels, 0 elsewhere
            grad_output = torch.zeros_like(decoded)
            grad_output[:, start_ch:end_ch, :, :] = 1.0
            
            # Compute gradient of decoded output w.r.t. latent z
            # This gives us ∂decoded/∂z for all spatial positions
            if z.grad is not None:
                z.grad.zero_()
            
            # Retain graph for multiple backward passes
            grads = torch.autograd.grad(
                outputs=decoded,
                inputs=z,
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=False,
                allow_unused=False
            )[0]  # [1, C_latent, H', W']
            
            # Extract gradient for this latent channel
            grad_latent_ch = grads[:, latent_ch, :, :]  # [1, H', W']
            
            # Compute Frobenius norm as sensitivity measure
            # ||∂decoded_layer/∂latent_channel||_F
            sensitivity = torch.norm(grad_latent_ch, p='fro')
            S[layer_idx, latent_ch] = sensitivity
    
    # Compute z statistics for polynomial predictor
    # z: [1, C_latent, H', W']
    z_detached = z.detach()
    z_stats = torch.cat([
        z_detached.mean(dim=[0, 2, 3]),   # [C_latent] - channel means
        z_detached.std(dim=[0, 2, 3]),    # [C_latent] - channel stds
        z_detached.amin(dim=[0, 2, 3]),   # [C_latent] - channel mins
        z_detached.amax(dim=[0, 2, 3]),   # [C_latent] - channel maxs
    ])  # [C_latent * 4]
    
    return S, z_stats


def compute_dataset_sensitivity(
    vae: nn.Module,
    latent_dir: Union[str, Path],
    layer_channel_ranges: Dict[str, Tuple[int, int]],
    num_samples: int = 750,
    device: torch.device = None,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Compute sensitivity statistics over dataset samples.
    
    Loads pre-saved latents from disk and computes Jacobian sensitivity
    for a random subset. Returns mean/std sensitivity matrices and
    data for polynomial fitting.
    
    Args:
        vae: Full VAE model (will use decoder for sensitivity computation)
        latent_dir: Directory containing saved latent_*.pt files
        layer_channel_ranges: Dict mapping layer names to (start_ch, end_ch)
        num_samples: Number of latents to sample (default: 750)
        device: Device for computation
        seed: Random seed for reproducibility
        
    Returns:
        Dict containing:
            - 'sensitivity_matrix': Mean S [n_layers, n_latent]
            - 'sensitivity_std': Std of S [n_layers, n_latent]
            - 'z_stats_all': All z_stats [num_samples, n_latent * 4]
            - 'S_all': All sensitivity matrices [num_samples, n_layers, n_latent]
            - 'layer_names': List of layer names
    """
    if device is None:
        device = next(vae.parameters()).device
    
    latent_dir = Path(latent_dir)
    
    if not latent_dir.exists() or not latent_dir.is_dir():
        raise ValueError(f"Invalid latent directory: {latent_dir}")
    
    # Set VAE to eval mode
    vae.eval()
    
    # Find all latent files
    latent_files = sorted(latent_dir.glob('latent_*.pt'))
    if len(latent_files) == 0:
        raise ValueError(f"No latent files found in {latent_dir}")
    
    # Sample random subset
    random.seed(seed)
    if len(latent_files) > num_samples:
        sampled_files = random.sample(latent_files, num_samples)
    else:
        sampled_files = latent_files
        print(f"Warning: Only {len(latent_files)} latents available, using all of them")
    
    n_layers = len(layer_channel_ranges)
    layer_names = list(layer_channel_ranges.keys())
    
    # Infer latent channels from first file
    sample_z = torch.load(sampled_files[0], map_location=device, weights_only=True)
    if sample_z.dim() == 3:
        sample_z = sample_z.unsqueeze(0)  # Add batch dim if missing
    n_latent = sample_z.shape[1]
    
    # Initialize storage
    S_all = torch.zeros(len(sampled_files), n_layers, n_latent, device=device)
    z_stats_all = torch.zeros(len(sampled_files), n_latent * 4, device=device)
    
    print(f"\n{'='*60}")
    print(f"Computing Jacobian Sensitivity over {len(sampled_files)} samples")
    print(f"  Latent channels: {n_latent}")
    print(f"  Output layers: {layer_names}")
    print(f"{'='*60}")
    
    # Compute sensitivity for each sample
    with torch.no_grad():
        vae.eval()
    
    for i, latent_file in enumerate(tqdm(sampled_files, desc="Computing Jacobian")):
        # Load latent
        z = torch.load(latent_file, map_location=device, weights_only=True)
        if z.dim() == 3:
            z = z.unsqueeze(0)  # [1, C, H', W']
        
        # Compute sensitivity (requires grad tracking)
        S, z_stats = compute_jacobian_sensitivity(
            vae_or_decoder=vae,
            z=z,
            layer_channel_ranges=layer_channel_ranges,
            device=device,
        )
        
        S_all[i] = S
        z_stats_all[i] = z_stats
    
    # Compute statistics
    S_mean = S_all.mean(dim=0)  # [n_layers, n_latent]
    S_std = S_all.std(dim=0)    # [n_layers, n_latent]
    
    # Normalize S to sum to 1 per layer (for interpretability)
    S_normalized = S_mean / S_mean.sum(dim=1, keepdim=True)
    
    print(f"\n{'='*60}")
    print("Sensitivity Matrix (normalized per layer):")
    print(f"{'='*60}")
    print(f"{'Layer':<15} " + " ".join([f"Latent_{j:<5}" for j in range(n_latent)]))
    print("-" * 60)
    for layer_idx, layer_name in enumerate(layer_names):
        row = S_normalized[layer_idx].cpu().numpy()
        print(f"{layer_name:<15} " + " ".join([f"{v:>8.3f}" for v in row]))
    print(f"{'='*60}\n")
    
    return {
        'sensitivity_matrix': S_mean.cpu(),
        'sensitivity_normalized': S_normalized.cpu(),
        'sensitivity_std': S_std.cpu(),
        'z_stats_all': z_stats_all.cpu(),
        'S_all': S_all.cpu(),
        'layer_names': layer_names,
    }


def fit_polynomial_predictor(
    z_stats_all: torch.Tensor,
    S_all: torch.Tensor,
    degree: int = 2,
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Fit polynomial predictor: f(z_stats) -> S_matrix.
    
    For each (layer, latent) pair, fits a polynomial regression model
    that predicts sensitivity from latent statistics.
    
    Args:
        z_stats_all: Input features [num_samples, n_latent * 4]
        S_all: Target sensitivities [num_samples, n_layers, n_latent]
        degree: Polynomial degree (default: 2)
        alpha: Ridge regularization strength (default: 1.0)
        
    Returns:
        Dict containing:
            - 'coefficients': Nested dict [layer_idx][latent_idx] -> coefficients
            - 'poly_features': PolynomialFeatures object (for transform)
            - 'degree': Polynomial degree used
            - 'n_layers': Number of layers
            - 'n_latent': Number of latent channels
            - 'intercepts': Intercept terms [n_layers, n_latent]
    """
    z_stats_np = z_stats_all.numpy() if isinstance(z_stats_all, torch.Tensor) else z_stats_all
    S_np = S_all.numpy() if isinstance(S_all, torch.Tensor) else S_all
    
    num_samples, n_layers, n_latent = S_np.shape
    
    print(f"\n{'='*60}")
    print(f"Fitting Polynomial Predictor (degree={degree})")
    print(f"  Input features: {z_stats_np.shape[1]}")
    print(f"  Target pairs: {n_layers} layers x {n_latent} latents = {n_layers * n_latent}")
    print(f"  Samples: {num_samples}")
    print(f"{'='*60}")
    
    # Create polynomial features transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    z_poly = poly.fit_transform(z_stats_np)  # [num_samples, n_poly_features]
    
    n_poly_features = z_poly.shape[1]
    print(f"  Polynomial features: {n_poly_features}")
    
    # Storage for coefficients
    coefficients = {}
    intercepts = np.zeros((n_layers, n_latent))
    
    # Fit model for each (layer, latent) pair
    for layer_idx in range(n_layers):
        coefficients[layer_idx] = {}
        for latent_idx in range(n_latent):
            # Target: sensitivity for this (layer, latent) pair
            y = S_np[:, layer_idx, latent_idx]
            
            # Fit Ridge regression
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(z_poly, y)
            
            coefficients[layer_idx][latent_idx] = model.coef_
            intercepts[layer_idx, latent_idx] = model.intercept_
    
    # Compute R² for diagnostics
    y_pred_all = np.zeros_like(S_np)
    for layer_idx in range(n_layers):
        for latent_idx in range(n_latent):
            y_pred_all[:, layer_idx, latent_idx] = (
                z_poly @ coefficients[layer_idx][latent_idx] + 
                intercepts[layer_idx, latent_idx]
            )
    
    ss_res = np.sum((S_np - y_pred_all) ** 2)
    ss_tot = np.sum((S_np - S_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"  Overall R²: {r2:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'coefficients': coefficients,
        'intercepts': intercepts,
        'degree': degree,
        'n_layers': n_layers,
        'n_latent': n_latent,
        'n_poly_features': n_poly_features,
        'r2_score': r2,
    }


class SensitivityPredictor:
    """
    Predicts sensitivity matrix from latent statistics.
    
    Supports two modes:
    - 'linear': Uses fixed average sensitivity matrix (zero runtime cost)
    - 'polynomial': Uses polynomial regression for per-sample prediction
    
    Usage:
        # Linear mode
        predictor = SensitivityPredictor.from_fixed(S_matrix)
        S = predictor.predict(z)  # Returns same S for all z
        
        # Polynomial mode
        predictor = SensitivityPredictor.from_polynomial(poly_data)
        S = predictor.predict(z)  # Returns different S per sample
    """
    
    def __init__(
        self,
        mode: str,
        S_fixed: Optional[torch.Tensor] = None,
        poly_data: Optional[Dict] = None,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            mode: 'linear' or 'polynomial'
            S_fixed: Fixed sensitivity matrix [n_layers, n_latent] (for linear mode)
            poly_data: Polynomial regression data (for polynomial mode)
            layer_names: List of layer names
        """
        self.mode = mode
        self.S_fixed = S_fixed
        self.poly_data = poly_data
        self.layer_names = layer_names
        
        if mode == 'linear' and S_fixed is None:
            raise ValueError("S_fixed required for linear mode")
        if mode == 'polynomial' and poly_data is None:
            raise ValueError("poly_data required for polynomial mode")
        
        # Pre-compute polynomial features transformer if needed
        if mode == 'polynomial':
            self._poly_transformer = PolynomialFeatures(
                degree=poly_data['degree'],
                include_bias=False
            )
            # Fit transformer with dummy data to initialize
            n_latent = poly_data['n_latent']
            dummy = np.zeros((1, n_latent * 4))
            self._poly_transformer.fit(dummy)
    
    @classmethod
    def from_fixed(
        cls,
        S_matrix: torch.Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> 'SensitivityPredictor':
        """
        Create predictor with fixed sensitivity matrix (linear mode).
        
        Args:
            S_matrix: Sensitivity matrix [n_layers, n_latent]
            layer_names: List of layer names
        """
        return cls(
            mode='linear',
            S_fixed=S_matrix,
            layer_names=layer_names,
        )
    
    @classmethod
    def from_polynomial(
        cls,
        poly_data: Dict,
        S_fallback: Optional[torch.Tensor] = None,
        layer_names: Optional[List[str]] = None,
    ) -> 'SensitivityPredictor':
        """
        Create predictor with polynomial regression (polynomial mode).
        
        Args:
            poly_data: Dict from fit_polynomial_predictor()
            S_fallback: Fallback fixed S matrix (optional)
            layer_names: List of layer names
        """
        predictor = cls(
            mode='polynomial',
            poly_data=poly_data,
            S_fixed=S_fallback,
            layer_names=layer_names,
        )
        return predictor
    
    def compute_z_stats(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute statistics from latent tensor.
        
        Args:
            z: Latent tensor [B, C, H', W']
            
        Returns:
            z_stats: Statistics [B, C * 4] - [means, stds, mins, maxs]
        """
        return torch.cat([
            z.mean(dim=[2, 3]),   # [B, C]
            z.std(dim=[2, 3]),    # [B, C]
            z.amin(dim=[2, 3]),   # [B, C]
            z.amax(dim=[2, 3]),   # [B, C]
        ], dim=1)  # [B, C * 4]
    
    def predict(self, z: torch.Tensor = None, device: torch.device = None) -> torch.Tensor:
        """
        Predict sensitivity matrix for given latent.
        
        Args:
            z: Latent tensor [B, C, H', W'] (optional for linear mode)
            device: Device for output tensor (used when z is None in linear mode)
            
        Returns:
            S: Sensitivity matrix:
               - Linear mode: [n_layers, n_latent] (same for all samples)
               - Polynomial mode: [B, n_layers, n_latent] (per-sample)
        """
        if self.mode == 'linear':
            # For linear mode, z is optional - just return fixed S
            if z is not None:
                return self.S_fixed.to(z.device)
            elif device is not None:
                return self.S_fixed.to(device)
            else:
                return self.S_fixed
        
        elif self.mode == 'polynomial':
            batch_size = z.shape[0]
            n_layers = self.poly_data['n_layers']
            n_latent = self.poly_data['n_latent']
            device = z.device
            
            # Compute z statistics
            z_stats = self.compute_z_stats(z)  # [B, C * 4]
            z_stats_np = z_stats.cpu().numpy()
            
            # Apply polynomial transformation
            z_poly = self._poly_transformer.transform(z_stats_np)  # [B, n_poly]
            
            # Predict sensitivity for each (layer, latent) pair
            S_pred = np.zeros((batch_size, n_layers, n_latent))
            
            coefficients = self.poly_data['coefficients']
            intercepts = self.poly_data['intercepts']
            
            for layer_idx in range(n_layers):
                for latent_idx in range(n_latent):
                    S_pred[:, layer_idx, latent_idx] = (
                        z_poly @ coefficients[layer_idx][latent_idx] +
                        intercepts[layer_idx, latent_idx]
                    )
            
            # Ensure non-negative sensitivity
            S_pred = np.maximum(S_pred, 0.0)
            
            return torch.tensor(S_pred, device=device, dtype=z.dtype)
    
    def compute_latent_weights(
        self,
        z: torch.Tensor = None,
        layer_weights: torch.Tensor = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute latent channel weights from desired layer weights.
        
        Uses the formula: latent_weights = S.T @ layer_weights
        
        Args:
            z: Latent tensor [B, C, H', W'] (optional for linear mode)
            layer_weights: Desired layer weights [n_layers]
            device: Device for output tensor (used when z is None in linear mode)
            
        Returns:
            latent_weights:
               - Linear mode: [n_latent] (same for all samples)
               - Polynomial mode: [B, n_latent] (per-sample)
        """
        # Determine device: from z if provided, else from layer_weights, else from device param
        output_device = None
        if z is not None:
            output_device = z.device
        elif layer_weights is not None and hasattr(layer_weights, 'device'):
            output_device = layer_weights.device
        elif device is not None:
            output_device = device
        
        S = self.predict(z, device=output_device)  # [n_layers, n_latent] or [B, n_layers, n_latent]
        
        if self.mode == 'linear':
            # S: [n_layers, n_latent], layer_weights: [n_layers]
            # Result: [n_latent]
            return S.T @ layer_weights.to(S.device)
        
        else:
            # S: [B, n_layers, n_latent], layer_weights: [n_layers]
            # Result: [B, n_latent]
            return torch.einsum('blc,l->bc', S, layer_weights.to(S.device))
    
    def state_dict(self) -> Dict:
        """
        Get state dict for saving to checkpoint.
        """
        state = {
            'mode': self.mode,
            'layer_names': self.layer_names,
        }
        
        if self.S_fixed is not None:
            state['S_fixed'] = self.S_fixed
        
        if self.poly_data is not None:
            # Convert numpy arrays to lists for JSON-safe storage
            poly_state = {
                'degree': self.poly_data['degree'],
                'n_layers': self.poly_data['n_layers'],
                'n_latent': self.poly_data['n_latent'],
                'n_poly_features': self.poly_data['n_poly_features'],
                'intercepts': self.poly_data['intercepts'].tolist(),
                'coefficients': {},
            }
            for layer_idx in self.poly_data['coefficients']:
                poly_state['coefficients'][layer_idx] = {}
                for latent_idx in self.poly_data['coefficients'][layer_idx]:
                    poly_state['coefficients'][layer_idx][latent_idx] = (
                        self.poly_data['coefficients'][layer_idx][latent_idx].tolist()
                    )
            state['poly_data'] = poly_state
        
        return state
    
    @classmethod
    def from_state_dict(cls, state: Dict) -> 'SensitivityPredictor':
        """
        Load predictor from state dict.
        """
        mode = state['mode']
        layer_names = state.get('layer_names')
        
        S_fixed = state.get('S_fixed')
        if S_fixed is not None and not isinstance(S_fixed, torch.Tensor):
            S_fixed = torch.tensor(S_fixed)
        
        poly_data = None
        if 'poly_data' in state:
            ps = state['poly_data']
            poly_data = {
                'degree': ps['degree'],
                'n_layers': ps['n_layers'],
                'n_latent': ps['n_latent'],
                'n_poly_features': ps['n_poly_features'],
                'intercepts': np.array(ps['intercepts']),
                'coefficients': {},
            }
            for layer_idx_str in ps['coefficients']:
                layer_idx = int(layer_idx_str)
                poly_data['coefficients'][layer_idx] = {}
                for latent_idx_str in ps['coefficients'][layer_idx_str]:
                    latent_idx = int(latent_idx_str)
                    poly_data['coefficients'][layer_idx][latent_idx] = np.array(
                        ps['coefficients'][layer_idx_str][latent_idx_str]
                    )
        
        return cls(
            mode=mode,
            S_fixed=S_fixed,
            poly_data=poly_data,
            layer_names=layer_names,
        )


def compute_and_save_sensitivity(
    vae: nn.Module,
    latent_dir: Union[str, Path],
    layer_channel_ranges: Dict[str, Tuple[int, int]],
    checkpoint_path: Union[str, Path],
    num_samples: int = 750,
    polynomial_degree: int = 2,
    device: torch.device = None,
) -> Dict:
    """
    High-level function to compute sensitivity and update VAE checkpoint.
    
    Called at end of VAE training to add sensitivity data to checkpoint.
    
    Args:
        vae: Trained VAE model
        latent_dir: Directory containing saved latents
        layer_channel_ranges: Dict mapping layer names to channel ranges
        checkpoint_path: Path to VAE checkpoint to update
        num_samples: Number of samples for sensitivity computation
        polynomial_degree: Degree for polynomial predictor
        device: Device for computation
        
    Returns:
        Dict containing all sensitivity data
    """
    print(f"\n{'='*60}")
    print("Computing and Saving Sensitivity Data")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Latent directory: {latent_dir}")
    print(f"  Num samples: {num_samples}")
    print(f"  Polynomial degree: {polynomial_degree}")
    print(f"{'='*60}")
    
    # Compute dataset sensitivity
    sensitivity_data = compute_dataset_sensitivity(
        vae=vae,
        latent_dir=latent_dir,
        layer_channel_ranges=layer_channel_ranges,
        num_samples=num_samples,
        device=device,
    )
    
    # Fit polynomial predictor
    poly_data = fit_polynomial_predictor(
        z_stats_all=sensitivity_data['z_stats_all'],
        S_all=sensitivity_data['S_all'],
        degree=polynomial_degree,
    )
    
    # Create both predictors for saving
    predictor_linear = SensitivityPredictor.from_fixed(
        S_matrix=sensitivity_data['sensitivity_normalized'],
        layer_names=sensitivity_data['layer_names'],
    )
    
    predictor_poly = SensitivityPredictor.from_polynomial(
        poly_data=poly_data,
        S_fallback=sensitivity_data['sensitivity_normalized'],
        layer_names=sensitivity_data['layer_names'],
    )
    
    # Load existing checkpoint and add sensitivity data
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    checkpoint['sensitivity'] = {
        'sensitivity_matrix': sensitivity_data['sensitivity_matrix'],
        'sensitivity_normalized': sensitivity_data['sensitivity_normalized'],
        'sensitivity_std': sensitivity_data['sensitivity_std'],
        'layer_names': sensitivity_data['layer_names'],
        'linear_predictor': predictor_linear.state_dict(),
        'polynomial_predictor': predictor_poly.state_dict(),
        'computation_params': {
            'num_samples': num_samples,
            'polynomial_degree': polynomial_degree,
            'polynomial_r2': poly_data['r2_score'],
        },
    }
    
    # Save updated checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"\n✓ Sensitivity data saved to checkpoint: {checkpoint_path}")
    print(f"  - Linear predictor: sensitivity['linear_predictor']")
    print(f"  - Polynomial predictor: sensitivity['polynomial_predictor']")
    print(f"  - Polynomial R²: {poly_data['r2_score']:.4f}")
    
    return {
        'sensitivity_data': sensitivity_data,
        'poly_data': poly_data,
        'predictor_linear': predictor_linear,
        'predictor_poly': predictor_poly,
    }


def load_sensitivity_predictor(
    checkpoint_path: Union[str, Path],
    mode: str = 'linear',
) -> SensitivityPredictor:
    """
    Load sensitivity predictor from VAE checkpoint.
    
    Args:
        checkpoint_path: Path to VAE checkpoint
        mode: 'linear' or 'polynomial'
        
    Returns:
        SensitivityPredictor instance
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'sensitivity' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain sensitivity data. "
            f"Run sensitivity computation first."
        )
    
    sensitivity = checkpoint['sensitivity']
    
    if mode == 'linear':
        return SensitivityPredictor.from_state_dict(
            sensitivity['linear_predictor']
        )
    elif mode == 'polynomial':
        return SensitivityPredictor.from_state_dict(
            sensitivity['polynomial_predictor']
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'linear' or 'polynomial'.")

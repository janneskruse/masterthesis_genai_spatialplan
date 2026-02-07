"""
=============================================================================
LanPaint Mathematical Utilities
Adapted from official implementation: https://github.com/scraed/LanPaint
File: src/LanPaint/utils.py
=============================================================================
"""

###### import libraries ######
# Standard library
from typing import Optional, Tuple

# Data handling
import torch


def _expm1_x(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1) / x with numerical stability."""

    result = torch.special.expm1(x) / x
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x) < 1e-2
    result = torch.where(mask, 1 + x/2. + x**2 / 6., result)
    return result


def _expm1mx_x2(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1 - x) / x**2 with numerical stability."""

    # Compute the (exp(x) - 1 - x) / x**2 term with a small value to avoid division by zero.
    result = (torch.special.expm1(x) - x) / x**2
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**2) < 1e-2
    result = torch.where(mask, 1/2. + x/6 + x**2 / 24 + x**3 / 120, result)
    return result


def _expm1mxmhx2_x3(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1 - x - x**2/2) / x**3 with numerical stability."""

    # Compute the (exp(x) - 1 - x - x**2 / 2) / x**3 term with a small value to avoid division by zero.
    result = (torch.special.expm1(x) - x - x**2 / 2) / x**3
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**3) < 1e-2
    result = torch.where(mask, 1/6 + x/24 + x**2 / 120 + x**3 / 720 + x**4 / 5040, result)
    return result


def _exp_1mcosh_GD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:

    """
    Compute e^(-Γt) * (1 - cosh(Γt√Δ))/ ( (Γt)**2 Δ )

    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)

    Returns:
    Result of the computation with numerical stability handling
    """
    # Main computation
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta
    numerator_pos =  torch.exp(-gamma_t) - (torch.exp(gamma_t * (sqrt_abs_delta - 1)) + torch.exp(gamma_t * (-sqrt_abs_delta - 1))) / 2
    numerator_neg = torch.exp(-gamma_t) * ( 1 -  torch.cos(gamma_t * sqrt_abs_delta ) )
    numerator = torch.where(is_positive, numerator_pos, numerator_neg)
    result =  numerator / (delta * gamma_t**2 )
    # Handle NaN/inf cases
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    # Handle numerical instability for small delta
    mask = torch.abs(gamma_t_sqrt_delta**2) < 5e-2
    taylor = ( -0.5  - gamma_t**2 / 24 * delta - gamma_t**4 / 720 * delta**2 ) * torch.exp(-gamma_t)
    result = torch.where(mask, taylor, result)
    return result


def _exp_sinh_GsqrtD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute e^(-Γt) * sinh(Γt√Δ) / (Γt√Δ)

    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)

    Returns:
    Result of the computation with numerical stability handling
    """
    # Main computation
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta
    numerator_pos =  (torch.exp(gamma_t * (sqrt_abs_delta - 1)) - torch.exp(gamma_t * (-sqrt_abs_delta - 1))) / 2
    result_pos = numerator_pos / gamma_t_sqrt_delta
    result_pos = torch.where(torch.isfinite(result_pos), result_pos, torch.zeros_like(result_pos))

    # Taylor expansion for small gamma_t_sqrt_delta
    mask = torch.abs(gamma_t_sqrt_delta) < 1e-2
    taylor = ( 1  + gamma_t**2 / 6 * delta + gamma_t**4 / 120 * delta**2 ) * torch.exp(-gamma_t)
    result_pos = torch.where(mask, taylor, result_pos)

    # Handle negative delta
    result_neg = torch.exp(-gamma_t) * torch.special.sinc(gamma_t_sqrt_delta/torch.pi)
    result = torch.where(is_positive, result_pos, result_neg)
    return result


def _exp_cosh(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute e^(-Γt) * cosh(Γt√Δ)

    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)

    Returns:
    Result of the computation with numerical stability handling
    """
    exp_1mcosh_GD_result = _exp_1mcosh_GD(gamma_t, delta) # e^(-Γt) * (1 - cosh(Γt√Δ))/ ( (Γt)**2 Δ )
    result = torch.exp(-gamma_t) - gamma_t**2 * delta * exp_1mcosh_GD_result
    return result


def _exp_sinh_sqrtD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Compute e^(-Γt) * sinh(Γt√Δ) / √Δ
    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)
    Returns:
    Result of the computation with numerical stability handling
    """
    exp_sinh_GsqrtD_result = _exp_sinh_GsqrtD(gamma_t, delta) # e^(-Γt) * sinh(Γt√Δ) / (Γt√Δ)
    result = gamma_t * exp_sinh_GsqrtD_result
    return result

def _zeta1(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute ζ₁ coefficient for SHO."""
    half_gamma_t = gamma_t / 2
    exp_cosh_term = _exp_cosh(half_gamma_t, delta)
    exp_sinh_term = _exp_sinh_sqrtD(half_gamma_t, delta)
    
    # Main computation
    numerator = 1 - (exp_cosh_term + exp_sinh_term)
    denominator = gamma_t * (1 - delta) / 4
    result = 1 - numerator / denominator
    
    # Handle numerical instability
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    
    # Taylore expansion for small denominator (smaller to epxm1x approach)
    mask = torch.abs(denominator) < 5e-3
    term1 = _expm1_x(-gamma_t)
    term2 = _expm1mx_x2(-gamma_t)
    term3 = _expm1mxmhx2_x3(-gamma_t)
    taylor = (
        term1 + 
        (1/2. + term1 - 3*term2) * denominator + 
        (-1/6. + term1/2 - 4*term2 + 10*term3) * denominator**2
    )
    result = torch.where(mask, taylor, result)
    return result


def _zeta2(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute ζ₂ coefficient for SHO."""
    half_gamma_t = gamma_t / 2
    return _exp_sinh_GsqrtD(half_gamma_t, delta)


def _sig11(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute σ₁₁ variance coefficient."""
    return (
        1 - torch.exp(-gamma_t) + 
        gamma_t**2 * _exp_1mcosh_GD(gamma_t, delta) + 
        _exp_sinh_sqrtD(gamma_t, delta)
    )


def _sig22(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute σ₂₂ variance coefficient."""
    return 1 - _zeta1(2*gamma_t, delta) + 2 * gamma_t * _exp_1mcosh_GD(gamma_t, delta)


class StochasticHarmonicOscillator:
    """
    Exact analytical integrator for the Stochastic Harmonic Oscillator.
    
    Ported from official LanPaint implementation:
    https://github.com/scraed/LanPaint/blob/main/src/LanPaint/utils.py
    https://github.com/scraed/LanPaintBench/blob/main/utils_math.py
    
    Solves the SDE system:
        dy(t) = q(t) dt
        dq(t) = -Γ A y(t) dt + Γ C dt + Γ D dw(t) - Γ q(t) dt
        
    Also defines v(t) = q(t) / √Γ, which is numerically more stable.
    
    Where:
        y(t) - Position variable
        q(t) - Velocity variable
        Γ - Damping coefficient
        A - Harmonic potential strength
        C - Constant force term (equilibrium position)
        D - Noise amplitude
        dw(t) - Wiener process (Brownian motion)
        
    Uses exact analytical solution via special functions (zeta1, zeta2, etc.)
    with multivariate normal sampling for correlated position-velocity updates.
    """
    
    def __init__(
        self,
        Gamma: torch.Tensor,
        A: torch.Tensor, 
        C: torch.Tensor,
        D: torch.Tensor
    ):
        """
        Initialize the oscillator with physical parameters.
        
        Args:
            Gamma: Damping coefficient Γ
            A: Harmonic potential strength (spring constant)
            C: Equilibrium position / constant force term
            D: Noise amplitude
        """
        self.Gamma = Gamma
        self.A = A
        self.C = C
        self.D = D
        # Delta parameter: Δ = 1 - 4A/Γ (discriminant for eigenvalues)
        self.Delta = 1 - 4 * A / Gamma
    
    def dynamics(
        self,
        y0: torch.Tensor,
        v0: Optional[torch.Tensor],
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate position and velocity at time t using exact analytical solution.
        
        This is the core SHO integrator that computes the exact solution of the
        stochastic differential equation, including proper covariance structure
        for the noise terms.
        
        Args:
            y0: Initial position (B, C, H, W)
            v0: Initial velocity v(0) = q(0) / √Γ. If None, sampled from stationary distribution.
            t: Time step (scalar tensor)
            
        Returns:
            (y(t), v(t)): Updated position and velocity
        """
        # Create dummy zero tensor with proper device/dtype
        dummyzero = y0.new_zeros([1] * y0.dim())
        
        # Broadcast parameters to match y0 shape
        Delta = self.Delta + dummyzero
        Gamma_hat = self.Gamma * t + dummyzero
        A = self.A + dummyzero
        C = self.C + dummyzero
        D = self.D + dummyzero
        Gamma = self.Gamma + dummyzero
        
        # Compute special function coefficients
        zeta_1 = _zeta1(Gamma_hat, Delta)
        zeta_2 = _zeta2(Gamma_hat, Delta)
        EE = 1 - Gamma_hat * zeta_2
        
        # Initialize velocity from stationary distribution if not provided
        if v0 is None:
            v0 = torch.randn_like(y0) * D / 2 ** 0.5
        
        # Calculate mean position and velocity (exact analytical solution)
        term1 = (1 - zeta_1) * (C * t - A * t * y0) + zeta_2 * (Gamma ** 0.5) * v0 * t
        y_mean = term1 + y0
        v_mean = (1 - EE) * (C - A * y0) / (Gamma ** 0.5) + (EE - A * t * (1 - zeta_1)) * v0
        
        # Compute covariance matrix elements
        cov_yy = D**2 * t * _sig22(Gamma_hat, Delta)
        cov_vv = D**2 * _sig11(Gamma_hat, Delta) / 2
        cov_yv = (_zeta2(Gamma_hat, Delta) * Gamma_hat * D)**2 / 2 / (Gamma ** 0.5)
        
        # Build Cholesky decomposition for multivariate normal sampling
        #scale_tril = torch.linalg.cholesky(cov_matrix)
        batch_shape = y0.shape
        scale_tril = torch.zeros(*batch_shape, 2, 2, device=y0.device, dtype=y0.dtype)
        
        tol = 1e-8
        cov_yy = torch.clamp(cov_yy, min=tol)
        sd_yy = torch.sqrt(cov_yy)
        inv_sd_yy = 1 / sd_yy
        
        # check if it matches torch.linalg.
        #assert torch.allclose(torch.linalg.cholesky(cov_matrix), scale_tril, atol = 1e-4, rtol = 1e-4 )
        # Sample correlated noise from multivariate normal
        scale_tril[..., 0, 0] = sd_yy
        scale_tril[..., 0, 1] = 0.
        scale_tril[..., 1, 0] = cov_yv * inv_sd_yy
        scale_tril[..., 1, 1] = torch.clamp(cov_vv - cov_yv**2 / cov_yy, min=tol) ** 0.5
        
        # Sample correlated noise from multivariate normal
        mean = torch.zeros(*batch_shape, 2, device=y0.device, dtype=y0.dtype)
        mean[..., 0] = y_mean
        mean[..., 1] = v_mean
        
        new_yv = torch.distributions.MultivariateNormal(
            loc=mean,
            scale_tril=scale_tril
        ).sample()
        
        return new_yv[..., 0], new_yv[..., 1]


class StochasticHarmonicOscillatorSimple:
    """
    Simplified SHO integrator for use when full analytical solution is not needed.
    
    This provides a simpler API that creates the oscillator per-step rather than
    requiring pre-initialization. Used as a wrapper for LanPaint integration.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def step(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor],
        dt: torch.Tensor,
        Gamma: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of exact SHO integration.
        
        Args:
            x: Position (B, C, H, W)
            v: Velocity (B, C, H, W), or None to sample from stationary distribution
            dt: Time step
            Gamma: Friction coefficient Γ
            A: Spring constant
            C: Equilibrium position
            D: Diffusion coefficient
            
        Returns:
            (x_new, v_new): Updated position and velocity
        """
        dtype = x.dtype
        
        # Use float32 for numerical stability
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            osc = StochasticHarmonicOscillator(Gamma, A, C, D)
            x_new, v_new = osc.dynamics(x, v, dt)
        
        return x_new.to(dtype), v_new.to(dtype)
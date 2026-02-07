# Test script for Jacobian Sensitivity computation
# Validates the sensitivity analysis works correctly before using in training

###### import libraries ######
# Standard libraries
import os
import sys
import argparse
from pathlib import Path
import time

# Data Science/ML libraries
import torch
import numpy as np

# Local imports
from model.blocks.vae import VAE
from model.utils.jacobian_sensitivity import (
    compute_jacobian_sensitivity,
    compute_dataset_sensitivity,
    fit_polynomial_predictor,
    compute_and_save_sensitivity,
    load_sensitivity_predictor,
    SensitivityPredictor,
)
from helpers.load_configs import load_configs


def test_single_sample_jacobian(vae, latent_dir: Path, layer_channel_ranges: dict, device: torch.device):
    """Test Jacobian computation on a single sample."""
    print("\n" + "="*60)
    print("TEST 1: Single Sample Jacobian Computation")
    print("="*60)
    
    # Load one latent
    latent_files = sorted(latent_dir.glob('latent_*.pt'))
    if not latent_files:
        raise ValueError(f"No latent files found in {latent_dir}")
    
    z = torch.load(latent_files[0], map_location=device)
    if z.dim() == 3:
        z = z.unsqueeze(0)
    
    print(f"  Latent shape: {z.shape}")
    print(f"  Latent range: [{z.min().item():.3f}, {z.max().item():.3f}]")
    print(f"  Layer channel ranges: {layer_channel_ranges}")
    
    # Test decoder output
    with torch.no_grad():
        decoded = vae.decode(z)
    print(f"  Decoded shape: {decoded.shape}")
    print(f"  Decoded range: [{decoded.min().item():.3f}, {decoded.max().item():.3f}]")
    
    # Compute Jacobian - pass full VAE model
    start_time = time.time()
    S, z_stats = compute_jacobian_sensitivity(
        vae_or_decoder=vae,
        z=z,
        layer_channel_ranges=layer_channel_ranges,
        device=device,
    )
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Jacobian computed in {elapsed:.3f}s")
    print(f"  Sensitivity matrix S shape: {S.shape}")
    print(f"  z_stats shape: {z_stats.shape}")
    
    # Print sensitivity matrix
    n_layers = len(layer_channel_ranges)
    n_latent = z.shape[1]
    layer_names = list(layer_channel_ranges.keys())
    
    print(f"\n  Sensitivity Matrix S [n_layers={n_layers}, n_latent={n_latent}]:")
    print(f"  {'Layer':<15} " + " ".join([f"Latent_{j}" for j in range(n_latent)]))
    print("  " + "-" * 50)
    for i, layer_name in enumerate(layer_names):
        row = S[i].cpu().numpy()
        print(f"  {layer_name:<15} " + " ".join([f"{v:>8.2f}" for v in row]))
    
    # Validate sensitivity values are reasonable
    assert S.min() >= 0, "Sensitivity should be non-negative (Frobenius norm)"
    assert S.max() < 1e6, "Sensitivity values seem too large"
    assert not torch.isnan(S).any(), "NaN values in sensitivity matrix"
    assert not torch.isinf(S).any(), "Inf values in sensitivity matrix"
    
    print("\n  ✓ TEST 1 PASSED: Single sample Jacobian computation works!")
    return S, z_stats


def test_dataset_sensitivity(vae, latent_dir: Path, layer_channel_ranges: dict, device: torch.device, num_samples: int = 50):
    """Test sensitivity computation over multiple samples."""
    print("\n" + "="*60)
    print(f"TEST 2: Dataset Sensitivity Computation ({num_samples} samples)")
    print("="*60)
    
    start_time = time.time()
    sensitivity_data = compute_dataset_sensitivity(
        vae=vae,
        latent_dir=latent_dir,
        layer_channel_ranges=layer_channel_ranges,
        num_samples=num_samples,
        device=device,
        seed=42,
    )
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Dataset sensitivity computed in {elapsed:.2f}s")
    print(f"  Mean sensitivity shape: {sensitivity_data['sensitivity_matrix'].shape}")
    print(f"  Std sensitivity shape: {sensitivity_data['sensitivity_std'].shape}")
    print(f"  z_stats_all shape: {sensitivity_data['z_stats_all'].shape}")
    print(f"  S_all shape: {sensitivity_data['S_all'].shape}")
    
    # Check normalized sensitivity sums to 1 per layer
    S_norm = sensitivity_data['sensitivity_normalized']
    row_sums = S_norm.sum(dim=1)
    print(f"\n  Normalized sensitivity row sums (should be ~1.0): {row_sums.numpy()}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Normalized rows should sum to 1"
    
    print("\n  ✓ TEST 2 PASSED: Dataset sensitivity computation works!")
    return sensitivity_data


def test_polynomial_fitting(sensitivity_data: dict):
    """Test polynomial predictor fitting."""
    print("\n" + "="*60)
    print("TEST 3: Polynomial Predictor Fitting")
    print("="*60)
    
    start_time = time.time()
    poly_data = fit_polynomial_predictor(
        z_stats_all=sensitivity_data['z_stats_all'],
        S_all=sensitivity_data['S_all'],
        degree=2,
        alpha=1.0,
    )
    elapsed = time.time() - start_time
    
    print(f"\n  ✓ Polynomial fitting completed in {elapsed:.3f}s")
    print(f"  Polynomial degree: {poly_data['degree']}")
    print(f"  Number of polynomial features: {poly_data['n_poly_features']}")
    print(f"  R² score: {poly_data['r2_score']:.4f}")
    
    # R² should be reasonable (not too low, not exactly 1.0)
    assert poly_data['r2_score'] > 0.0, "R² should be positive"
    assert poly_data['r2_score'] <= 1.0, "R² should be <= 1.0"
    
    print("\n  ✓ TEST 3 PASSED: Polynomial predictor fitting works!")
    return poly_data


def test_sensitivity_predictor(sensitivity_data: dict, poly_data: dict, latent_dir: Path, device: torch.device):
    """Test SensitivityPredictor class."""
    print("\n" + "="*60)
    print("TEST 4: SensitivityPredictor Class")
    print("="*60)
    
    # Load a batch of latents
    latent_files = sorted(latent_dir.glob('latent_*.pt'))[:4]
    z_batch = torch.stack([
        torch.load(f, map_location=device) for f in latent_files
    ])
    if z_batch.dim() == 4:
        pass  # Already [B, C, H, W]
    else:
        z_batch = z_batch.unsqueeze(1)  # Add channel dim if needed
    
    print(f"  Test batch shape: {z_batch.shape}")
    
    # Test linear predictor
    print("\n  --- Linear Predictor ---")
    predictor_linear = SensitivityPredictor.from_fixed(
        S_matrix=sensitivity_data['sensitivity_normalized'],
        layer_names=sensitivity_data['layer_names'],
    )
    
    S_linear = predictor_linear.predict(z_batch)
    print(f"  S_linear shape: {S_linear.shape} (should be [n_layers, n_latent])")
    assert S_linear.dim() == 2, "Linear predictor should return 2D tensor"
    
    # Test latent weights computation
    layer_weights = torch.tensor([3.0, 0.3, 1.0])  # buildings, streets, vegetation
    latent_weights_linear = predictor_linear.compute_latent_weights(z_batch, layer_weights)
    print(f"  latent_weights_linear shape: {latent_weights_linear.shape} (should be [n_latent])")
    print(f"  latent_weights_linear: {latent_weights_linear.cpu().numpy()}")
    
    # Test polynomial predictor
    print("\n  --- Polynomial Predictor ---")
    predictor_poly = SensitivityPredictor.from_polynomial(
        poly_data=poly_data,
        S_fallback=sensitivity_data['sensitivity_normalized'],
        layer_names=sensitivity_data['layer_names'],
    )
    
    S_poly = predictor_poly.predict(z_batch)
    print(f"  S_poly shape: {S_poly.shape} (should be [B, n_layers, n_latent])")
    assert S_poly.dim() == 3, "Polynomial predictor should return 3D tensor"
    assert S_poly.shape[0] == z_batch.shape[0], "Batch dimension should match"
    
    latent_weights_poly = predictor_poly.compute_latent_weights(z_batch, layer_weights)
    print(f"  latent_weights_poly shape: {latent_weights_poly.shape} (should be [B, n_latent])")
    print(f"  latent_weights_poly[0]: {latent_weights_poly[0].cpu().numpy()}")
    
    # Test state_dict serialization
    print("\n  --- State Dict Serialization ---")
    linear_state = predictor_linear.state_dict()
    poly_state = predictor_poly.state_dict()
    
    # Reconstruct from state dict
    predictor_linear_restored = SensitivityPredictor.from_state_dict(linear_state)
    predictor_poly_restored = SensitivityPredictor.from_state_dict(poly_state)
    
    # Verify restored predictors work
    S_linear_restored = predictor_linear_restored.predict(z_batch)
    S_poly_restored = predictor_poly_restored.predict(z_batch)
    
    assert torch.allclose(S_linear, S_linear_restored), "Linear predictor state dict round-trip failed"
    assert torch.allclose(S_poly, S_poly_restored, atol=1e-5), "Polynomial predictor state dict round-trip failed"
    
    print("  ✓ State dict serialization works!")
    
    print("\n  ✓ TEST 4 PASSED: SensitivityPredictor class works!")
    return predictor_linear, predictor_poly


def test_full_integration(vae, latent_dir: Path, layer_channel_ranges: dict, checkpoint_path: Path, device: torch.device):
    """Test full integration: compute and save to checkpoint."""
    print("\n" + "="*60)
    print("TEST 5: Full Integration (Compute + Save to Checkpoint)")
    print("="*60)
    
    # Create a temporary checkpoint copy for testing
    import shutil
    test_checkpoint_path = checkpoint_path.parent / "test_sensitivity_checkpoint.pth"
    shutil.copy(checkpoint_path, test_checkpoint_path)
    
    print(f"  Created test checkpoint: {test_checkpoint_path}")
    
    try:
        # Run full computation and save
        start_time = time.time()
        result = compute_and_save_sensitivity(
            vae=vae,
            latent_dir=latent_dir,
            layer_channel_ranges=layer_channel_ranges,
            checkpoint_path=test_checkpoint_path,
            num_samples=50,  # Small for testing
            polynomial_degree=2,
            device=device,
        )
        elapsed = time.time() - start_time
        
        print(f"\n  ✓ Full computation completed in {elapsed:.2f}s")
        
        # Verify checkpoint was updated
        checkpoint = torch.load(test_checkpoint_path, map_location='cpu', weights_only=False)
        assert 'sensitivity' in checkpoint, "Sensitivity data not saved to checkpoint"
        
        sensitivity = checkpoint['sensitivity']
        assert 'sensitivity_matrix' in sensitivity
        assert 'sensitivity_normalized' in sensitivity
        assert 'linear_predictor' in sensitivity
        assert 'polynomial_predictor' in sensitivity
        
        print("  ✓ Checkpoint contains all sensitivity data!")
        
        # Test loading predictor from checkpoint
        predictor_linear = load_sensitivity_predictor(test_checkpoint_path, mode='linear')
        predictor_poly = load_sensitivity_predictor(test_checkpoint_path, mode='polynomial')
        
        print(f"  ✓ Loaded linear predictor: mode={predictor_linear.mode}")
        print(f"  ✓ Loaded polynomial predictor: mode={predictor_poly.mode}")
        
        print("\n  ✓ TEST 5 PASSED: Full integration works!")
        
    finally:
        # Cleanup test checkpoint
        if test_checkpoint_path.exists():
            test_checkpoint_path.unlink()
            print(f"\n  Cleaned up test checkpoint: {test_checkpoint_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test Jacobian Sensitivity Computation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint (e.g., semantic_vae_ckpt.pth)')
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Path to directory with saved latents')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, will use default)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples for dataset tests (default: 50)')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Layer names to use (default: infer from config or use buildings,streets,vegetation)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("Jacobian Sensitivity Test Suite")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Latent dir: {args.latent_dir}")
    
    checkpoint_path = Path(args.checkpoint)
    latent_dir = Path(args.latent_dir)
    
    # Verify paths exist
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert latent_dir.exists(), f"Latent directory not found: {latent_dir}"
    
    # Load config
    try:
        config = load_configs()
    except Exception as e:
        print(f"  Warning: Could not load config: {e}")
        print("  Using default semantic VAE configuration...")
        config = None
    
    # Setup layer_channel_ranges for semantic VAE
    # Priority: 1) --layers argument, 2) config file, 3) default
    if args.layers is not None:
        # Use user-provided layers
        layers = args.layers
        print(f"  Using user-specified layers: {layers}")
    elif config is not None:
        # Infer from config
        vae_groups = config.get('vae_groups', {})
        semantic_group = vae_groups.get('semantic', {})
        layers = semantic_group.get('layers', ['buildings', 'streets', 'vegetation'])
        print(f"  Using layers from config: {layers}")
    else:
        # Default fallback
        layers = ['buildings', 'streets', 'vegetation']
        print(f"  Using default layers: {layers}")
    
    # Build channel ranges (assume 1 channel per layer for semantic)
    layer_channel_ranges = {}
    offset = 0
    for layer_name in layers:
        n_channels = 1  # Semantic layers are typically 1 channel each
        if config is not None:
            layer_config = config.get('layers', {}).get(layer_name, {})
            n_channels = len(layer_config.get('channels', [''])) if 'channels' in layer_config else 1
        layer_channel_ranges[layer_name] = (offset, offset + n_channels)
        offset += n_channels
    
    print(f"  Layer channel ranges: {layer_channel_ranges}")
    
    # Load checkpoint to get VAE config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Infer VAE architecture from checkpoint
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # Find input channels from first conv layer
    encoder_conv_in_weight = model_state.get('encoder_conv_in.weight')
    if encoder_conv_in_weight is not None:
        im_channels = encoder_conv_in_weight.shape[1]
    else:
        im_channels = 3  # Default for semantic
    
    # Find latent channels from post_quant_conv (decoder side, more reliable)
    # post_quant_conv goes from z_channels -> z_channels
    post_quant_conv_weight = model_state.get('post_quant_conv.weight')
    if post_quant_conv_weight is not None:
        z_channels = post_quant_conv_weight.shape[0]  # Output channels = z_channels
    else:
        # Fallback: infer from pre_quant_conv (encoder side: 2*z_channels -> 2*z_channels)
        pre_quant_conv_weight = model_state.get('pre_quant_conv.weight')
        if pre_quant_conv_weight is not None:
            z_channels = pre_quant_conv_weight.shape[0] // 2
        else:
            z_channels = 3  # Default
    
    print(f"  Inferred im_channels: {im_channels}")
    print(f"  Inferred z_channels: {z_channels}")
    
    # Build VAE config - use inferred values, don't override from config
    vae_config = {
        'z_channels': z_channels,  # Use inferred value!
        'down_channels': [32, 64, 128, 128],
        'mid_channels': [128, 128],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 2,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
    }
    
    # Override architecture params from config (but NOT z_channels which we inferred)
    if config is not None:
        vae_groups = config.get('vae_groups', {})
        semantic_config = vae_groups.get('semantic', {})
        for key in ['down_channels', 'mid_channels', 'down_sample', 'attn_down', 
                    'norm_channels', 'num_heads', 'num_down_layers', 'num_mid_layers', 'num_up_layers']:
            if key in semantic_config:
                vae_config[key] = semantic_config[key]
    
    # Create and load VAE
    print(f"\n  Creating VAE model...")
    vae = VAE(im_channels=im_channels, model_config=vae_config).to(device)
    
    # Load weights
    vae.load_state_dict(model_state)
    vae.eval()
    print(f"  ✓ VAE loaded successfully!")
    
    # Run tests
    try:
        # Test 1: Single sample
        S, z_stats = test_single_sample_jacobian(vae, latent_dir, layer_channel_ranges, device)
        
        # Test 2: Dataset sensitivity
        sensitivity_data = test_dataset_sensitivity(vae, latent_dir, layer_channel_ranges, device, num_samples=args.num_samples)
        
        # Test 3: Polynomial fitting
        poly_data = test_polynomial_fitting(sensitivity_data)
        
        # Test 4: SensitivityPredictor class
        predictor_linear, predictor_poly = test_sensitivity_predictor(sensitivity_data, poly_data, latent_dir, device)
        
        # Test 5: Full integration
        result = test_full_integration(vae, latent_dir, layer_channel_ranges, checkpoint_path, device)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe Jacobian sensitivity computation is working correctly.")
        print("You can safely run VAE training with automatic sensitivity computation.")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

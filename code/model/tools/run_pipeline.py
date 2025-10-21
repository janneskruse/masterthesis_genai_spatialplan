#!/usr/bin/env python
"""
Quick start script for urban inpainting training pipeline.
Runs all steps in sequence with error checking.
"""

import subprocess
import sys
import os
import argparse


def run_command(cmd, description):
    """Run a command and check for errors."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        print(f"Exit code: {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Urban inpainting training pipeline')
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip dataset validation'
    )
    parser.add_argument(
        '--skip-vae',
        action='store_true',
        help='Skip VAE training (if already trained)'
    )
    parser.add_argument(
        '--skip-diffusion',
        action='store_true',
        help='Skip diffusion training (if already trained)'
    )
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Only run sampling (skip training)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='Number of samples to generate'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("URBAN INPAINTING TRAINING PIPELINE")
    print("="*60)
    print()
    
    # Change to model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    os.chdir(model_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    success = True
    
    # Step 1: Validate dataset
    if not args.skip_validation and not args.sample_only:
        cmd = f"python tools/validate_dataset.py --num_samples 3"
        success = run_command(cmd, "Dataset Validation")
        if not success:
            print("\n⚠️  Dataset validation failed. Please fix dataset issues before continuing.")
            return 1
    
    # Step 2: Train VAE
    if not args.skip_vae and not args.sample_only:
        cmd = f"python tools/train_vae_urban.py"
        success = run_command(cmd, "VAE Training")
        if not success:
            print("\n⚠️  VAE training failed. Check error messages above.")
            return 1
    
    # Step 3: Train Diffusion Model
    if not args.skip_diffusion and not args.sample_only:
        cmd = f"python tools/train_urban_inpainting.py"
        success = run_command(cmd, "Diffusion Model Training")
        if not success:
            print("\n⚠️  Diffusion training failed. Check error messages above.")
            return 1
    
    # Step 4: Generate Samples
    cmd = f"python tools/sample_urban_inpainting.py --num_samples {args.num_samples}"
    success = run_command(cmd, "Sample Generation")
    if not success:
        print("\n⚠️  Sampling failed. Check error messages above.")
        return 1
    
    # Success!
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck outputs in: urban_layout_inpainting/")
    print("  - VAE samples: vae_samples/")
    print("  - Inpainting samples: inpainting_samples/")
    print("\nTo generate more samples:")
    print(f"  python tools/sample_urban_inpainting.py --num_samples 16")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

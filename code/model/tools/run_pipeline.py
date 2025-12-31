#!/usr/bin/env python
"""
Quick start script for urban inpainting training pipeline.
Runs all steps in sequence with error checking.
"""
###### import libraries ######
# Standard libraries
import subprocess
import sys
import os
import argparse
from pathlib import Path

# local imports
from helpers.load_configs import load_configs
from model.utils.config_utils import get_config_value


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
    # Create parser with all arguments
    parser = argparse.ArgumentParser(description='Urban inpainting training pipeline')
    
    # Add config file arguments
    from helpers.load_configs import add_config_arguments
    add_config_arguments(parser)
    
    # Add pipeline-specific arguments
    parser.add_argument(
        '--validate-dataset',
        action='store_true',
        help='Validate the dataset before training',
        default=False
    )
    parser.add_argument(
        '--skip-semantic-vae',
        action='store_true',
        help='Skip Semantic VAE training (if already trained)'
    )
    parser.add_argument(
        '--skip-satellite-vae',
        action='store_true',
        help='Skip Satellite VAE training (if already trained)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*60)
    print("URBAN INPAINTING TRAINING PIPELINE")
    print("="*60)
    print()
    
    # Load configs using the parser
    config = load_configs(parser)
    # data_config = config['data_config']
    big_data_storage_path = config['dataset_params']['big_data_storage_path']
    task_name = config['train_params']['task_name']
    
    cluster_run = get_config_value(config, 'cluster', default=False)
    
    # Change to model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    os.chdir(model_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    success = True
    
    # Step 1: Validate dataset
    if args.validate_dataset:
        cmd = f"python tools/validate_dataset.py --config {args.config} --num_samples 3"
        success = run_command(cmd, "Dataset Validation")
        if not success:
            print("\n⚠️  Dataset validation failed. Please fix dataset issues before continuing.")
            return 1
    
    # Step 2: Create patches
    cache_dir_base = Path(big_data_storage_path) / "processed" / task_name
    cache_dir_semantic = cache_dir_base / "semantic"
    cache_dir_satellite = cache_dir_base / "satellite"
    
    if not cache_dir_semantic.exists() or len(os.listdir(cache_dir_semantic)) == 0 or \
       not cache_dir_satellite.exists() or len(os.listdir(cache_dir_satellite)) == 0:
        cmd = f"python tools/prepare_patches.py --config {args.config}"
        success = run_command(cmd, "Create Patches")
        if not success:
            print("\n⚠️  Patch preparation failed. Check error messages above.")
            return 1
    
    # Step 2: Submit pipelines
    if not args.skip_semantic_vae:
        if cluster_run:
            cmd = f"sbatch tools/train_semantic_vae_ddp.sh --config {args.config}"
        else:
            cmd = f"python tools/train_semantic_vae.py --config {args.config}"
        success = run_command(cmd, "Semantic VAE Training")
        if not success:
            print("\n⚠️  Semantic VAE training failed. Check error messages above.")
            return 1
    else: # submit diffusion training directly if VAE training is skipped
        print("\n⚠️  Skipping Semantic VAE training as per user request.")
        print("    Proceeding to Semantic Diffusion training step.\n")
        if cluster_run:
            cmd = f"sbatch tools/train_semantic_diffusion_inpainting_ddp.sh --config {args.config}"
        else:
            cmd = f"python tools/train_semantic_diffusion_inpainting.py --config {args.config}"
        success = run_command(cmd, "Semantic Diffusion Training")
        if not success:
            print("\n⚠️  Semantic Diffusion training failed. Check error messages above.")
            return 1
    
    if not args.skip_satellite_vae:
        if cluster_run:
            cmd = f"sbatch tools/train_satellite_vae_ddp.sh --config {args.config}"
        else:
            cmd = f"python tools/train_satellite_vae.py --config {args.config}"
        success = run_command(cmd, "Satellite VAE Training")
        if not success:
            print("\n⚠️  Satellite VAE training failed. Check error messages above.")
            return 1
    else:
        print("\n⚠️  Skipping Satellite VAE training as per user request.\n")   
        print("    Proceeding to Satellite Diffusion training step.\n")
        if cluster_run:
            cmd = f"sbatch tools/train_satellite_diffusion_inpainting_ddp.sh --config {args.config}"
        else:
            cmd = f"python tools/train_satellite_diffusion_inpainting.py --config {args.config}"
        success = run_command(cmd, "Satellite Diffusion Training")
        if not success:
            print("\n⚠️  Satellite Diffusion training failed. Check error messages above.")
            return 1
        
    
    # Success!
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck outputs in: urban_layout_inpainting/")
    print("  - VAE samples: vae_samples/")
    print("  - Inpainting samples: inpainting_samples/")
    print("\nTo generate more samples:")
    print(f"  python tools/sample_urban_inpainting.py --config {args.config} --num_samples 16")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

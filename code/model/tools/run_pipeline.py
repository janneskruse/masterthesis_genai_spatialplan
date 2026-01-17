#!/usr/bin/env python
"""
Quick start script for urban inpainting training pipeline.
Runs all steps in sequence with error checking.
Config-driven: dynamically submits jobs for all VAE groups and diffusion stages.
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


def find_sbatch_script(mode_type, mode_name, script_dir):
    """
    Find sbatch script for a given mode.
    
    Args:
        mode_type: 'vae' or 'diffusion'
        mode_name: e.g., 'semantic', 'satellite', 'environmental'
        script_dir: Directory containing sbatch scripts
        
    Returns:
        Path to script if found, None otherwise
    """
    script_name = f"train_{mode_name}_{mode_type}_ddp.sh"
    script_path = script_dir / script_name
    
    if script_path.exists():
        return script_path
    return None


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
        '--skip-modes',
        nargs='+',
        default=[],
        help='List of modes to skip (e.g., vae:semantic diffusion:satellite)',
        metavar='MODE'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*60)
    print("URBAN INPAINTING TRAINING PIPELINE")
    print("="*60)
    print()
    
    # Load configs using the parser
    config = load_configs(parser)
    big_data_storage_path = config['dataset_params']['big_data_storage_path']
    task_name = config['train_params']['task_name']
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Parse skip modes
    skip_modes = set(args.skip_modes)
    if skip_modes:
        print(f"Skip modes: {', '.join(skip_modes)}\n")
    
    success = True
    
    # Step 1: Validate dataset
    if args.validate_dataset:
        cmd = f"python validate_dataset.py --config {args.config} --num_samples 3"
        success = run_command(cmd, "Dataset Validation")
        if not success:
            print("\n⚠️  Dataset validation failed. Please fix dataset issues before continuing.")
            return 1
    
    # Step 2: Check for cached patches
    cache_dir = Path(big_data_storage_path) / "processed" / task_name / "patches"
    
    print("\nChecking for existing cached patches...")
    print(f"Cache directory: {cache_dir}")
    
    has_cached_patches = cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 0
    
    if not has_cached_patches:
        print("\nCached patches not found.")
        print("Starting patch preparation step...")
        cmd = f"python prepare_patches.py --config {args.config}"
        success = run_command(cmd, "Create Patches")
        if not success:
            print("\n⚠️  Patch preparation failed. Check error messages above.")
            return 1
    else:
        print(f"✓ Found {len(list(cache_dir.glob('*.pt')))} cached patches\n")
    
    # Step 3: Submit VAE training jobs for all groups
    # VAE sbatch scripts should chain to diffusion training when complete
    vae_groups = config.get('vae_groups', {})
    diffusion_stages = config.get('diffusion_stages', {})
    
    print("\n" + "="*60)
    print(f"SUBMITTING VAE TRAINING JOBS ({len(vae_groups)} groups)")
    print("="*60)
    print("Note: VAE sbatch scripts will automatically trigger diffusion training upon completion")
    
    for group_name in vae_groups.keys():
        mode_str = f"vae:{group_name}"
        
        if mode_str in skip_modes:
            print(f"\n⚠️  Skipping VAE training for '{group_name}' (user requested)")
            
            # Check if we should submit diffusion for this group instead
            corresponding_diffusion = None
            for stage_name, stage_config in diffusion_stages.items():
                pred_group = stage_config.get('prediction_group')
                if pred_group == group_name:
                    corresponding_diffusion = stage_name
                    break
            
            if corresponding_diffusion:
                diffusion_mode_str = f"diffusion:{corresponding_diffusion}"
                if diffusion_mode_str not in skip_modes:
                    print(f"    → Submitting Diffusion training for '{corresponding_diffusion}' instead")
                    
                    sbatch_script = find_sbatch_script('diffusion_inpainting', corresponding_diffusion, script_dir)
                    
                    if sbatch_script is None:
                        print(f"\n⚠️  Sbatch script not found for Diffusion stage '{corresponding_diffusion}'")
                        print(f"    Expected: {script_dir / f'train_{corresponding_diffusion}_diffusion_inpainting_ddp.sh'}")
                        print(f"    Skipping...")
                        continue
                    
                    cmd = f"sbatch {sbatch_script} --config {args.config}"
                    success = run_command(cmd, f"Diffusion Training: {corresponding_diffusion}")
                    
                    if not success:
                        print(f"\n⚠️  Diffusion training submission failed for '{corresponding_diffusion}'")
                        return 1
            continue
        
        # Find sbatch script
        sbatch_script = find_sbatch_script('vae', group_name, script_dir)
        
        if sbatch_script is None:
            print(f"\n⚠️  Sbatch script not found for VAE group '{group_name}'")
            print(f"    Expected: {script_dir / f'train_{group_name}_vae_ddp.sh'}")
            print(f"    Skipping this group...")
            continue
        
        # Submit VAE job (will chain to diffusion in sbatch script)
        cmd = f"sbatch {sbatch_script} --config {args.config}"
        success = run_command(cmd, f"VAE Training: {group_name}")
        
        if not success:
            print(f"\n⚠️  VAE training submission failed for '{group_name}'")
            return 1
    
    # Success!
    print("\n" + "="*60)
    print("🎉 PIPELINE SUBMITTED TO CLUSTER SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults will be saved to:")
    print(f"  {Path(big_data_storage_path) / 'results' / task_name}")
    print("\nMonitor job status with: squeue -u $USER")
    print("\nTo generate samples after training:")
    print(f"  python sample_urban_inpainting.py --config {args.config} --num_samples 16")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

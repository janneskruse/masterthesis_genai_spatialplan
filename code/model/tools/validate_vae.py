# Validation script for VAE training
# Creates reconstruction samples on validation set to assess VAE quality

###### import libraries ######
# Standard libraries
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Data handling
import torch
from torchvision.utils import make_grid, save_image

# Local libraries
from model.dataset.dataset import UrbanInpaintingDataset
from model.utils.vae_registry import VAERegistry
from model.utils.vae_utils import save_vae_reconstruction_samples
from model.utils.layer_config import count_layer_channels, get_layer_info
from helpers.load_configs import load_configs
from helpers.indexed_outputs import get_next_run_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_vae(
    vae_registry: VAERegistry,
    dataset,
    vae_group: str,
    vae_config,
    layers_registry,
    num_samples=8,
    save_dir=None,
    overwrite_samples=False
):
    """
    Validate VAE by creating reconstruction samples on validation set.
    
    Args:
        vae_registry: VAERegistry with loaded VAEs
        dataset: UrbanInpaintingDataset in validation mode
        vae_group: VAE group name (e.g., 'satellite', 'semantic', 'environmental')
        vae_config: VAE configuration dict
        layers_registry: Global layers registry
        num_samples: Number of validation samples to process
        save_dir: Directory to save samples
        overwrite_samples: Whether to overwrite existing samples
        
    Returns:
        None
    """
    
    # Get VAE from registry
    vae = vae_registry.get_vae(vae_group)
    
    if vae is None:
        print(f"\n✗ VAE group '{vae_group}' not yet trained")
        print(f"  No checkpoint found for this VAE group")
        print(f"  Please train the VAE first before validation")
        return
    
    vae.eval()
    
    # Get layers for this group
    group_layers = vae_config.get('layers', [])
    
    # Count total channels
    num_input_channels = 0
    for layer_name in group_layers:
        layer_config = get_layer_info(layers_registry, layer_name)
        num_channels = count_layer_channels(layer_config)
        num_input_channels += num_channels
    
    print("\n" + "="*60)
    print(f"VAE Validation: {vae_group.upper()}")
    print("="*60)
    print(f"VAE group: {vae_group}")
    print(f"Layers: {group_layers}")
    print(f"Total channels: {num_input_channels}")
    print(f"Number of samples: {num_samples}")
    print(f"Split: validation")
    
    # Limit number of samples to dataset size
    num_samples = min(num_samples, len(dataset))
    
    # Get random validation samples
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"\n✓ Selected {num_samples} random validation samples")
    print(f"  Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    
    # Setup output directories
    if save_dir is None:
        save_dir = f"./results/{vae_group}_validation"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get next run index
    base_name = f'{vae_group}_val'
    run_idx = get_next_run_idx(save_dir, base_name)
    if overwrite_samples and run_idx > 0:
        run_idx = 0
    
    print(f"\n{'='*60}")
    print(f"Output Run Index: {run_idx}")
    print(f"{'='*60}")
    
    # Create samples directory
    # samples_dir = os.path.join(save_dir, f'{base_name}_idx{run_idx}_samples')
    samples_dir = save_dir
    os.makedirs(samples_dir, exist_ok=True)
    
    ################# Validation Loop ########################
    print("\n" + "="*60)
    print("Starting VAE Validation")
    print("="*60)
    
    all_inputs = []
    all_recons = []
    
    with torch.no_grad():
        for sample_idx, data_idx in enumerate(tqdm(sample_indices, desc="Validating")):
            # Load sample from dataset
            data = dataset[data_idx]
            
            # Extract data
            if len(data) == 2:
                input_tensor, meta_dict = data
                # Extract metadata
                meta = meta_dict.get('meta', {})
                if isinstance(meta, dict):
                    channel_names = meta.get('channel_names', [])
                    layer_names = meta.get('layer_names', [])
                else:
                    channel_names = []
                    layer_names = []
            else:
                input_tensor = data
                channel_names = []
                layer_names = []
            
            # Move to device and add batch dimension
            input_tensor = input_tensor.unsqueeze(0).float().to(device)
            
            # Validate channel count
            if input_tensor.shape[1] != num_input_channels:
                print(f"\n⚠ Warning: Expected {num_input_channels} channels but got {input_tensor.shape[1]}")
                continue
            
            # Forward pass through VAE
            recon, z, mean, logvar = vae(input_tensor)
            
            # Save individual sample
            sample_pt_path = os.path.join(samples_dir, f'sample_{sample_idx}.pt')
            torch.save({
                'input': input_tensor[0].cpu(),
                'reconstruction': recon[0].cpu(),
                'latent': z[0].cpu(),
                'channel_names': channel_names,
                'layer_names': layer_names,
                'data_index': data_idx,
            }, sample_pt_path)
            
            # Keep for visualization
            all_inputs.append(input_tensor)
            all_recons.append(recon)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Stack all samples
    input_batch = torch.cat(all_inputs, dim=0)  # [N, C, H, W]
    recon_batch = torch.cat(all_recons, dim=0)  # [N, C, H, W]
    
    # Save reconstruction samples using same utility as training
    if channel_names and layer_names:
        save_vae_reconstruction_samples(
            input_tensor=input_batch,
            recon_tensor=recon_batch,
            channel_names=channel_names,
            layer_names=layer_names,
            layers_registry=layers_registry,
            save_dir=save_dir,
            step=run_idx,
            n_samples=num_samples,
            save_rgb_composite=True
        )
        print(f"✓ Saved reconstruction visualizations to {save_dir}")
    else:
        print("⚠ No channel/layer names available, skipping detailed visualizations")
    
    print(f"\n{'='*60}")
    print(f"✓ Validation Complete!")
    print(f"  Processed {len(sample_indices)} validation samples")
    print(f"  Saved to: {samples_dir}")
    print(f"{'='*60}")


def main(args, config):
    """Main validation function"""
    
    # Extract configs
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    vae_groups = config['vae_groups']
    layers_registry = config.get('layers', {})
    
    big_data_storage_path = dataset_config.get('big_data_storage_path', '/work/zt75vipu-thesis/data')
    task_name = train_config.get('task_name', 'urban_inpainting')
    
    # Determine which VAE group to validate
    vae_group = args.mode
    
    if vae_group not in vae_groups:
        print(f"\n✗ VAE group '{vae_group}' not found in config")
        print(f"  Available groups: {list(vae_groups.keys())}")
        return
    
    vae_config = vae_groups[vae_group]
    
    print(f"\n{'='*60}")
    print(f"VAE Validation Setup")
    print(f"{'='*60}")
    print(f"VAE group: {vae_group}")
    print(f"Task: {task_name}")
    print(f"Number of samples: {args.num_samples}")
    
    # Set seed for reproducibility
    seed = train_config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"✓ Set random seed: {seed}")
    
    ########## Load Dataset #############
    print("\n" + "="*60)
    print("Loading Validation Dataset")
    print("="*60)
    
    # Load dataset in VAE mode for validation split
    dataset = UrbanInpaintingDataset(
        split='val',
        mode=f'vae:{vae_group}',
        use_cached_patches=True,
    )
    
    print(f"✓ Loaded {len(dataset)} validation patches")
    
    ########## Load VAE #############
    print("\n" + "="*60)
    print("Loading VAE Model")
    print("="*60)
    
    # Initialize VAE Registry
    vae_registry = VAERegistry(config, device)
    
    # Load VAE checkpoint
    data_dir = f"{big_data_storage_path}/results/{task_name}"
    vae_checkpoint = vae_config.get('checkpoint_name', f'{vae_group}_vae_ckpt.pth')
    vae_path = os.path.join(data_dir, vae_checkpoint)
    
    print(f"VAE checkpoint path: {vae_path}")
    
    # Try to load VAE
    vae_registry.load_vae(
        group_name=vae_group,
        checkpoint_path=vae_path,
        autoencoder_config=vae_config,
    )
    
    # Setup output directory
    repo_dir = config.get('repo_dir', '.')
    save_dir = f"{repo_dir}/results/{task_name}/{vae_group}_validation"
    
    ########## Validate VAE #############
    validate_vae(
        vae_registry=vae_registry,
        dataset=dataset,
        vae_group=vae_group,
        vae_config=vae_config,
        layers_registry=layers_registry,
        num_samples=args.num_samples,
        save_dir=save_dir,
        overwrite_samples=args.overwrite_samples
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate VAE training with reconstruction samples')
    parser.add_argument('--mode', type=str, required=True,
                       help='VAE group to validate (e.g., satellite, semantic, environmental)')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of validation samples to process')
    parser.add_argument('--overwrite_samples', action='store_true',
                       help='Overwrite existing validation samples (use run_idx=0)')
    parser.add_argument('--config', type=str, default='two_stage_4.yml',
                       help='Config file name')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        os.environ['CONFIG_PATH'] = args.config
    
    config = load_configs()
    
    main(args, config)

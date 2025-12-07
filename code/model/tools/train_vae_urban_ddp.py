# Training script for VAE on urban satellite imagery

###### import libraries ######
# system libraries
import os
import yaml
import re
import time
from pathlib import Path

# data and visualization
import numpy as np
from tqdm import tqdm

# ML libraries
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# custom modules
from model.dataset.dataset import UrbanInpaintingDataset
from model.diffusion_blocks.vae import VAE
from model.diffusion_blocks.discriminator import Discriminator
from model.diffusion_blocks.lpips import LPIPS
from model.utils.data_utils import collate_fn
from model.utils.load_cuda import load_cuda
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########## Distributed Setup #############
def setup_distributed():
    """Initialize distributed training with SLURM environment variables"""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
        
        # Set device BEFORE any distributed operations
        torch.cuda.set_device(local_rank)
        
        # MASTER_ADDR and MASTER_PORT are set by the bash script
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        
        # Verify IPv4 format
        if not re.match(r'^\d+\.\d+\.\d+\.\d+$', master_addr):
            raise ValueError(f"MASTER_ADDR must be IPv4 format, got: {master_addr}")
        
        if rank == 0:
            print(f"✓ Master node: {master_addr}:{master_port} (IPv4)")
            print(f"✓ World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        
        # Force IPv4 socket family
        os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # init
        dist.init_process_group('nccl')
        
        # Verify initialization
        if rank == 0:
            print(f"✓ Distributed initialization successful")
            print(f"✓ Backend: {dist.get_backend()}")
            print(f"✓ World size from DDP: {dist.get_world_size()}")
            print(f"✓ Rank from DDP: {dist.get_rank()}")
        
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        
        
def save_latents_distributed(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    latent_dir: Path,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> int:
    """
    Save latent encodings from VAE in distributed setting.
    
    Each rank processes a disjoint subset of the dataset to avoid duplicates.
    Uses deterministic indexing to ensure consistent global indices across runs.
    
    Args:
        model: Trained VAE model (wrapped in DDP)
        dataset: Full training dataset
        latent_dir: Directory to save latent .pt files
        batch_size: Batch size for encoding
        rank: Current process rank
        world_size: Total number of processes
        device: Device for computation
        
    Returns:
        Number of latents saved by this rank
    """
    # Unwrap DDP model for inference
    model_unwrapped = model.module if hasattr(model, 'module') else model
    model_unwrapped.eval()
    
    # Create latent directory (only rank 0)
    if rank == 0:
        latent_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()  # Wait for directory creation
    
    # Calculate this rank's data indices
    total_samples = len(dataset)
    samples_per_rank = (total_samples + world_size - 1) // world_size  # Ceiling division
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Encoding and Saving Latents (Distributed) at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Samples per rank: {samples_per_rank}")
        print(f"World size: {world_size}")
    
    print(f"Rank {rank}: Processing indices {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    
    # Create subset of dataset for this rank
    rank_indices = list(range(start_idx, end_idx))
    rank_dataset = torch.utils.data.Subset(dataset, rank_indices)
    
    # Create dataloader for this rank's subset
    rank_loader = DataLoader(
        rank_dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain deterministic order
        num_workers=0,  # Avoid multiprocessing issues in DDP
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Encode and save latents
    latent_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(
            rank_loader,
            desc=f"Rank {rank} encoding",
            disable=(rank != 0),  # Only show progress on main rank
            unit="batch"
        )
        
        for batch_idx, data in enumerate(progress_bar):
            # Handle different data formats
            if len(data) == 2:
                im, _ = data
            else:
                im = data
            
            im = im.float().to(device)
            
            # Encode to latent space
            _, z, _, _ = model_unwrapped(im)
            
            # Save each latent with global index
            for i in range(z.shape[0]):
                # Calculate global index for this sample
                global_idx = start_idx + batch_idx * batch_size + i
                
                # Ensure we don't exceed dataset bounds
                if global_idx >= end_idx or global_idx >= total_samples:
                    break
                
                # Save latent to disk
                latent_path = latent_dir / f'latent_{global_idx}.pt'
                torch.save(z[i].cpu(), latent_path)
                latent_count += 1
    
    # Synchronize all ranks
    if world_size > 1:
        dist.barrier()
    
    # Verify completeness (rank 0 only)
    if rank == 0:
        saved_latents = sorted([
            int(f.stem.split('_')[1]) 
            for f in latent_dir.glob('latent_*.pt')
        ])
        
        expected_latents = list(range(total_samples))
        missing_latents = set(expected_latents) - set(saved_latents)
        duplicate_latents = len(saved_latents) - len(set(saved_latents))
        
        print(f"\n{'='*60}")
        print(f"✓ Total latents saved: {len(saved_latents)}/{total_samples}")
        
        if missing_latents:
            print(f"⚠ Missing latents: {sorted(missing_latents)[:10]}{'...' if len(missing_latents) > 10 else ''}")
        
        if duplicate_latents > 0:
            print(f"⚠ Duplicate latents found: {duplicate_latents}")
        
        if len(saved_latents) == total_samples and not missing_latents and duplicate_latents == 0:
            print(f"✓ All latents saved successfully!")
        
        print(f"{'='*60}\n")
    
    return latent_count

########## Main Training Function #############
def train_vae():
    # Record training start time
    training_start_time = time.time()
    
    ###### setup config variables #######
    config = load_configs()
    # repo_dir = config['repo_dir']
    data_config = config['data_config']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    
    if is_main:
        print(f"\n{'='*50}")
        print(f"Distributed Training Setup at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"{'='*50}")
        print(f"✓ World size: {world_size}")
        print(f"✓ Rank: {rank}")
        print(f"✓ Local rank: {local_rank}")
        
    if is_main:
        print("="*50)
        print("VAE Training Configuration")
        print("="*50)
        print(yaml.dump(config, default_flow_style=False))
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create output directories
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    latent_dir = os.path.join(out_dir, 'vae_ddp_latents')
    samples_dir = os.path.join(out_dir, 'vae_ddp_samples')
    
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
    
    
    batch_size = train_config['autoencoder_batch_size']
    num_gpus = 1
    if device.type == 'cuda':
        # original_batch_size = batch_size
        num_gpus = torch.cuda.device_count()
        if is_main:
            print(f"✓ Available GPUs: {num_gpus}")
        
        # # Adjust batch size for multi-GPU
        # if num_gpus > 1:
        #     batch_size = batch_size * num_gpus
        #     print(f"✓ Scaling batch size: {original_batch_size} → {batch_size}")
    
    
    ########## Load Dataset #############
    if is_main:
        print(f"\n{'='*50}")
        print("Loading Urban Dataset")
        print('='*50)
    
    # For VAE training, we don't use latents and don't need conditioning
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        use_latents=False,
        latent_path=None
    )
    
    if is_main:
        print(f"✓ Loaded {len(urban_dataset)} training patches")
        print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
        print(f"✓ Image channels: {dataset_config['im_channels']}")
    
    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(
        urban_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=0, #4
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    ########## Create Models #############
    if is_main:
        print(f"\n{'='*50}")
        print("Initializing Models")
        print('='*50)
    
    # VAE model
    model = VAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_config
    ).to(device)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if is_main:
            print(f"✓ Wrapped model in DistributedDataParallel")
    
    if is_main:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Created VAE with {param_count:.2f}M parameters")
        print(f"  - Latent channels: {autoencoder_config['z_channels']}")
        print(f"  - Downsampling factor: {2 ** sum(autoencoder_config['down_sample'])}")
    
    # Discriminator for adversarial loss
    discriminator = Discriminator(
        im_channels=dataset_config['im_channels']
    ).to(device)
    
    # if world_size > 1:
    #     discriminator = DDP(
    #         discriminator,
    #         device_ids=[local_rank],
    #         output_device=local_rank,
    #         find_unused_parameters=False
    #     )
    #     if is_main:
    #         print(f"✓ Wrapped discriminator in DistributedDataParallel")
    
    # No wrap of the discriminator in DDP – each rank has its own copy
    if is_main and world_size > 1:
        print("✓ Using per-rank discriminator (no DDP wrapper)")
    
    if is_main:
        disc_params = sum(p.numel() for p in discriminator.parameters()) / 1e6
        print(f"✓ Created Discriminator with {disc_params:.2f}M parameters")
    
    # LPIPS perceptual loss
    lpips_model = LPIPS().eval().to(device)
    if is_main:
        print("✓ Created LPIPS perceptual loss model")
    
    ########## Training Setup #############
    num_epochs = train_config['autoencoder_epochs']
    
    # Scale learning rate with world size (linear scaling rule)
    base_lr = train_config['autoencoder_lr']
    adjusted_lr = base_lr * world_size
    if is_main and world_size > 1:
        print(f"✓ Adjusted learning rate: {base_lr} → {adjusted_lr}")
    
    optimizer_vae = Adam(model.parameters(), lr=adjusted_lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=adjusted_lr)
    
    # Loss weights
    kl_weight = train_config.get('kl_weight', 0.000001)
    perceptual_weight = train_config.get('perceptual_weight', 1.0)
    disc_weight = train_config.get('disc_weight', 0.5)
    disc_start_epoch = train_config.get('disc_start', 10000) // len(data_loader)  # Convert steps to epochs
    
    if is_main:
        print(f"\n✓ Training for {num_epochs} epochs")
        print(f"✓ Learning rate: {train_config['autoencoder_lr']}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ KL weight: {kl_weight}")
        print(f"✓ Perceptual weight: {perceptual_weight}")
        print(f"✓ Discriminator weight: {disc_weight} (starting epoch {disc_start_epoch})")
    
    ########## Training Loop #############
    if is_main:
        print("\n" + "="*50)
        print(f"Starting Training with {num_epochs} epochs")
        print("="*50)
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if world_size > 1:
            sampler.set_epoch(epoch_idx)
        
        losses_vae = []
        losses_disc = []
        
        if is_main:
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        else:
            progress_bar = data_loader
        
        for batch_idx, data in enumerate(progress_bar):
            # Unpack data (ignore conditioning for VAE training)
            if len(data) == 2:
                im, _ = data
            else:
                im = data
            
            im = im.float().to(device)
            
            ########## Train VAE ##########
            
            ############################
            # 1) VAE / Generator step
            ############################

            # Freeze discriminator params for generator step
            for p in discriminator.parameters():
                p.requires_grad = False
            
            optimizer_vae.zero_grad()
            
            # Forward pass
            im_recon, z, mean, logvar = model(im)
            
            # Reconstruction loss (L1)
            recon_loss = torch.abs(im - im_recon).mean()
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / (im.shape[0] * im.shape[2] * im.shape[3])  # Normalize
            
            # Perceptual loss (LPIPS)
            perceptual_loss = lpips_model(im, im_recon).mean()
            
            # Generator loss (fool discriminator)
            if epoch_idx >= disc_start_epoch:
                disc_fake = discriminator(im_recon)
                gen_loss = -torch.mean(disc_fake)
            else:
                gen_loss = torch.tensor(0.0).to(device)
            
            # Total VAE loss
            vae_loss = (recon_loss + 
                       kl_weight * kl_loss + 
                       perceptual_weight * perceptual_loss +
                       disc_weight * gen_loss)
            
            vae_loss.backward()
            optimizer_vae.step()
            
            ############################
            # 2) Discriminator step    #
            ############################
            if epoch_idx >= disc_start_epoch:
                # Unfreeze discriminator params
                for p in discriminator.parameters():
                    p.requires_grad = True
                    
                optimizer_disc.zero_grad()
                
                # Important: .detach() inputs so D doesn't backprop into VAE
                # Discriminator on real images
                disc_real = discriminator(im.detach())
                
                # Discriminator on fake images
                disc_fake = discriminator(im_recon.detach())
                
                # Discriminator loss (hinge loss) --> prevents discriminator from being too "strong"
                disc_loss = torch.mean(torch.relu(1.0 - disc_real)) + torch.mean(torch.relu(1.0 + disc_fake))
                
                disc_loss.backward()
                optimizer_disc.step()
                
                losses_disc.append(disc_loss.item())
            
            losses_vae.append(vae_loss.item())
            global_step += 1
            
            # Update progress bar (main process only)
            if is_main:
                if epoch_idx >= disc_start_epoch:
                    progress_bar.set_postfix({
                        'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}',
                        'disc_loss': f'{np.mean(losses_disc[-100:]):.4f}'
                    })
                else:
                    progress_bar.set_postfix({'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}'})
            
            # Save sample reconstructions (main process only)
            if is_main and global_step % train_config.get('autoencoder_img_save_steps', 500) == 0:
                with torch.no_grad():
                    # Take first 8 images
                    sample_im = im[:8]
                    sample_recon, _, _, _ = model(sample_im)
                    
                    # Normalize to [0, 1]
                    sample_im = torch.clamp(sample_im, -1., 1.)
                    sample_im = (sample_im + 1) / 2
                    sample_recon = torch.clamp(sample_recon, -1., 1.)
                    sample_recon = (sample_recon + 1) / 2
                    
                    # Create comparison grid
                    comparison = torch.cat([sample_im, sample_recon], dim=0)
                    grid = make_grid(comparison, nrow=8, padding=2, pad_value=1.0)
                    
                    save_path = os.path.join(
                        samples_dir,
                        f'recon_step_{global_step}.png'
                    )
                    save_image(grid, save_path)
        
        # Save checkpoint (main process only)
        if is_main:
            # Get model without DDP wrapper
            model_to_save = model.module if hasattr(model, 'module') else model
            
            checkpoint_path = os.path.join(
                out_dir,
                train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth')
            )
            torch.save(model_to_save.state_dict(), checkpoint_path)
            
            if (epoch_idx + 1) % 10 == 0:
                periodic_path = os.path.join(out_dir, f'vae_urban_ddp_epoch_{epoch_idx + 1}.pth')
                torch.save(model_to_save.state_dict(), periodic_path)
                print(f'✓ Saved checkpoint: {periodic_path}')
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
    ########## Save Latents ##########
    if train_config.get('save_latents', True):
        latent_dir_name = train_config.get('latents_dir_name', 'vae_ddp_latents')
        latent_dir = Path(big_data_storage_path) / "results" / train_config['task_name'] / latent_dir_name
        
        # Save latents in distributed manner
        latent_count = save_latents_distributed(
            model=model,
            dataset=urban_dataset,
            latent_dir=latent_dir,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        
        print(f"Rank {rank}: Saved {latent_count} latents")
        
        # Save dataset statistics (main rank only)
        if is_main:
            stats_path = urban_dataset.save_stats(f"{out_dir}/vae_ddp_stats")
            print(f"✓ Saved dataset statistics to {stats_path}")
    
    
    # if is_main and train_config.get('save_latents', True):
    #     print("\n" + "="*50)
    #     print("Encoding and Saving Latents")
    #     print("="*50)
        
    #     model.eval()
        
    #     with torch.no_grad():
    #         for idx, data in enumerate(tqdm(data_loader, desc='Encoding latents')):
    #             if len(data) == 2:
    #                 im, _ = data
    #             else:
    #                 im = data
                
    #             im = im.float().to(device)
    #             _, z, _, _ = model(im)
                
    #             # Save each latent
    #             print("Shape of z:", z.shape)
    #             for i in range(z.shape[0]):
    #                 global_idx = idx * batch_size + i
    #                 latent_path = os.path.join(latent_dir, f'latent_{global_idx}.pt')
    #                 torch.save(z[i].cpu(), latent_path)
        
    #     print(f"✓ Saved {len(urban_dataset)} latents to {latent_dir}")
    
    #     # save statistics about inpainting masks
    #     stats_path = urban_dataset.save_stats(f"{out_dir}/vae_ddp_stats")
    #     print(f"✓ Saved dataset statistics to {stats_path}")
    
    if is_main:
        # Calculate total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)
        
        print(f"\n{'='*50}")
        print(f'✓ VAE Training Complete at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} !')
        print(f'✓ Total Training Time: {hours}h {minutes}m {seconds}s ({total_training_time:.2f} seconds)')
        print('='*50)
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


if __name__ == '__main__':
    train_vae()

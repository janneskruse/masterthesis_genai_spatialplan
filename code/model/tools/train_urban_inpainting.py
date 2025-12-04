# Training script for urban inpainting latent diffusion model

###### import libraries ######
# system libraries
import sys
import os
import yaml
from tqdm import tqdm

# data science libraries
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as torchF

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local libraries
from dataset.dataset import UrbanInpaintingDataset
from diffusion_blocks.unet_cond_base import Unet
from diffusion_blocks.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value
from utils.data_utils import collate_fn
from utils.load_cuda import load_cuda
from helpers.load_configs import load_configs

# Load CUDA
load_cuda()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_inpainting_loss(noise_pred, noise, mask_latent, mask_loss_weight=2.0):
    """
    Compute loss with higher weight in masked region.
    
    Args:
        noise_pred: Predicted noise [B, C, H, W]
        noise: Ground truth noise [B, C, H, W]
        mask_latent: Mask in latent space [B, 1, H, W], 1=regenerate, 0=keep
        mask_loss_weight: Weight multiplier for masked region
    
    Returns:
        Weighted MSE loss
    """
    # Basic MSE loss
    loss = torchF.mse_loss(noise_pred, noise, reduction='none')
    
    # Apply mask weighting: higher weight where mask==1
    mask_weight = 1.0 + (mask_loss_weight - 1.0) * mask_latent
    weighted_loss = loss * mask_weight
    
    return weighted_loss.mean()


def train():
    
    ###### setup config variables #######
    config = load_configs()
    # repo_dir = config['repo_dir']
    data_config = config['data_config']

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
        
    
    print("="*50)
    print("Urban Inpainting Training Configuration")
    print("="*50)
    print(yaml.dump(config, default_flow_style=False))
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create output directory
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    print(f"\n✓ Created noise scheduler with {diffusion_config['num_timesteps']} timesteps")
    
    ########## Load Dataset #############
    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    assert condition_config is not None, "Condition config required for urban inpainting"
    
    print("\n" + "="*50)
    print("Loading Urban Dataset")
    print("="*50)
    
    # Check if latents exist
    # latent_path = os.path.join(train_config['task_name'], 
    #                            train_config.get('vqvae_latent_dir_name', 'vqvae_latents'))
    latent_path = f'{big_data_storage_path}/results/{train_config["task_name"]}/{train_config.get("latents_dir_name", "vae_ddp_latents")}'
    use_latents = os.path.exists(latent_path) and len(os.listdir(latent_path)) > 0
    
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        use_latents=use_latents,
        latent_path=latent_path if use_latents else None
    )
    
    print(f"✓ Loaded {len(urban_dataset)} training patches")
    print(f"✓ Using latents: {use_latents}")
    print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
    print(f"✓ Conditioning types: {condition_config['condition_types']}")
    
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=train_config['ldm_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    ########## Create Model #############
    print("\n" + "="*50)
    print("Initializing Models")
    print("="*50)
    
    # Instantiate the U-Net model
    # Input channels: 4 (noisy latent) for now
    # We'll handle spatial conditioning via image_cond
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    model.train()
    
    print(f"✓ Created U-Net with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Load VAE if not using latents
    vae = None
    if not use_latents:
        print('\n⚠ Loading VAE model as latents not present')
        vae = VAE(
            im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config
        ).to(device)
        vae.eval()
        
        # Load VAE checkpoint if exists
        vae_path = os.path.join(
            train_config['task_name'],
            train_config.get('autoencoder_ckpt_name', 'vae_urban_ddp_ckpt.pth')
        )
        if os.path.exists(vae_path):
            print(f'✓ Loaded VAE checkpoint from {vae_path}')
            vae.load_state_dict(torch.load(vae_path, map_location=device))
        else:
            raise Exception(f'VAE checkpoint not found at {vae_path}. Please train VAE first.')
        
        # Freeze VAE
        for param in vae.parameters():
            param.requires_grad = False
    
    ########## Training Setup #############
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    
    # Loss weights
    mask_loss_weight = train_config.get('mask_loss_weight', 2.0)
    
    # Conditioning dropout probability
    cond_drop_prob = get_config_value(
        condition_config.get('image_condition_config', {}),
        'cond_drop_prob',
        0.1
    )
    
    print(f"\n✓ Training for {num_epochs} epochs")
    print(f"✓ Learning rate: {train_config['ldm_lr']}")
    print(f"✓ Batch size: {train_config['ldm_batch_size']}")
    print(f"✓ Mask loss weight: {mask_loss_weight}")
    print(f"✓ Conditioning dropout: {cond_drop_prob}")
    
    ########## Training Loop #############
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch_idx in range(num_epochs):
        losses = []
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        
        for batch_idx, data in enumerate(progress_bar):
            # Unpack data
            if len(data) == 2:
                im, cond_input = data
            else:
                im = data
                cond_input = None
            
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Encode to latent space if not using pre-computed latents
            if not use_latents and vae is not None:
                with torch.no_grad():
                    im, _ = vae.encode(im)
            
            ########## Handle Conditional Input ##########
            if cond_input is not None:
                # Move all tensors in cond_input to device
                for key in cond_input:
                    if key == 'meta':
                        continue  # Skip metadata
                    if isinstance(cond_input[key], torch.Tensor):
                        cond_input[key] = cond_input[key].to(device)
                
                # Apply conditioning dropout for classifier-free guidance
                if 'image' in cond_input and np.random.rand() < cond_drop_prob:
                    # Replace with zeros (unconditional)
                    cond_input['image'] = torch.zeros_like(cond_input['image'])
                
                # Get mask for loss weighting
                # Extract mask from spatial conditions if available
                if 'mask' in cond_input:
                    mask = cond_input['mask'].to(device)
                    # Downsample mask to latent resolution
                    mask_latent = torchF.interpolate(
                        mask,
                        size=im.shape[-2:],
                        mode='nearest'
                    )
                else:
                    # No explicit mask, use uniform weighting
                    mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            else:
                mask_latent = torch.ones((im.shape[0], 1, im.shape[2], im.shape[3])).to(device)
            
            ########## Diffusion Forward Process ##########
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample random timestep
            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (im.shape[0],)
            ).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            
            # Predict noise
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            
            # Compute weighted loss
            loss = compute_inpainting_loss(
                noise_pred,
                noise,
                mask_latent,
                mask_loss_weight
            )
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})
        
        # Epoch summary
        epoch_loss = np.mean(losses)
        print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            train_config['task_name'],
            train_config.get('ldm_ckpt_name', 'ddpm_urban_inpainting_ckpt.pth')
        )
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch_idx + 1) % 10 == 0:
            periodic_path = os.path.join(
                train_config['task_name'],
                f'ddpm_urban_inpainting_epoch_{epoch_idx + 1}.pth'
            )
            torch.save(model.state_dict(), periodic_path)
            print(f'✓ Saved checkpoint: {periodic_path}')
    
    print('\n' + "="*50)
    print('✓ Training Complete!')
    print("="*50)


if __name__ == '__main__':
    train()

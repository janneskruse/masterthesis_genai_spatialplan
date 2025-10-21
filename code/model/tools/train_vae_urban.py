# Training script for VAE on urban satellite imagery
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import UrbanInpaintingDataset
from diffusion_blocks.vae import VAE
from diffusion_blocks.discriminator import Discriminator
from diffusion_blocks.lpips import LPIPS
from utils.data_utils import collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_vae():

    ###### setup config variables #######
    repo_name = 'masterthesis_genai_spatialplan'
    if not repo_name in os.getcwd():
        os.chdir(repo_name)

    p=os.popen('git rev-parse --show-toplevel')
    repo_dir = p.read().strip()
    p.close()

    with open(f"{repo_dir}/code/model/config/class_cond.yml", 'r') as stream:
        config = yaml.safe_load(stream)
        
    with open(f"{repo_dir}/config.yml", 'r') as stream:
        data_config = yaml.safe_load(stream)

    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    print("="*50)
    print("VAE Training Configuration")
    print("="*50)
    print(yaml.dump(config, default_flow_style=False))
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create output directories
    out_dir = f"{big_data_storage_path}/results/{train_config.get('task_name', 'urban_inpainting')}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vae_samples'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vqvae_latents'), exist_ok=True)
    
    ########## Load Dataset #############
    print("\n" + "="*50)
    print("Loading Urban Dataset")
    print("="*50)
    
    # For VAE training, we don't use latents and don't need conditioning
    urban_dataset = UrbanInpaintingDataset(
        split='train',
        use_latents=False,
        latent_path=None
    )
    
    print(f"✓ Loaded {len(urban_dataset)} training patches")
    print(f"✓ Patch size: {urban_dataset.patch_size}x{urban_dataset.patch_size}")
    print(f"✓ Image channels: {dataset_config['im_channels']}")
    
    data_loader = DataLoader(
        urban_dataset,
        batch_size=train_config['autoencoder_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    ########## Create Models #############
    print("\n" + "="*50)
    print("Initializing Models")
    print("="*50)
    
    # VAE model
    model = VAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_config
    ).to(device)
    model.train()
    
    print(f"✓ Created VAE with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"  - Latent channels: {autoencoder_config['z_channels']}")
    print(f"  - Downsampling factor: {2 ** sum(autoencoder_config['down_sample'])}")
    
    # Discriminator for adversarial loss
    discriminator = Discriminator(
        im_channels=dataset_config['im_channels']
    ).to(device)
    discriminator.train()
    
    print(f"✓ Created Discriminator with {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M parameters")
    
    # LPIPS perceptual loss
    lpips_model = LPIPS().eval().to(device)
    print("✓ Created LPIPS perceptual loss model")
    
    ########## Training Setup #############
    num_epochs = train_config['autoencoder_epochs']
    optimizer_vae = Adam(model.parameters(), lr=train_config['autoencoder_lr'])
    optimizer_disc = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'])
    
    # Loss weights
    kl_weight = train_config.get('kl_weight', 0.000001)
    perceptual_weight = train_config.get('perceptual_weight', 1.0)
    disc_weight = train_config.get('disc_weight', 0.5)
    disc_start_epoch = train_config.get('disc_start', 10000) // len(data_loader)  # Convert steps to epochs
    
    print(f"\n✓ Training for {num_epochs} epochs")
    print(f"✓ Learning rate: {train_config['autoencoder_lr']}")
    print(f"✓ Batch size: {train_config['autoencoder_batch_size']}")
    print(f"✓ KL weight: {kl_weight}")
    print(f"✓ Perceptual weight: {perceptual_weight}")
    print(f"✓ Discriminator weight: {disc_weight} (starting epoch {disc_start_epoch})")
    
    ########## Training Loop #############
    print("\n" + "="*50)
    print(f"Starting Training with {num_epochs} epochs")
    print("="*50)
    
    global_step = 0
    
    for epoch_idx in range(num_epochs):
        losses_vae = []
        losses_disc = []
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}')
        
        for batch_idx, data in enumerate(progress_bar):
            # Unpack data (ignore conditioning for VAE training)
            if len(data) == 2:
                im, _ = data
            else:
                im = data
            
            im = im.float().to(device)
            
            ########## Train VAE ##########
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
            
            ########## Train Discriminator ##########
            if epoch_idx >= disc_start_epoch:
                optimizer_disc.zero_grad()
                
                # Discriminator on real images
                disc_real = discriminator(im.detach())
                
                # Discriminator on fake images
                disc_fake = discriminator(im_recon.detach())
                
                # Discriminator loss (hinge loss)
                disc_loss = torch.mean(torch.relu(1.0 - disc_real)) + torch.mean(torch.relu(1.0 + disc_fake))
                
                disc_loss.backward()
                optimizer_disc.step()
                
                losses_disc.append(disc_loss.item())
            
            losses_vae.append(vae_loss.item())
            global_step += 1
            
            # Update progress bar
            if epoch_idx >= disc_start_epoch:
                progress_bar.set_postfix({
                    'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}',
                    'disc_loss': f'{np.mean(losses_disc[-100:]):.4f}'
                })
            else:
                progress_bar.set_postfix({'vae_loss': f'{np.mean(losses_vae[-100:]):.4f}'})
            
            # Save sample reconstructions
            if global_step % train_config.get('autoencoder_img_save_steps', 500) == 0:
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
                        train_config['task_name'],
                        'vae_samples',
                        f'recon_step_{global_step}.png'
                    )
                    save_image(grid, save_path)
        
        # Epoch summary
        epoch_vae_loss = np.mean(losses_vae)
        if epoch_idx >= disc_start_epoch:
            epoch_disc_loss = np.mean(losses_disc)
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | VAE Loss: {epoch_vae_loss:.4f} | Disc Loss: {epoch_disc_loss:.4f}')
        else:
            print(f'\n✓ Epoch {epoch_idx + 1}/{num_epochs} | VAE Loss: {epoch_vae_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            train_config['task_name'],
            train_config.get('vqvae_autoencoder_ckpt_name', 'vqvae_urban_ckpt.pth')
        )
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch_idx + 1) % 10 == 0:
            periodic_path = os.path.join(
                train_config['task_name'],
                f'vqvae_urban_epoch_{epoch_idx + 1}.pth'
            )
            torch.save(model.state_dict(), periodic_path)
            print(f'✓ Saved checkpoint: {periodic_path}')
    
    ########## Save Latents ##########
    if train_config.get('save_latents', True):
        print("\n" + "="*50)
        print("Encoding and Saving Latents")
        print("="*50)
        
        model.eval()
        latent_dir = os.path.join(train_config['task_name'], 'vqvae_latents')
        
        with torch.no_grad():
            for idx, data in enumerate(tqdm(data_loader, desc='Encoding latents')):
                if len(data) == 2:
                    im, _ = data
                else:
                    im = data
                
                im = im.float().to(device)
                _, z, _, _ = model(im)
                
                # Save each latent
                for i in range(z.shape[0]):
                    global_idx = idx * train_config['autoencoder_batch_size'] + i
                    latent_path = os.path.join(latent_dir, f'latent_{global_idx}.pt')
                    torch.save(z[i].cpu(), latent_path)
        
        print(f"✓ Saved {len(urban_dataset)} latents to {latent_dir}")
    
    print('\n' + "="*50)
    print('✓ VAE Training Complete!')
    print("="*50)


if __name__ == '__main__':
    train_vae()

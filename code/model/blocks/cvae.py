"""
===============================================================================
Conditional Variational Autoencoder (CVAE) for Inpainting.

Extends the base VAE to support:
  - Masked encoder input: encoder sees [x * (1-mask), mask] 
  - Conditioned decoder: decoder receives z concatenated with 
    projected conditioning (mask + environmental latent)
  - Scalar control injection into decoder via additive embedding 
    (injected through t_emb pathway of existing blocks)
  - Weight initialization from pretrained VAE checkpoint

Architecture follows the same patterns as the diffusion UNet conditioning
(see unet_cond_base.py) for consistency.

References:
    Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models.
    Advances in Neural Information Processing Systems (NeurIPS 2015). 
    https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html

    Zheng, C., Cham, T.-J., & Cai, J. (2019). Pluralistic Image Completion. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019). 
    https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pluralistic_Image_Completion_CVPR_2019_paper.html
    
===============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from model.blocks.blocks import DownBlock, MidBlock, UpBlock


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for inpainting that builds on the trained VAE architecture.
    
    Encoder receives masked input concatenated with mask channel:
        input = [x * (1 - mask), mask]  →  (im_channels + 1) input channels
    
    Decoder receives z concatenated with projected conditioning:
        z_cond = [z, cond_proj(cond)]  →  fused via decoder_conv_in
        
    Scalar controls are injected as an additive embedding into the decoder
    blocks via the t_emb pathway (same mechanism as UNet time embedding).
    
    Args:
        im_channels: Number of output/target channels (e.g., 4 for semantic group)
        model_config: VAE architecture config (same as base VAE)
        cond_channels: Total conditioning channels at latent resolution 
                       (e.g., 1 mask + 2 environmental = 3)
        cond_projected_channels: Projected conditioning size after 1x1 conv
                                 (default: min(cond_channels * 2, 64))
        scalar_specs: Dict of scalar control specifications 
                      {name: {mlp_hidden: int}} for building control MLPs
        cond_emb_dim: Embedding dimension for scalar controls 
                      (shared across all scalar MLPs, injected into decoder blocks)
    """
    
    def __init__(
        self,
        im_channels: int,
        model_config: dict,
        cond_channels: int = 3,
        cond_projected_channels: Optional[int] = None,
        scalar_specs: Optional[Dict[str, dict]] = None,
        cond_emb_dim: int = 128,
    ):
        super().__init__()
        self.im_channels = im_channels
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.tanh_activation = model_config.get('tanh_activation', False)
        self.tanh_scaling = model_config.get('tanh_scaling', 1.0)
        
        self.attns = model_config['attn_down']
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Conditioning config
        self.cond_channels = cond_channels
        if cond_projected_channels is None:
            cond_projected_channels = min(cond_channels * 2, 64)
        self.cond_projected_channels = cond_projected_channels
        self.cond_emb_dim = cond_emb_dim
        
        # Validation
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        self.up_sample = list(reversed(self.down_sample))
        
        # ==================== Encoder ====================
        # Encoder takes masked input + mask channel: im_channels + 1
        encoder_in_channels = im_channels + 1  # +1 for mask channel
        self.encoder_conv_in = nn.Conv2d(
            encoder_in_channels, self.down_channels[0], 
            kernel_size=3, padding=(1, 1)
        )
        
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(
                self.down_channels[i], self.down_channels[i + 1],
                t_emb_dim=None, down_sample=self.down_sample[i],
                num_heads=self.num_heads,
                num_layers=self.num_down_layers,
                attn=self.attns[i],
                norm_channels=self.norm_channels
            ))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(
                self.mid_channels[i], self.mid_channels[i + 1],
                t_emb_dim=None,
                num_heads=self.num_heads,
                num_layers=self.num_mid_layers,
                norm_channels=self.norm_channels
            ))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], 2 * self.z_channels, kernel_size=3, padding=1
        )
        self.pre_quant_conv = nn.Conv2d(
            2 * self.z_channels, 2 * self.z_channels, kernel_size=1
        )
        
        # ==================== Decoder Conditioning ====================
        # 1. Project conditioning to learned representation (1x1 conv, no spatial mixing)
        #    Following the same pattern as UNet's cond_conv_in
        self.cond_conv_in = nn.Conv2d(
            in_channels=self.cond_channels,
            out_channels=self.cond_projected_channels,
            kernel_size=1,
            bias=False
        )
        
        # ==================== Decoder ====================
        self.post_quant_conv = nn.Conv2d(
            self.z_channels, self.z_channels, kernel_size=1
        )
        
        # 2. Fuse z + projected conditioning (3x3 conv with spatial mixing)
        #    z_channels + cond_projected_channels → mid_channels[-1]
        self.decoder_conv_in = nn.Conv2d(
            self.z_channels + self.cond_projected_channels,
            self.mid_channels[-1],
            kernel_size=3, padding=(1, 1)
        )
        
        # Decoder blocks with t_emb_dim enabled for scalar control injection
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(
                self.mid_channels[i], self.mid_channels[i - 1],
                t_emb_dim=self.cond_emb_dim,  # Enable conditioning embedding
                num_heads=self.num_heads,
                num_layers=self.num_mid_layers,
                norm_channels=self.norm_channels
            ))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(
                self.down_channels[i], self.down_channels[i - 1],
                t_emb_dim=self.cond_emb_dim,  # Enable conditioning embedding
                up_sample=self.down_sample[i - 1],
                num_heads=self.num_heads,
                num_layers=self.num_up_layers,
                attn=self.attns[i - 1],
                norm_channels=self.norm_channels
            ))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(
            self.down_channels[0], im_channels, kernel_size=3, padding=1
        )
        
        # ==================== Scalar Control MLPs ====================
        # Each scalar key gets its own MLP: scalar → cond_emb_dim
        # Injected additively into the conditioning embedding (like UNet time embedding)
        self.scalar_mlps = nn.ModuleDict()
        if scalar_specs:
            for key, spec in scalar_specs.items():
                hidden = int(spec.get('mlp_hidden', 128))
                self.scalar_mlps[key] = nn.Sequential(
                    nn.Linear(1, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, self.cond_emb_dim),
                )
    
    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode masked input to latent distribution.
        
        Args:
            x: Target image [B, C, H, W] (full, unmasked — masked internally)
            mask: Inpainting mask [B, 1, H, W] (1 = inpaint region, 0 = keep)
            
        Returns:
            sample: Sampled latent [B, z_channels, H', W']
            mean: Posterior mean [B, z_channels, H', W']
            logvar: Posterior log-variance [B, z_channels, H', W']
        """
        # Create masked input: zero out inpainting region, keep context
        masked_input = x * (1.0 - mask)
        
        # Concatenate masked input + mask channel for encoder
        encoder_input = torch.cat([masked_input, mask], dim=1)  # [B, C+1, H, W]
        
        out = self.encoder_conv_in(encoder_input)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape, device=x.device)
        return sample, mean, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        scalar_cond: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Decode latent with conditioning.
        
        Args:
            z: Latent sample [B, z_channels, H', W']
            cond: Spatial conditioning at latent resolution [B, cond_channels, H', W']
                  (e.g., mask(1) + environmental_latent(2) = 3 channels)
            scalar_cond: Optional dict of scalar controls {key: [B] tensor}
            
        Returns:
            Reconstructed output [B, im_channels, H, W]
        """
        # Project conditioning: 1x1 conv (same as UNet pattern)
        cond_proj = self.cond_conv_in(cond)  # [B, cond_projected, H', W']
        
        # Post-quant conv on z
        out = self.post_quant_conv(z)  # [B, z_channels, H', W']
        
        # Concatenate z + projected conditioning, then fuse with 3x3 conv
        out = torch.cat([out, cond_proj], dim=1)  # [B, z + cond_proj, H', W']
        out = self.decoder_conv_in(out)  # [B, mid_channels[-1], H', W']
        
        # Build scalar conditioning embedding (additive, like UNet time embedding)
        # Always initialize to zeros — decoder blocks have t_emb_dim set and
        # expect a valid tensor even when no scalar controls are configured.
        cond_emb = torch.zeros(
            z.shape[0], self.cond_emb_dim, device=z.device
        )
        if self.scalar_mlps and scalar_cond is not None:
            for key, mlp in self.scalar_mlps.items():
                if key in scalar_cond:
                    scalar = scalar_cond[key].float()
                    if scalar.ndim == 1:
                        scalar = scalar[:, None]  # [B] -> [B, 1]
                    cond_emb = cond_emb + mlp(scalar)
        
        # Decoder blocks (pass cond_emb through t_emb pathway)
        for mid in self.decoder_mids:
            out = mid(out, t_emb=cond_emb)
        for up in self.decoder_layers:
            out = up(out, t_emb=cond_emb)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        
        if self.tanh_activation:
            out = torch.tanh(out) * self.tanh_scaling
        
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
        scalar_cond: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode masked input → sample z → decode with conditioning.
        
        Args:
            x: Full target image [B, C, H, W]
            mask: Inpainting mask [B, 1, H, W] (1 = inpaint, 0 = keep)
            cond: Spatial conditioning at latent resolution [B, cond_channels, H', W']
            scalar_cond: Optional scalar controls dict
            
        Returns:
            recon: Reconstructed image [B, C, H, W]
            z: Sampled latent [B, z_channels, H', W']
            mean: Posterior mean
            logvar: Posterior log-variance
        """
        z, mean, logvar = self.encode(x, mask)
        recon = self.decode(z, cond, scalar_cond)
        return recon, z, mean, logvar
    
    @classmethod
    def from_pretrained_vae(
        cls,
        pretrained_checkpoint_path: str,
        im_channels: int,
        model_config: dict,
        cond_channels: int = 3,
        cond_projected_channels: Optional[int] = None,
        scalar_specs: Optional[Dict[str, dict]] = None,
        cond_emb_dim: int = 128,
        device: str = 'cpu',
    ) -> 'ConditionalVAE':
        """
        Create a ConditionalVAE initialized from a pretrained VAE checkpoint.
        
        Loads all matching encoder/decoder weights from the base VAE.
        New conditioning layers (cond_conv_in, scalar MLPs) are randomly initialized.
        The extra mask channel in encoder_conv_in is zero-initialized so the model
        starts with behavior identical to the pretrained VAE (mask has no effect initially).
        
        Args:
            pretrained_checkpoint_path: Path to pretrained VAE checkpoint
            im_channels: Number of target channels
            model_config: Architecture config dict
            cond_channels: Conditioning channels at latent resolution
            cond_projected_channels: Projected conditioning size
            scalar_specs: Scalar control specifications
            cond_emb_dim: Scalar embedding dimension
            device: Device for loading
            
        Returns:
            ConditionalVAE with pretrained weights loaded
        """
        # Create the ConditionalVAE
        cvae = cls(
            im_channels=im_channels,
            model_config=model_config,
            cond_channels=cond_channels,
            cond_projected_channels=cond_projected_channels,
            scalar_specs=scalar_specs,
            cond_emb_dim=cond_emb_dim,
        )
        
        # Load pretrained VAE checkpoint
        checkpoint = torch.load(
            pretrained_checkpoint_path, 
            map_location=device,
            weights_only=False
        )
        
        if 'model_state_dict' in checkpoint:
            pretrained_state = checkpoint['model_state_dict']
        else:
            pretrained_state = checkpoint
        
        cvae_state = cvae.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for key, pretrained_param in pretrained_state.items():
            if key not in cvae_state:
                skipped_keys.append(f"{key} (not in CVAE)")
                continue
            
            cvae_param = cvae_state[key]
            
            if pretrained_param.shape == cvae_param.shape:
                # Exact shape match — direct copy
                cvae_state[key] = pretrained_param
                loaded_keys.append(key)
                
            elif key == 'encoder_conv_in.weight':
                # Encoder conv_in: pretrained has [out, im_channels, kH, kW]
                # CVAE has [out, im_channels + 1, kH, kW] — extra mask channel
                # Copy pretrained weights, zero-initialize mask channel
                out_ch, _, kH, kW = pretrained_param.shape
                cvae_state[key][:, :im_channels, :, :] = pretrained_param
                cvae_state[key][:, im_channels:, :, :] = 0.0  # Zero-init mask channel
                loaded_keys.append(f"{key} (partial: {pretrained_param.shape} → {cvae_param.shape})")
                
            elif key == 'encoder_conv_in.bias':
                # Bias shape matches (depends only on out_channels)
                if pretrained_param.shape == cvae_param.shape:
                    cvae_state[key] = pretrained_param
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
                
            elif key == 'decoder_conv_in.weight':
                # Decoder conv_in: pretrained has [out, z_channels, kH, kW]
                # CVAE has [out, z_channels + cond_projected_channels, kH, kW]
                # Copy z_channels weights, zero-initialize conditioning channels
                z_ch = model_config['z_channels']
                cvae_state[key][:, :z_ch, :, :] = pretrained_param
                cvae_state[key][:, z_ch:, :, :] = 0.0  # Zero-init conditioning channels
                loaded_keys.append(f"{key} (partial: {pretrained_param.shape} → {cvae_param.shape})")
                
            elif key == 'decoder_conv_in.bias':
                if pretrained_param.shape == cvae_param.shape:
                    cvae_state[key] = pretrained_param
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (shape mismatch: {pretrained_param.shape} vs {cvae_param.shape})")
        
        cvae.load_state_dict(cvae_state)
        
        # Print loading summary
        print(f"\n{'='*60}")
        print(f"ConditionalVAE initialized from pretrained VAE")
        print(f"{'='*60}")
        print(f"  ✓ Loaded {len(loaded_keys)} parameters")
        print(f"  ⚠ Skipped {len(skipped_keys)} parameters (new or shape mismatch)")
        
        if skipped_keys:
            print(f"  Skipped keys:")
            for k in skipped_keys:
                print(f"    - {k}")
        
        # Count new (randomly initialized) parameters
        new_param_count = sum(
            p.numel() for name, p in cvae.named_parameters()
            if name not in pretrained_state or pretrained_state.get(name, torch.tensor([])).shape != p.shape
        )
        total_param_count = sum(p.numel() for p in cvae.parameters())
        print(f"  New parameters: {new_param_count:,} / {total_param_count:,} total")
        print(f"{'='*60}\n")
        
        return cvae

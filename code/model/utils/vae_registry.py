###### import libraries ######
# Standard libraries
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Data Science/ML
import torch
import torch.nn as nn

# Local imports
from model.diffusion_blocks.vae import VAE


class VAERegistry:
    """
    Registry for managing multiple VAE models, each specializing in a layer group.
    
    Supports:
    - Loading multiple VAEs for different layer groups (rgb, landuse, environmental)
    - Encoding/decoding with automatic routing to correct VAE
    - Latent composition for diffusion conditioning
    - Checkpoint management
    
    Example usage:
        registry = VAERegistry(config, device='cuda')
        registry.load_vae('landuse', checkpoint_path)
        
        # Encode grouped data
        latents = registry.encode_groups(data_dict)  # data_dict['landuse'] -> latents['landuse']
        
        # Extract specific latent channels for conditioning
        cond_latents = registry.extract_latent_channels(
            latents,
            {'landuse': ['buildings', 'streets'], 'environmental': ['lst']}
        )
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Initialize VAE Registry.
        
        Args:
            config: Full configuration dict with 'vae_groups' and 'layers' keys
            device: Device to load models on
        """
        self.config = config
        self.device = device
        
        # Parse configurations
        self.vae_groups = config.get('vae_groups', {})
        self.layers = config.get('layers', {})
        
        # VAE storage: group_name -> VAE model
        self.vaes: Dict[str, VAE] = {}
        
        # Layer mappings
        self._build_layer_mappings()
        
    def _build_layer_mappings(self):
        """Build mappings between layers and VAE groups."""
        # layer_name -> (group_name, channel_index_in_group)
        self.layer_to_group: Dict[str, Tuple[str, int]] = {}
        
        # group_name -> list of layer names
        self.group_to_layers: Dict[str, List[str]] = {}
        
        for group_name, group_config in self.vae_groups.items():
            layers = group_config.get('layers', [])
            self.group_to_layers[group_name] = layers
            
            for channel_idx, layer_name in enumerate(layers):
                self.layer_to_group[layer_name] = (group_name, channel_idx)
    
    def register_vae(
        self, 
        group_name: str, 
        vae: VAE,
        checkpoint_path: Optional[str] = None
    ):
        """
        Register a VAE for a specific group.
        
        Args:
            group_name: Name of the layer group (e.g., 'landuse', 'rgb')
            vae: Instantiated VAE model
            checkpoint_path: Optional path to load weights from
        """
        if group_name not in self.vae_groups:
            raise ValueError(f"Unknown VAE group: {group_name}. Available: {list(self.vae_groups.keys())}")
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle both new format (dict with 'model_state_dict') and legacy format (direct state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✓ Loaded {group_name} VAE from {checkpoint_path} (epoch {epoch})")
            else:
                # Legacy format: checkpoint is the state_dict directly
                model_state = checkpoint
                print(f"✓ Loaded {group_name} VAE from {checkpoint_path}")
            
            vae.load_state_dict(model_state)
        
        vae.to(self.device)
        vae.eval()
        self.vaes[group_name] = vae
        
    def load_vae(
        self, 
        group_name: str, 
        checkpoint_path: str,
        autoencoder_config: Optional[dict] = None
    ):
        """
        Load VAE from checkpoint.
        
        Args:
            group_name: Name of layer group
            checkpoint_path: Path to checkpoint file
            autoencoder_config: Optional VAE architecture config (uses group config if None)
        """
        if group_name not in self.vae_groups:
            raise ValueError(f"Unknown VAE group: {group_name}")
        
        # Get group config
        group_config = self.vae_groups[group_name]
        
        # Use provided config or extract from group config
        if autoencoder_config is None:
            autoencoder_config = {
                k: v for k, v in group_config.items() 
                if k != 'layers'
            }
        
        # Count input channels
        im_channels = self._count_group_channels(group_name)
        
        # Create VAE
        vae = VAE(
            im_channels=im_channels,
            model_config=autoencoder_config
        )
        
        # Load weights
        self.register_vae(group_name, vae, checkpoint_path)
        
    def get_vae(self, group_name: str) -> VAE:
        """
        Retrieve registered VAE for a group.
        
        Args:
            group_name: Name of layer group
        Returns:
            VAE model
        """
        
        if group_name not in self.vaes:
            raise ValueError(f"VAE group '{group_name}' not loaded")
        return self.vaes[group_name]
        
    def _count_group_channels(self, group_name: str) -> int:
        """Count total channels for a VAE group."""
        layers = self.group_to_layers[group_name]
        total_channels = 0
        
        for layer_name in layers:
            layer_config = self.layers.get(layer_name, {})
            
            # Handle multi-channel layers (e.g., RGB)
            if 'channels' in layer_config:
                total_channels += len(layer_config['channels'])
            else:
                total_channels += 1
        
        return total_channels
    
    def encode_groups(
        self, 
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode grouped data through respective VAEs.
        
        Args:
            data_dict: Dictionary mapping group_name -> tensor [B, C, H, W]
            
        Returns:
            Dictionary mapping group_name -> latent tensor [B, Z, H', W']
        """
        latents = {}
        
        for group_name, data in data_dict.items():
            if group_name not in self.vaes:
                print(f"⚠ Warning: No VAE registered for group '{group_name}', skipping")
                continue
            
            vae = self.vaes[group_name]
            
            with torch.no_grad():
                data = data.to(self.device)
                latent, _ = vae.encode(data)  # Returns (latent, log_var)
                latents[group_name] = latent
        
        return latents
    
    def decode_groups(
        self, 
        latents_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latents through respective VAEs.
        
        Args:
            latents_dict: Dictionary mapping group_name -> latent tensor
            
        Returns:
            Dictionary mapping group_name -> reconstructed tensor
        """
        outputs = {}
        
        for group_name, latent in latents_dict.items():
            if group_name not in self.vaes:
                print(f"⚠ Warning: No VAE registered for group '{group_name}', skipping")
                continue
            
            vae = self.vaes[group_name]
            
            with torch.no_grad():
                latent = latent.to(self.device)
                output = vae.decode(latent)
                outputs[group_name] = output
        
        return outputs
    
    def freeze_all(self):
        """Freeze all VAE models (disable gradient computation)."""
        for group_name, vae in self.vaes.items():
            for param in vae.parameters():
                param.requires_grad = False
            print(f"✓ Frozen {group_name} VAE")
    
    def unfreeze_all(self):
        """Unfreeze all VAE models (enable gradient computation)."""
        for group_name, vae in self.vaes.items():
            for param in vae.parameters():
                param.requires_grad = True
            print(f"✓ Unfrozen {group_name} VAE")
    
    def freeze(self, group_name: str):
        """Freeze specific VAE model."""
        if group_name not in self.vaes:
            raise ValueError(f"VAE group '{group_name}' not loaded")
        
        for param in self.vaes[group_name].parameters():
            param.requires_grad = False
        print(f"✓ Frozen {group_name} VAE")
    
    def unfreeze(self, group_name: str):
        """Unfreeze specific VAE model."""
        if group_name not in self.vaes:
            raise ValueError(f"VAE group '{group_name}' not loaded")
        
        for param in self.vaes[group_name].parameters():
            param.requires_grad = True
        print(f"✓ Unfrozen {group_name} VAE")
    
    def __repr__(self):
        loaded_groups = list(self.vaes.keys())
        available_groups = list(self.vae_groups.keys())
        return (f"VAERegistry(loaded={loaded_groups}, "
                f"available={available_groups})")

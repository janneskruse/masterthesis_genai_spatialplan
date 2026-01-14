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
            vae.load_state_dict(checkpoint)
            print(f"✓ Loaded {group_name} VAE from {checkpoint_path}")
        
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
    
    def extract_latent_channels(
        self,
        latents_dict: Dict[str, torch.Tensor],
        layer_selection: Dict[str, List[str]]
    ) -> torch.Tensor:
        """
        Extract and concatenate specific latent channels for conditioning.
        
        Args:
            latents_dict: Full latents from encode_groups()
            layer_selection: Dict mapping group_name -> list of layer names to extract
            
        Returns:
            Concatenated latent tensor [B, sum(z_channels), H', W']
            
        Example:
            layer_selection = {
                'landuse': ['buildings', 'streets'],
                'environmental': ['lst']
            }
        """
        selected_latents = []
        
        for group_name, layer_names in layer_selection.items():
            if group_name not in latents_dict:
                raise ValueError(f"Group '{group_name}' not found in latents_dict")
            
            group_latent = latents_dict[group_name]
            group_layers = self.group_to_layers[group_name]
            
            # Find channel indices for requested layers
            for layer_name in layer_names:
                if layer_name not in group_layers:
                    raise ValueError(f"Layer '{layer_name}' not in group '{group_name}'")
                
                # Get channel index
                _, channel_idx = self.layer_to_group[layer_name]
                
                # Calculate latent channel index
                # Assuming uniform z_channels per input channel
                group_config = self.vae_groups[group_name]
                z_channels = group_config.get('z_channels', 4)
                im_channels = self._count_group_channels(group_name)
                z_per_channel = z_channels // im_channels
                
                start_z = channel_idx * z_per_channel
                end_z = start_z + z_per_channel
                
                selected_latents.append(group_latent[:, start_z:end_z])
        
        if not selected_latents:
            raise ValueError("No latent channels selected")
        
        return torch.cat(selected_latents, dim=1)
    
    def get_prediction_latent_size(self, stage_config: dict) -> int:
        """
        Calculate expected latent size for prediction.
        
        Args:
            stage_config: Diffusion stage config with 'prediction_group' key
            
        Returns:
            Number of latent channels
        """
        group_name = stage_config.get('prediction_group')
        if group_name not in self.vae_groups:
            raise ValueError(f"Unknown prediction group: {group_name}")
        
        return self.vae_groups[group_name].get('z_channels', 4)
    
    def get_conditioning_latent_size(self, stage_config: dict) -> int:
        """
        Calculate expected latent size for conditioning.
        
        Args:
            stage_config: Diffusion stage config with 'conditioning' key
            
        Returns:
            Number of conditioning latent channels
        """
        conditioning = stage_config.get('conditioning', {})
        latent_cond = conditioning.get('latent_space', [])
        
        total_z = 0
        for cond_spec in latent_cond:
            group_name = cond_spec['group']
            layers = cond_spec['layers']
            
            group_config = self.vae_groups[group_name]
            z_channels = group_config.get('z_channels', 4)
            im_channels = self._count_group_channels(group_name)
            z_per_channel = z_channels // im_channels
            
            total_z += len(layers) * z_per_channel
        
        return total_z
    
    def __repr__(self):
        loaded_groups = list(self.vaes.keys())
        available_groups = list(self.vae_groups.keys())
        return (f"VAERegistry(loaded={loaded_groups}, "
                f"available={available_groups})")

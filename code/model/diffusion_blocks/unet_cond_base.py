# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/models
# Import libraries
# Data Science/ML
import torch
from einops import einsum
import torch.nn as nn

# Local imports
from model.diffusion_blocks.blocks import get_time_embedding
from model.diffusion_blocks.blocks import DownBlock, MidBlock, UpBlockUnet
from model.utils.config_utils import *


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    
    def __init__(self, im_channels, model_config):
        """
        Args:
            im_channels: Number of input channels
            model_config: Model configuration dict
        """
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            
            # class and text are kept for potential future use cases
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            
            # currently hardcoded by build_unet_condition_config()
            if 'image' in condition_types: 
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config'][
                    'image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config'][
                    'image_condition_output_channels']
                
        if self.class_cond:
            # Rather than using a special null class we dont add the
            # class embedding information for unconditional generation
            self.class_emb = nn.Embedding(self.num_classes,
                                          self.t_emb_dim)
        
        if self.image_cond:
            # Validate that we have conditioning channels
            if self.im_cond_input_ch == 0:
                print(f"⚠ Warning: image conditioning enabled but im_cond_input_ch=0. Disabling image conditioning.")
                self.image_cond = False
            else:
                # Conditioning injection layers:
                # 1. Project conditioning to learned representation (1x1 conv, no spatial mixing)
                #    This allows the model to learn how to interpret raw conditioning signals
                #    Example: [B, 3, H, W] → [B, 6, H, W] where 3 = mask(1) + env_latent(2)
                self.cond_conv_in = nn.Conv2d(
                    in_channels=self.im_cond_input_ch,   # Total conditioning channels (pixel + latent groups)
                    out_channels=self.im_cond_output_ch,  # Projected size (typically 2x input or capped at 128)
                    kernel_size=1,  # Pointwise: per-pixel channel mixing only
                    bias=False
                )
                
                # 2. Fuse prediction input + projected conditioning (3x3 conv with spatial mixing)
                #    Concatenates [input_latent, projected_conditioning] then maps to first U-Net layer
                #    Example: [B, 4+6=10, H, W] → [B, 32, H, W] where 32 = down_channels[0]
                self.conv_in_concat = nn.Conv2d(
                    im_channels + self.im_cond_output_ch,  # Concatenated channels
                    self.down_channels[0],  # First U-Net layer size
                    kernel_size=3,  # Spatial mixing for joint representation
                    padding=1
                )
        
        if not self.image_cond:
            # No image conditioning - standard conv input
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        # Determine the conditioning used
        self.cond = self.text_cond or self.image_cond or self.class_cond
        
        
        # Generic scalar control conditioning (temperature, vegetation, heights, etc.)
        # Each scalar key gets its own MLP to inject into time embedding
        scalar_cfg = self.condition_config.get('scalar_condition_config', None) if self.condition_config else None
        self.scalar_mlps = nn.ModuleDict()
        
        if scalar_cfg:
            scalar_specs = scalar_cfg.get('scalars', {})
            for key, spec in scalar_specs.items():
                hidden = int(spec.get('mlp_hidden', 128))
                self.scalar_mlps[key] = nn.Sequential(
                    nn.Linear(1, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, self.t_emb_dim),
                )
            print(f"✓ Scalar controls enabled: {list(self.scalar_mlps.keys())}")
        
        #####################################################
        
        
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])
        
        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels,
                                        cross_attn=self.text_cond,
                                        context_dim=self.text_embed_dim))
        
        self.mids = nn.ModuleList([])
        # Build the Midblocks
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.text_cond,
                                      context_dim=self.text_embed_dim))
                
        self.ups = nn.ModuleList([])
        # Build the Upblocks
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels,
                            cross_attn=self.text_cond,
                            context_dim=self.text_embed_dim))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_input=None):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep [B] or int
            cond_input: Conditioning dictionary
            
        Returns:
            Noise prediction [B, C, H, W]
        """
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"
        if self.image_cond:
            ######## Image Conditioning ########
            validate_image_conditional_input(cond_input, x)
            
            # Collect all conditioning tensors
            cond_tensors = []
            
            # Add pixel-space conditioning (e.g., inpainting_mask)
            if 'image' in cond_input:
                pixel_cond = cond_input['image']
                pixel_cond = torch.nn.functional.interpolate(pixel_cond, size=x.shape[-2:])
                cond_tensors.append(pixel_cond)
            
            # Add latent-space conditioning groups (e.g., semantic, environmental)
            if 'image_condition_config' in self.condition_config:
                latent_specs = self.condition_config['image_condition_config'].get('latent_space_specs', [])
                for spec in latent_specs:
                    group_name = spec.get('group')
                    if group_name in cond_input:
                        latent_cond = cond_input[group_name]
                        # Interpolate to match prediction latent size
                        latent_cond = torch.nn.functional.interpolate(latent_cond, size=x.shape[-2:])
                        cond_tensors.append(latent_cond)
            
            # Concatenate all conditioning
            im_cond = torch.cat(cond_tensors, dim=1) if cond_tensors else torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            im_cond = self.cond_conv_in(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)
            # B x (C+N) x H x W
            out = self.conv_in_concat(x)
        else:
            # B x C x H x W
            out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        ######## Class Conditioning ########
        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)
            class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
            t_emb += class_embed

        
        ######## Generic Scalar Control Conditioning ########
        # Inject all configured scalar controls into time embedding
        if self.scalar_mlps and cond_input is not None:
            for key, mlp in self.scalar_mlps.items():
                if key in cond_input:
                    scalar = cond_input[key].float()  # [B] or [B, 1]
                    if scalar.ndim == 1:
                        scalar = scalar[:, None]  # [B] -> [B, 1]
                    t_emb = t_emb + mlp(scalar)
            
        ############## hidden states for cross-attention ##############
        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, \
                "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input['text']
        down_outs = []
        
        ########## Downsampling path ########
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        
        # Output (noise prediction)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        
        return out

# adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main/models
import torch
from einops import einsum
import torch.nn as nn
from model.diffusion_blocks.blocks import get_time_embedding
from model.diffusion_blocks.blocks import DownBlock, MidBlock, UpBlockUnet
from model.utils.config_utils import *


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    
    def __init__(self, im_channels, model_config):
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
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
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
            # Compute expected input channels from condition_config
            expected_channels = self._compute_expected_conditioning_channels()
            
            # Validate or override configured channels
            if expected_channels != self.im_cond_input_ch:
                print(f"⚠ Warning: Configured image_condition_input_channels={self.im_cond_input_ch}, "
                      f"but computed={expected_channels} based on condition_types.")
                print(f"  Using computed value: {expected_channels}")
                self.im_cond_input_ch = expected_channels
            
            # Map the mask image to a N channel image and
            # concat that with input across channel dimension
            self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                          out_channels=self.im_cond_output_ch,
                                          kernel_size=1,
                                          bias=False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond or self.image_cond or self.class_cond
        ###################################
        
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
        
        # Segmentation head for reconstructing OSM masks
        self.segmentation_head = None
        self.environmental_head = None
        
        if self.condition_config is not None:
            condition_types = self.condition_config.get('condition_types', [])
            
            # OSM feature prediction head
            if 'osm_features' in condition_types:
                num_osm_classes = len(self.condition_config.get('osm_layers', []))
                if num_osm_classes > 0:
                    self.segmentation_head = nn.Sequential(
                        nn.GroupNorm(self.norm_channels, self.conv_out_channels),
                        nn.SiLU(),
                        nn.Conv2d(self.conv_out_channels, num_osm_classes, kernel_size=3, padding=1)
                    )
                    print(f"✓ Initialized OSM segmentation head with {num_osm_classes} classes")
            
            # Environmental layer prediction head
            if 'environmental' in condition_types:
                # Use prediction layers (subset of conditioning layers)
                env_pred_layers = self.condition_config.get('environmental_prediction_layers', 
                                                           self.condition_config.get('environmental_layers', []))
                num_env_classes = len(env_pred_layers)
                if num_env_classes > 0:
                    self.environmental_head = nn.Sequential(
                        nn.GroupNorm(self.norm_channels, self.conv_out_channels),
                        nn.SiLU(),
                        nn.Conv2d(self.conv_out_channels, num_env_classes, kernel_size=3, padding=1)
                    )
                    print(f"✓ Initialized environmental prediction head with {num_env_classes} layers: {env_pred_layers}")
    
    def _compute_expected_conditioning_channels(self) -> int:
        """
        Compute expected number of image conditioning channels based on condition_config.
        
        Returns:
            Total channels: masked_rgb (3 if enabled) + mask (1) + osm (N) + env (M)
        """
        if not self.condition_config:
            return 0
        
        total_channels = 0
        condition_types = self.condition_config.get('condition_types', [])
        
        if 'inpainting' in condition_types:
            # Add masked RGB if explicitly included
            if 'masked_rgb' in condition_types:
                total_channels += 3  # RGB channels
            # Add mask channel
            total_channels += 1  # Binary mask
        
        if 'osm_features' in condition_types:
            osm_layers = self.condition_config.get('osm_layers', [])
            total_channels += len(osm_layers)
        
        if 'environmental' in condition_types:
            env_layers = self.condition_config.get('environmental_layers', [])
            total_channels += len(env_layers)
        
        return total_channels
    
    def forward(self, x, t, cond_input=None, return_segmentation=False):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep [B] or int
            cond_input: Conditioning dictionary
            return_segmentation: If True, return auxiliary predictions (OSM, environmental)
            
        Returns:
            If return_segmentation=False: noise prediction [B, C, H, W]
            If return_segmentation=True: (noise_pred, osm_seg, env_pred)
                - noise_pred: [B, C, H, W]
                - osm_seg: [B, num_osm, H, W] or None
                - env_pred: [B, num_env, H, W] or None
        """
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"
        if self.image_cond:
            ######## Mask Conditioning ########
            validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image']
            im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)
            # B x (C+N) x H x W
            out = self.conv_in_concat(x)
            #####################################
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
        ####################################
            
        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, \
                "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input['text']
        down_outs = []
        
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
        
        # RGB output (noise prediction)
        rgb_out = self.norm_out(out)
        rgb_out = nn.SiLU()(rgb_out)
        rgb_out = self.conv_out(rgb_out)
        # rgb_out B x C x H x W
        
        # Optional auxiliary predictions
        if return_segmentation:
            outputs = [rgb_out]
            
            # OSM segmentation
            if self.segmentation_head is not None:
                seg_out = self.segmentation_head(out)
                # seg_out B x num_osm_classes x H x W
                outputs.append(seg_out)
            else:
                outputs.append(None)
            
            # Environmental predictions
            if self.environmental_head is not None:
                env_out = self.environmental_head(out)
                # env_out B x num_env_classes x H x W
                outputs.append(env_out)
            else:
                outputs.append(None)
            
            return tuple(outputs)  # (rgb_out, seg_out, env_out)
        
        return rgb_out

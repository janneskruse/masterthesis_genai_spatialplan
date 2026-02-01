"""
Validation sampling utilities for diffusion models.

Functions for generating validation samples during training.
"""
###### import libraries ######
# Standard libraries
import os
from typing import Optional, Dict, Any

# Data Science/ML
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
from model.utils.samples import (
    save_layerwise_samples, 
    save_rgb_composite, 
    save_layerwise_comparisons
)

def run_validation_sampling(
    model: torch.nn.Module,
    val_scheduler: Any,
    val_loader: Optional[Any],
    fallback_loader: Any,
    vae: torch.nn.Module,
    vae_registry: Any,
    urban_dataset: Any,
    prediction_layers: list,
    layers_registry: Dict[str, Any],
    out_dir: str,
    validation_dir_name: str,
    epoch_idx: int,
    val_num_samples: int,
    inpainting_mode: str,
    prediction_type: str,
    device: torch.device,
    val_guidance_scale: Optional[float] = None,
    ema_model: Optional[Any] = None
) -> None:
    """
    Generate validation samples during training.
    
    Args:
        model: Diffusion U-Net model
        val_scheduler: DDIM scheduler for fast validation sampling
        val_loader: Validation data loader (None = use fallback_loader)
        fallback_loader: Training data loader (used if val_loader is None)
        vae: VAE model for prediction group
        vae_registry: Registry of all VAE models
        urban_dataset: Dataset object (for latent_size info)
        prediction_layers: List of layer names being predicted
        layers_registry: Layer configuration registry
        out_dir: Output directory for results
        validation_dir_name: Subdirectory name for validation samples
        epoch_idx: Current epoch index
        val_num_samples: Number of validation samples to generate
        inpainting_mode: Inpainting mode ('hard' | 'sdlike')
        prediction_type: Prediction type ('epsilon' | 'v_prediction')
        device: Torch device
        val_guidance_scale: CFG guidance scale (None = no CFG)
        ema_model: Optional EMA model wrapper
    """
    print(f"\n{'='*50}")
    print(f"Generating Validation Samples (Epoch {epoch_idx + 1})")
    print(f"{'='*50}")
    
    with torch.no_grad():
        # Use EMA weights for validation if available
        if ema_model is not None:
            model_for_ema = model.module if hasattr(model, 'module') else model
            ema_model.store(model_for_ema)  # Store current weights
            ema_model.copy_to(model_for_ema)  # Copy EMA weights to model
            print("✓ Using EMA weights for validation sampling")
        
        model.eval()
        
        # Get validation batch from proper validation split (or training if unavailable)
        if val_loader is not None:
            val_data = next(iter(val_loader))
            print("✓ Using validation split")
        else:
            val_data = next(iter(fallback_loader))
            print("⚠ Using training split (validation unavailable)")
        
        if len(val_data) == 2:
            val_prediction_data, val_cond_input = val_data
        else:
            val_prediction_data = val_data
            val_cond_input = {}
        
        val_prediction_data = val_prediction_data[:val_num_samples].float().to(device)
        
        # Move conditioning to device
        if 'image' in val_cond_input:
            val_cond_input['image'] = val_cond_input['image'][:val_num_samples].float().to(device)
        
        # Slice metadata list (one dict per sample)
        if 'meta' in val_cond_input:
            val_cond_input['meta'] = val_cond_input['meta'][:val_num_samples]
        
        # Encode conditioning groups that need encoding (*_image keys)
        # This happens when validation latents don't exist - dataset provides full-res images
        if vae_registry is not None:
            groups_to_encode = [k for k in val_cond_input.keys() if k.endswith('_image')]
            if groups_to_encode:
                print(f"⚠ Encoding conditioning groups on-the-fly (validation latents missing): {groups_to_encode}")
                for group_key in groups_to_encode:
                    group_name = group_key.replace('_image', '')
                    group_image = val_cond_input.pop(group_key)[:val_num_samples].float().to(device)
                    
                    # Encode through VAE
                    vae_model = vae_registry.get_vae(group_name)
                    if vae_model is not None:
                        with torch.no_grad():
                            group_latent, _, _ = vae_model.encode(group_image)
                        val_cond_input[group_name] = group_latent
                    else:
                        raise ValueError(f"VAE for group '{group_name}' not found in registry")
        
        # Slice and move latent-space conditioning groups (use metadata if available)
        if 'meta' in val_cond_input and 'latent_group_names' in val_cond_input['meta'][0]:
            val_latent_group_keys = val_cond_input['meta'][0]['latent_group_names']
        else:
            # Fallback: infer from keys (excludes image, meta, and scalar controls)
            val_latent_group_keys = [k for k in val_cond_input.keys() if k not in ['image', 'meta'] and isinstance(val_cond_input.get(k), torch.Tensor) and val_cond_input[k].ndim > 2]
        
        for group_key in val_latent_group_keys:
            val_cond_input[group_key] = val_cond_input[group_key][:val_num_samples].float().to(device)
        
        expected_latent_size = urban_dataset.latent_size  
        is_already_latent = (val_prediction_data.shape[-1] == expected_latent_size)
        
        if is_already_latent:
            # Data is already in latent space (precomputed latents)
            val_im_latent = val_prediction_data
        elif vae is not None:
            # Data is in pixel space - encode to latent
            val_im_latent, _, _ = vae.encode(val_prediction_data)
        else:
            raise ValueError("VAE is required for encoding pixel-space validation data to latents.")
        
        # Extract mask
        val_mask_latent = None
        if 'image' in val_cond_input and 'meta' in val_cond_input:
            pixel_space_names = val_cond_input['meta'][0].get('pixel_space_names', [])
            if pixel_space_names and 'inpainting_mask' in pixel_space_names:
                mask_idx = pixel_space_names.index('inpainting_mask')
                val_mask_latent = val_cond_input['image'][:, mask_idx:mask_idx+1, :, :]
        
        if val_mask_latent is None:
            val_mask_latent = torch.ones_like(val_im_latent[:, :1, :, :])
        
        # Initialize from noise in masked region
        x_val = val_im_latent.clone()
        val_noise_context = None
        
        if inpainting_mode == "hard":
            x_val = val_mask_latent * torch.randn_like(x_val) + (1 - val_mask_latent) * x_val
            val_noise_context = torch.randn_like(x_val)
        else:
            x_val = torch.randn_like(x_val)
        
        # Sampling loop with DDIM scheduler (fast validation)
        for step_idx in tqdm(reversed(range(val_scheduler.ddim_steps)), desc="DDIM validation sampling", total=val_scheduler.ddim_steps):
            # Get full timestep value for model conditioning
            t_value = val_scheduler.ddim_timesteps[step_idx].item()
            t_tensor = torch.full((val_num_samples,), t_value, device=device, dtype=torch.long)
            
            # Get model prediction
            model_output = model(x_val, t_tensor, cond_input=val_cond_input)
            
            # Convert to epsilon if using v-prediction
            if prediction_type == 'v_prediction':
                noise_pred = val_scheduler.velocity_to_epsilon(model_output, x_val, t_value)
            else:
                noise_pred = model_output
            
            # Apply CFG if specified
            if val_guidance_scale is not None and val_guidance_scale != 1.0:
                # Unconditional prediction
                from model.utils.diffusion_utils import make_uncond_input_keep_mask
                uncond_input = make_uncond_input_keep_mask(val_cond_input)
                noise_pred_uncond = model(x_val, t_tensor, cond_input=uncond_input)
                # CFG: noise = uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + val_guidance_scale * (noise_pred - noise_pred_uncond)
            
            # DDIM denoise step (uses step_idx, not timestep value)
            if inpainting_mode == "hard":
                x_val, _ = val_scheduler.sample_prev_timestep_inpainting(
                    x_val, noise_pred, step_idx,
                    val_im_latent,
                    val_mask_latent,
                    noise_context=val_noise_context
                )
            else:
                x_val, _ = val_scheduler.sample_prev_timestep(x_val, noise_pred, step_idx)
        
        # Decode to pixel space
        if vae is not None:
            val_decoded = vae.decode(x_val)
            val_decoded_gt = vae.decode(val_im_latent)
        else:
            val_decoded = x_val
            val_decoded_gt = val_im_latent
        
        # Save validation samples
        val_save_dir = os.path.join(out_dir, validation_dir_name, f'epoch_{epoch_idx + 1:04d}')
        os.makedirs(val_save_dir, exist_ok=True)
        
        # Upsample mask to match decoded resolution for visualization
        val_mask_vis = None
        if val_mask_latent is not None:
            val_mask_vis = F.interpolate(
                val_mask_latent,
                size=(val_decoded.shape[2], val_decoded.shape[3]),
                mode='nearest'
            )
        
        # Save comparison: ground truth vs predictions (top: GT, bottom: predictions)
        save_layerwise_comparisons(
            input_tensor=val_decoded_gt,
            recon_tensor=val_decoded,
            channel_names=[f'channel_{i}' for i in range(len(prediction_layers))],
            layer_names=prediction_layers,
            layers_registry=layers_registry,
            save_dir=val_save_dir,
            filename_prefix='validation_comparison',
            n_samples=val_num_samples,
            use_colormaps=True,
            mask=val_mask_vis
        )
        
        # Also save predictions alone for reference
        save_layerwise_samples(
            tensor=val_decoded,
            layer_names=prediction_layers,
            layers_registry=layers_registry,
            save_dir=val_save_dir,
            filename_prefix='validation_prediction',
            n_samples=val_num_samples,
            is_reconstruction=True,
            use_colormaps=True,
            mask=val_mask_vis
        )
        
        if 'rgb' in [l.lower() for l in prediction_layers]:
            rgb_val_path = os.path.join(val_save_dir, 'validation_RGB_composite_prediction.png')
            save_rgb_composite(
                tensor=val_decoded,
                layer_names=prediction_layers,
                save_path=rgb_val_path,
                n_samples=val_num_samples,
                normalize_per_channel=True
            )
        
        print(f"✓ Saved validation samples to {val_save_dir}")
        
        # Restore original weights if using EMA
        if ema_model is not None:
            model_for_ema = model.module if hasattr(model, 'module') else model
            ema_model.restore(model_for_ema)  # Restore original weights
        
        model.train()

# Quick Reference: Urban Inpainting Training

## 🚀 Quick Start (Full Pipeline)

```bash
cd code/model
python tools/run_pipeline.py
```

---

## 📋 Step-by-Step Commands

### 1. Validate Dataset
```bash
python tools/validate_dataset.py --num_samples 5
```
**Check:** `urban_layout_inpainting/dataset_validation/` for visualizations

---

### 2. Train VAE (2-4 hours)
```bash
python tools/train_vae_urban.py
```
**Check:** `urban_layout_inpainting/vae_samples/` for reconstructions

---

### 3. Train Diffusion Model (8-12 hours)
```bash
python tools/train_urban_inpainting.py
```
**Check:** Training loss should decrease to ~0.01-0.03

---

### 4. Generate Samples (5 minutes)
```bash
python tools/sample_urban_inpainting.py --num_samples 8
```
**Check:** `urban_layout_inpainting/inpainting_samples/`

---

## ⚙️ Key Configuration Settings

```yaml
# In config/class_cond.yml

# Dataset
dataset_params:
  regions: ['Leipzig']        # Your city
  patch_size_m: 650          # Patch size in meters
  res: 3                     # Resolution (m/pixel)
  min_valid_percent: 90      # Min valid data per patch

# Inpainting
ldm_params:
  condition_config:
    hole_config:
      type: 'random_square'
      size_px: 80            # Mask size (adjust for larger/smaller holes)
    osm_layers: ['buildings', 'streets', 'water']
    environmental_layers: ['ndvi', 'landsat_surface_temp_b10_masked']
    cond_drop_prob: 0.1      # Classifier-free guidance dropout

# Training
train_params:
  ldm_batch_size: 4          # Reduce if OOM
  ldm_epochs: 200
  mask_loss_weight: 2.0      # Higher = stronger boundary learning
  cf_guidance_scale: 7.5     # Higher = stronger conditioning
```

---

## 🔧 Troubleshooting

### Out of Memory
```yaml
# reduce batch size
ldm_batch_size: 2
autoencoder_batch_size: 4

# or reduce patch size
patch_size_m: 512
```

### Poor Inpainting Boundaries
```yaml
# increase mask loss weight
mask_loss_weight: 3.0  # or 4.0

# train longer
ldm_epochs: 300
```

### Generated Images Look Unconditional
```yaml
# stronger guidance during sampling
cf_guidance_scale: 10.0  # or 15.0

# lower conditioning dropout during training
cond_drop_prob: 0.05
```

### No Valid Patches Found
```yaml
# lower threshold
min_valid_percent: 70
```

---

## 📊 Expected Results


## 📁 Output Structure

```
urban_layout_inpainting/
├── vqvae_urban_ckpt.pth              # VAE checkpoint
├── ddpm_urban_inpainting_ckpt.pth    # Diffusion checkpoint
├── vqvae_latents/                    # Pre-computed latents
│   ├── latent_0.pt
│   ├── latent_1.pt
│   └── ...
├── vae_samples/                      # VAE reconstructions
│   ├── recon_step_500.png
│   └── ...
├── inpainting_samples/               # Generated samples
│   ├── samples_guidance7.5.png
│   ├── sample_0.png
│   └── ...
└── dataset_validation/               # Dataset visualizations
    ├── sample_0.png
    └── ...
```

---

## 🔬 Advanced Usage


## 📚 Documentation Files


## 💡 Tips


## 🆘 Getting Help



## ✅ Success Indicators


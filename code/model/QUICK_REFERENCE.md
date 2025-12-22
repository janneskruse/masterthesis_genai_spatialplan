# Quick Reference: Urban Inpainting Training

## 🚀 Quick Start (Full Pipeline)

```bash
cd code/model
python tools/run_pipeline.py --config config/diffusion_1.yml
```

---

## ⚡ Distributed Data Parallel (DDP) Training

For **significantly faster training** on HPC clusters with multiple GPUs:

### Why Use DDP?
- **4x faster** with 4 GPUs (3 hours vs 12 hours for VAE, 6 hours vs 24+ hours for diffusion)
- **Synchronized training** across GPUs for better convergence
- **Automatic checkpointing** and job chaining
- **SLURM integration** for easy cluster deployment

### Requirements
- HPC cluster with SLURM workload manager
- Multiple GPUs (tested with 4x A30 24GB)
- Conda environment: `genaiSpatialplan`

### Usage
```bash
# Submit VAE training (automatically chains to diffusion training)
sbatch tools/train_vae_urban_ddp.sh config/diffusion_1.yml

# Or submit diffusion training only (if VAE checkpoint exists)
sbatch tools/train_urban_inpainting_ddp.sh config/diffusion_1.yml

# Monitor progress
squeue -u $USER
tail -f log/train_vae_urban_ddp.out-<job_id>
```

### For Local Laptop Training
**Use standard (non-DDP) scripts** for single GPU or CPU training:
```bash
python tools/train_vae_urban.py --config config/diffusion_1.yml
python tools/train_urban_inpainting.py --config config/diffusion_1.yml
```

---

## 📋 Step-by-Step Commands

### 1. Validate Dataset
```bash
python tools/validate_dataset.py --config config/diffusion_1.yml --num_samples 5
```
**Check:** `urban_layout_inpainting/dataset_validation/` for visualizations

---

### 2. Train VAE

**Standard (laptop/single GPU):**
```bash
python tools/train_vae_urban.py --config config/diffusion_1.yml
```

**Distributed Data Parallel (HPC cluster with SLURM):**
```bash
sbatch tools/train_vae_urban_ddp.sh config/diffusion_1.yml
```
- Uses 4x A30 GPUs (configurable in .sh script)
- Significantly faster training (~3 hours vs 8-12 hours)
- Automatically chains to diffusion training on completion

**Check:** `urban_layout_inpainting/vae_samples/` for reconstructions

---

### 3. Train Diffusion Model

**Standard (laptop/single GPU):**
```bash
python tools/train_urban_inpainting.py --config config/diffusion_1.yml
```

**Distributed Data Parallel (HPC cluster with SLURM):**
```bash
sbatch tools/train_urban_inpainting_ddp.sh config/diffusion_1.yml
```
- Uses 4x A30 GPUs (configurable in .sh script)
- Reduces training time by ~75% (6 hours vs 24+ hours)
- Synchronized batch normalization across GPUs

**Check:** Training loss should decrease to ~0.01-0.03

---

### 4. Generate Samples (5 minutes)
```bash
python tools/sample_urban_inpainting.py --config config/diffusion_1.yml --num_samples 8
```
**Check:** `urban_layout_inpainting/inpainting_samples/`

---

## ⚙️ Key Configuration Settings

```yaml
# In config/diffusion_1.yml

# Cluster mode (enables DDP-specific settings)
cluster: True  # Set to False for local laptop training

# Dataset
dataset_params:
  train_regions: ['Dresden', 'Hamburg', 'Stuttgart']  # Training cities
  eval_regions: ['Leipzig']  # Evaluation/validation cities
  zarr_name: "input_config_ge25_cc10_2019_2024_clipped.zarr"
  patch_size_m: 800          # Patch size in meters
  res: 3                     # Resolution (m/pixel)
  min_valid_percent: 90      # Min valid data per patch
  stride_overlap: 2          # 50% overlap (2), 25% overlap (4)

# Diffusion model architecture
ldm_params:
  down_channels: [32, 64, 128, 256]
  mid_channels: [256, 256, 128]
  conv_out_channels: 128     # Latent space channels
  time_emb_dim: 128
  num_heads: 4
  
  # Conditioning configuration
  condition_config:
    condition_types: ['inpainting', 'osm_features', 'environmental']
    
    # Inpainting mask configuration
    hole_config:
      type: 'street_blocks'   # 'random_square', 'center_square', 'street_blocks'
      max_coverage_percent: 25  # Max % of image to mask
      size_px: 80             # Size for random/center square
      min_size_px: 40         # For variable mask sizes
      max_size_px: 120

    # OSM feature layers for conditioning
    osm_layers: ['buildings', 'streets', 'street_blocks', 'water']
    
    # Environmental layers for conditioning (input)
    environmental_layers: ['ndvi', 'landsat_surface_temp_b10_masked']
    
    # Environmental layers to predict (output heads)
    environmental_prediction_layers: ['landsat_surface_temp_b10_masked']
    
    # Conditioning dropout for classifier-free guidance
    image_condition_config:
      cond_drop_prob: 0.1    # 10% unconditional training

# Autoencoder (VAE) architecture
autoencoder_params:
  z_channels: 4
  codebook_size: 8192
  down_channels: [32, 64, 128, 128]
  down_sample: [True, True, True]  # 3 levels: 8x compression
  tanh_activation: False
  tanh_scaling: 0.95

# Training parameters
train_params:
  seed: 42
  ldm_batch_size: 4          # Reduce if OOM
  autoencoder_batch_size: 2
  num_workers: 4
  ldm_epochs: 200
  autoencoder_epochs: 50
  ldm_lr: 0.00001
  autoencoder_lr: 0.0001
  
  # Loss weights
  mask_loss_weight: 2.0      # Higher = stronger boundary learning
  boundary_loss_weight: 1.5  # Extra weight for boundary pixels
  temperature_loss_weight: 0.0  # Gradually increases during training
  seg_loss_weight: 0.1       # OSM segmentation auxiliary loss
  env_loss_weight: 0.1       # Environmental prediction auxiliary loss
  
  # Sampling
  cf_guidance_scale: 7.5     # Classifier-free guidance (higher = stronger conditioning)
  clamp_sampling: True       # Clamp outputs to [-1, 1]
  num_samples: 16
  
  # File/folder names
  task_name: 'diffusion_1'
  autoencoder_ckpt_name: 'vae_urban_ddp_ckpt.pth'  # or 'vae_urban_ckpt.pth' for non-DDP
  latents_dir_name: 'vae_ddp_latents'
  ldm_ckpt_name: 'ddpm_urban_inpainting_ckpt.pth'
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size
ldm_batch_size: 2
autoencoder_batch_size: 1

# Or reduce patch size
patch_size_m: 650  # or 512

# Or reduce number of workers
num_workers: 2
```

### Poor Inpainting Boundaries
```yaml
# Increase mask loss weight
mask_loss_weight: 3.0  # or 4.0
boundary_loss_weight: 2.0

# Train longer
ldm_epochs: 300

# Use smaller, more focused masks
hole_config:
  size_px: 60
  max_coverage_percent: 20
```

### Generated Images Look Unconditional
```yaml
# Stronger guidance during sampling
cf_guidance_scale: 10.0  # or 15.0

# Lower conditioning dropout during training
cond_drop_prob: 0.05  # from 0.1

# Increase conditioning loss weights
seg_loss_weight: 0.2
env_loss_weight: 0.2
```

### No Valid Patches Found
```yaml
# Lower threshold
min_valid_percent: 70  # or 60

# Increase stride overlap
stride_overlap: 4  # 25% overlap for more patches
```

### DDP Training Fails / Hangs
```bash
# Check NCCL debug output in SLURM logs
tail -f log/train_vae_urban_ddp.out-<job_id>

# Common fixes:
# 1. Ensure all GPUs are visible
# 2. Check network connectivity between nodes
# 3. Verify NCCL_SOCKET_IFNAME excludes loopback
# 4. Try reducing number of GPUs in .sh script
```

### Temperature Prediction Looks Wrong
```yaml
# Increase temperature loss weight gradually
temperature_loss_weight: 1.0  # starts at 0.0, increases during training

# Or adjust in training code warmup schedule
# (usually ramps up over first 20-30% of training)
```

---

## 📊 Expected Results

### VAE Training
- **Reconstruction loss** should decrease to ~0.05-0.15
- **Perceptual loss** decreases to ~0.1-0.3
- Visual samples show sharp, realistic reconstructions
- Training time: 
  - Standard: 8-12 hours (50 epochs, single GPU)
  - DDP: 3-4 hours (50 epochs, 4x A30 GPUs)

### Diffusion Training
- **Total loss** decreases to ~0.01-0.03
- **Mask region loss** should be ~2x lower than unmasked
- **Temperature MSE** (if used) around 5-15
- Training time:
  - Standard: 24-36 hours (200 epochs, single GPU)
  - DDP: 6-8 hours (200 epochs, 4x A30 GPUs)

### Sampling
- Inpainted regions blend seamlessly with surroundings
- Generated structures respect OSM conditions (buildings, streets)
- Temperature predictions correlate with urban density
- Higher `cf_guidance_scale` → stronger condition adherence
- Sampling time: ~30-60 seconds per image (50 diffusion steps)

---

## 🖥️ Hardware Requirements

### Minimum (Laptop/Workstation)
- **GPU:** 12GB+ VRAM (e.g., RTX 3080, RTX 4070)
- **RAM:** 32GB system memory
- **Storage:** 100GB+ free space (datasets + checkpoints)
- **Training time:** 24-48 hours total

### Recommended (HPC Cluster with SLURM)
- **GPUs:** 4x A30 (24GB each) or equivalent
- **RAM:** 64GB+ system memory
- **Storage:** 500GB+ (distributed storage)
- **Training time:** 6-10 hours total with DDP
- **Network:** InfiniBand or 10Gb+ Ethernet for multi-node

**Note:** DDP scripts are optimized for SLURM workload manager. For other schedulers, adapt the `.sh` scripts accordingly.


## 📁 Output Structure

```
urban_layout_inpainting/
├── vae_urban_ckpt.pth                # Standard VAE checkpoint
├── vae_urban_ddp_ckpt.pth            # DDP VAE checkpoint
├── ddpm_urban_inpainting_ckpt.pth    # Diffusion model checkpoint
├── vae_latents/                      # Pre-computed latents (standard)
│   ├── latent_0.pt
│   ├── latent_1.pt
│   └── ...
├── vae_ddp_latents/                  # Pre-computed latents (DDP)
│   ├── latent_0.pt
│   ├── latent_1.pt
│   └── ...
├── vae_samples/                      # VAE reconstructions during training
│   ├── recon_step_500.png
│   ├── recon_step_1000.png
│   └── ...
├── inpainting_samples/               # Generated inpainting samples
│   ├── samples_guidance7.5.png       # Grid of samples
│   ├── sample_0.png                  # Individual samples
│   ├── sample_0_temperature.png      # Temperature predictions
│   └── ...
└── dataset_validation/               # Dataset visualizations
    ├── sample_0.png                  # RGB + conditions
    ├── sample_1.png
    └── ...
```

---

## 🔬 Advanced Usage

### Custom Mask Types
Modify `hole_config.type` in config:
- `'random_square'`: Random position square masks
- `'center_square'`: Fixed center masks (for testing)
- `'street_blocks'`: Mask entire street blocks using OSM data (most realistic)

### Multi-Region Training
```yaml
dataset_params:
  train_regions: ['Dresden', 'Hamburg', 'Stuttgart', 'Munich']  # Add more cities
  eval_regions: ['Leipzig', 'Berlin']  # Multiple eval cities
```

### Auxiliary Prediction Tasks
Enable/disable auxiliary losses:
```yaml
train_params:
  seg_loss_weight: 0.1     # Set to 0.0 to disable OSM segmentation prediction
  env_loss_weight: 0.1     # Set to 0.0 to disable environmental prediction
  temperature_loss_weight: 1.0  # Temperature prediction (LST)
```

### Pipeline Arguments
```bash
# Skip already completed steps
python tools/run_pipeline.py --skip-vae --skip-validation

# Sample only (use existing checkpoints)
python tools/run_pipeline.py --sample-only --num_samples 32

# Custom config
python tools/run_pipeline.py --config config/diffusion_2.yml
```

### Distributed Training on SLURM
```bash
# Submit VAE training (auto-chains to diffusion)
sbatch tools/train_vae_urban_ddp.sh config/diffusion_1.yml

# Or submit diffusion training directly (if VAE exists)
sbatch tools/train_urban_inpainting_ddp.sh config/diffusion_1.yml

# Monitor job
squeue -u $USER
tail -f log/train_vae_urban_ddp.out-<job_id>

# Check GPU utilization
srun --jobid=<job_id> --pty nvidia-smi
```

---

## 📚 Documentation Files

- `QUICK_REFERENCE.md` (this file): Quick start guide
- `checklist/`: Detailed implementation checklists and notes
- `config/`: Configuration files for different experiments
  - `diffusion_1.yml`: Main training configuration
  - `diffusion_2.yml`: Alternative experiment setup
  - `benchmark_autoencoder.yml`: VAE benchmarking config
- `tools/`: Training and sampling scripts
  - `*_ddp.py`: Distributed Data Parallel versions
  - `*_ddp.sh`: SLURM submission scripts for DDP
  - `*.py`: Standard single-GPU versions

---

## 💡 Tips

### Performance Optimization
1. **Use DDP for training**: 4x speedup with 4 GPUs
2. **Pre-compute latents**: Set `save_latents: True` to speed up diffusion training
3. **Adjust num_workers**: Match to CPU cores available (4-8 typical)
4. **Use mixed precision**: Automatically enabled in DDP scripts

### Quality Improvements
1. **Train longer**: 200+ epochs for diffusion, 50+ for VAE
2. **Use street_blocks masks**: More realistic than random squares
3. **Tune guidance scale**: Test 5.0, 7.5, 10.0, 15.0 during sampling
4. **Multi-region training**: More diverse training data → better generalization

### Debugging
1. **Start with validation**: Always run `validate_dataset.py` first
2. **Check VAE quality**: Inspect `vae_samples/` before diffusion training
3. **Monitor losses**: Use tensorboard or log files in `log/`
4. **Visualize predictions**: Enable auxiliary task visualizations

### Configuration Management
1. **Create experiment configs**: Copy `diffusion_1.yml` to `diffusion_2.yml`
2. **Track changes**: Use `task_name` to separate experiments
3. **Version checkpoints**: Different configs → different checkpoint names

---

## 🆘 Getting Help

### Common Issues
1. **Import errors**: Ensure package installed with `pip install -e .`
2. **CUDA OOM**: Reduce batch sizes or patch size
3. **Slow training**: Use DDP or reduce dataset size for testing
4. **Poor results**: Check dataset quality, increase training time, tune guidance

### Log Files
- SLURM output: `log/<job_name>.out-<job_id>`
- SLURM errors: `log/<job_name>.err-<job_id>`
- Python logs: Check terminal output or redirect to file

### Validation Checklist
- [ ] Dataset validates without errors
- [ ] VAE reconstructions look sharp and realistic
- [ ] Training losses decrease steadily
- [ ] Inpainted regions blend seamlessly
- [ ] Conditions (OSM, temperature) are respected

---

## ✅ Success Indicators

### Dataset Validation
- ✅ All samples render without errors
- ✅ Masks cover appropriate regions (not too large/small)
- ✅ OSM layers show clear structures
- ✅ Environmental layers have valid ranges

### VAE Training
- ✅ Reconstruction loss < 0.2
- ✅ Visual samples are sharp and detailed
- ✅ No color artifacts or blurriness
- ✅ Latent space is smooth and continuous

### Diffusion Training  
- ✅ Total loss converges to ~0.01-0.03
- ✅ Mask loss weighted properly (2-3x total loss)
- ✅ Training stable without NaN/Inf
- ✅ Periodic samples show improvement over time

### Sampling/Inference
- ✅ Inpainted regions are realistic and detailed
- ✅ Boundaries blend smoothly (no visible seams)
- ✅ Buildings align with streets (OSM conditioning works)
- ✅ Temperature patterns match urban density
- ✅ Higher guidance scale increases condition adherence


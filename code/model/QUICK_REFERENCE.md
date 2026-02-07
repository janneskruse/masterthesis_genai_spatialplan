# Quick Reference: Two-Stage Urban Inpainting

## Overview

Two-stage latent diffusion model for urban inpainting:
- **Stage 1 (Semantic):** Generate semantic layouts (buildings, streets, heights, vegetation)
- **Stage 2 (Satellite):** Render satellite RGB imagery from generated semantics
- **Optional:** Temperature control via scalar conditioning + latent guidance

All scripts live in `tools/`, configs in `config/`, HPC SLURM jobs in `tools/slurm/`.

---

## Training Pipeline

All training uses DDP (Distributed Data Parallel) on SLURM with 4× A30 GPUs.

### 1. Prepare Patches

Cache dataset patches to disk for faster training.

```bash
python tools/prepare_patches.py --config two_stage_14.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--max_patches` | all | Limit patches (for testing) |
| `--skip_train` | off | Skip training split |
| `--skip_val` | off | Skip validation split |

### 2. Train VAEs (one per group)

```bash
# SLURM (recommended)
sbatch tools/slurm/train_semantic_vae_ddp.sh two_stage_14.yml
sbatch tools/slurm/train_satellite_vae_ddp.sh two_stage_14.yml
sbatch tools/slurm/train_environmental_vae_ddp.sh two_stage_14.yml

# Direct (single GPU)
torchrun --nproc_per_node=1 tools/train_vae_ddp.py --config two_stage_14.yml --mode semantic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--mode` | *required* | VAE group: `semantic`, `satellite`, `environmental` |
| `--load_checkpoint` | `None` | Resume from checkpoint |

### 3. Train Diffusion Models (one per stage)

```bash
# SLURM
sbatch tools/slurm/train_semantic_diffusion_inpainting_ddp.sh two_stage_14.yml
sbatch tools/slurm/train_satellite_diffusion_inpainting_ddp.sh two_stage_14.yml

# Direct
torchrun --nproc_per_node=1 tools/train_diffusion_inpainting_ddp.py --config two_stage_14.yml --mode semantic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--mode` | `semantic` | Diffusion stage: `semantic`, `satellite` |
| `--load_checkpoint` | `None` | Resume from checkpoint |

### 4. Train Temperature Predictors (optional)

```bash
# Full-resolution predictor
sbatch tools/slurm/train_temperature_predictor_ddp.sh two_stage_14.yml

# Latent-space predictor (for guidance)
sbatch tools/slurm/train_latent_temperature_predictor_ddp.sh two_stage_14.yml semantic
```

| Script | Key Flags |
|--------|-----------|
| `train_temperature_predictor_ddp.py` | `--config` |
| `train_latent_temperature_predictor_ddp.py` | `--config`, `--mode` (`semantic`/`satellite`), `--load_checkpoint` |

---

## Sampling

### Stage 1: Sample Semantic Layouts

```bash
python tools/sample_semantics_inpainting.py --config two_stage_14.yml \
    --num_samples 8 --guidance_scale 7.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `None` | Config file |
| `--mode` | `semantic` | Diffusion stage |
| `--num_samples` | `4` | Number of samples |
| `--guidance_scale` | `7.5` | CFG scale |
| `--overwrite_samples` | off | Overwrite existing (use run_idx=0) |
| `--target_temperature` | `None` | Target p95 temperature (°C) |
| `--latent_guidance_scale` | `None` | Latent guidance strength |
| `--control KEY=VALUE` | `None` | Scalar control (repeatable, e.g. `tmax=35.0`) |

### Stage 2: Render Satellite from Semantics

```bash
python tools/sample_satellite_from_semantics.py --config two_stage_14.yml \
    --guidance_scale 7.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--semantic_dir` | `None` | Stage 1 output dir (auto-detected if omitted) |
| `--guidance_scale` | `7.5` | CFG scale |
| `--num_samples` | all | Limit number of samples |
| `--overwrite_samples` | off | Overwrite existing |

---

## Validation

### Dataset Validation

Visualize dataset patches, layer statistics, and normalization.

```bash
python tools/validate_dataset.py --config two_stage_14.yml --num_samples 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--num_samples` | `5` | Samples to visualize |
| `--mode` | `default` | `default`, `vae:<group>`, or `diffusion:<stage>` |
| `--use_cached_patches` | off | Use pre-cached patches |
| `--recompute_layer_stats` | off | Force recompute statistics |
| `--no_plots` | off | Disable visualizations |

### VAE Validation

Generate reconstruction samples from trained VAE.

```bash
python tools/validate_vae.py --config two_stage_14.yml --mode semantic --num_samples 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--mode` | *required* | VAE group: `semantic`, `satellite`, `environmental` |
| `--num_samples` | `4` | Samples to reconstruct |
| `--overwrite_samples` | off | Overwrite existing |

### Inpainting Mask Coverage

Analyze mask type distribution, coverage statistics, and visualize examples.

```bash
python tools/validate_inpainting_mask_coverage.py --config two_stage_14.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--split` | `val` | Dataset split: `train` or `val` |
| `--num_samples` | `100` | Samples to analyze |
| `--samples_per_type` | `3` | Visual examples per mask type |
| `--no_plot` | off | Disable plotting |

### Temperature Predictor Validation

Validate full-resolution temperature predictor.

```bash
python tools/validate_temperature_predictor.py --config two_stage_14.yml --num_samples 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--num_samples` | `8` | Samples to validate |
| `--overwrite_samples` | off | Overwrite existing |

### Latent Temperature Predictor Validation

Validate latent-space temperature predictor (for guidance).

```bash
python tools/validate_latent_temperature_predictor.py --config two_stage_14.yml --mode semantic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `two_stage_4.yml` | Config file |
| `--mode` | `semantic` | Mode: `semantic` or `satellite` |
| `--num_samples` | `500` | Samples to validate |
| `--batch_size` | `32` | Batch size |
| `--overwrite_samples` | off | Overwrite existing |
| `--save_latents` / `--no_save_latents` | on | Save latent visualizations |

### Jacobian Sensitivity Analysis

Test latent→layer sensitivity for class-balanced loss weighting.

```bash
python tools/test_jacobian_sensitivity.py \
    --checkpoint /path/to/semantic_vae_ckpt.pth \
    --latent_dir /path/to/semantic_latents \
    --num_samples 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Path to VAE checkpoint |
| `--latent_dir` | *required* | Directory with saved latents |
| `--config` | `None` | Config file (optional) |
| `--device` | `cuda` | Device |
| `--num_samples` | `50` | Samples for sensitivity computation |
| `--layers` | auto | Layer names (e.g. `buildings streets vegetation`) |

---

## SLURM Scripts

All SLURM scripts use: `paula` partition, 4× A30 GPUs, 64G memory, conda env `genaiSpatialplan`.

| Script | Time | Runs |
|--------|------|------|
| `train_semantic_vae_ddp.sh` | 3h | `train_vae_ddp.py --mode semantic` |
| `train_satellite_vae_ddp.sh` | 3h | `train_vae_ddp.py --mode satellite` |
| `train_environmental_vae_ddp.sh` | 3h | `train_vae_ddp.py --mode environmental` |
| `train_semantic_diffusion_inpainting_ddp.sh` | 3h | `train_diffusion_inpainting_ddp.py --mode semantic` |
| `train_satellite_diffusion_inpainting_ddp.sh` | 6h | `train_diffusion_inpainting_ddp.py --mode satellite` |
| `train_temperature_predictor_ddp.sh` | 3h | `train_temperature_predictor_ddp.py` |
| `train_latent_temperature_predictor_ddp.sh` | 2h | `train_latent_temperature_predictor_ddp.py` |

Usage: `sbatch tools/slurm/<script>.sh <config> [checkpoint]`

```bash
# Example: train semantic VAE with two_stage_14 config
sbatch tools/slurm/train_semantic_vae_ddp.sh two_stage_14.yml

# Monitor
squeue -u $USER
tail -f tools/log/train_vae_semantic_ddp.out-<job_id>
```

---

## Output Structure

```
<results_dir>/<task_name>/
├── patches/                          # Cached dataset patches
├── satellite_vae_ckpt.pth            # VAE checkpoints
├── semantic_vae_ckpt.pth
├── environmental_vae_ckpt.pth
├── satellite_latents/                # Pre-computed VAE latents
├── semantic_latents/
├── environmental_latents/
├── satellite_samples/                # VAE reconstruction samples
├── semantic_samples/
├── environmental_samples/
├── semantic_diffusion_ckpt.pth       # Diffusion checkpoints
├── satellite_diffusion_ckpt.pth
├── semantic_diffusion_samples/       # Diffusion training samples
├── satellite_diffusion_samples/
├── latent_temperature_predictor_*.pth
├── temperature_predictor_best.pth
├── semantic_validation/              # Validation outputs
├── satellite_validation/
└── environmental_validation/
```

---

## Key Config Sections

```yaml
# config/two_stage_14.yml (example)

layers:             # Global layer registry (types, normalization, loss)
inpainting_params:  # Mask generation (mixed strategy with fallback)
dataset_params:     # Regions, resolution, patch size
temperature_control:  # Guidance + hard check settings
scalar_controls:      # Temperature, vegetation, building coverage, height
vae_groups:           # VAE architectures per group (semantic/satellite/environmental)
diffusion_params:     # Timesteps, beta schedule, prediction type
diffusion_stages:     # Stage-specific U-Net, conditioning, inpainting, CFG
train_params:         # Seeds, epochs, batch sizes, learning rates, EMA, sampling
```

### Reusing Checkpoints Across Experiments

```yaml
train_params:
  existing_paths:
    patches: /path/to/previous/patches
    latents:
      semantic: /path/to/semantic_latents
    vae_checkpoints:
      semantic: /path/to/semantic_vae_ckpt.pth
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or `patch_size_m` in config |
| Import errors | Run `pip install -e .` from project root |
| No valid patches | Lower `min_valid_percent` or increase `stride_overlap` |
| Blurry VAE output | Check `kl_weight`, enable `use_perceptual` for satellite |
| Visible seams | Enable seam strategy (`dilate` or `feather`) in inpainting config |
| Weak conditioning | Increase `cf_guidance_scale` during sampling |
| NaN loss | Check `clamp_range`, reduce learning rate |

### Log Files
- SLURM output: `log/<job_name>.out-<job_id>`
- SLURM errors: `log/<job_name>.err-<job_id>`
- Python logs: Check terminal output or redirect to file


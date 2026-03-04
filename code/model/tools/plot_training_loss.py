"""
==============================================================================
Plot training losses from SLURM log files.
==============================================================================
"""

###### import libraries ######
# Standard libraries
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Data Science/ML libraries
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Local imports
from helpers.load_configs import add_config_arguments, load_configs


def parse_log_file(log_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse a SLURM training log and extract all epoch-level losses.
    
    Returns:
        Dict mapping loss name -> list of (epoch, value) tuples.
        e.g. {'VAE Loss': [(1, 0.69), (2, 0.32), ...],
              'Recon': [(1, 0.05), ...],
              'buildings_bce': [(1, 0.034), ...]}
    """
    losses: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    
    # ── Regex patterns ──
    # Old format:  ✓ Epoch 1/50 | VAE Loss: 0.6981
    # Also matches: ✓ Epoch 1/50 | VAE Loss: 0.6981 | Disc Loss: 0.5678
    old_epoch_re = re.compile(
        r'Epoch\s+(\d+)/(\d+)\s*\|\s*VAE Loss:\s*([\d.]+)'
        r'(?:\s*\|\s*Disc Loss:\s*([\d.]+))?'
    )
    
    # New format:  ✓ Epoch 1/50 | Total: 0.0412 | Recon: 0.0389 | KL: 0.000023
    # Captures all key: value pairs after Epoch X/N
    new_epoch_re = re.compile(
        r'Epoch\s+(\d+)/(\d+)\s*\|(.+)'
    )
    
    # Key-value pair inside a line:  Name: 0.1234
    kv_re = re.compile(r'([\w\s]+?):\s*([\d.eE+-]+)')
    
    # Diffusion format:  ✓ Epoch 1/300 | Noise Loss: 0.1706
    # (already captured by new_epoch_re)
    
    # Per-layer loss lines:    buildings_bce                   0.034521
    layer_loss_re = re.compile(
        r'^\s{2,}(\S+)\s+([\d.eE+-]+)\s*$'
    )
    
    # Track whether we're in a "Per-layer losses:" block
    in_per_layer_block = False
    current_epoch: Optional[int] = None
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            
            # ── Try old format first ──
            m_old = old_epoch_re.search(stripped)
            if m_old:
                epoch = int(m_old.group(1))
                current_epoch = epoch
                in_per_layer_block = False
                
                vae_loss = float(m_old.group(3))
                losses['VAE Loss'].append((epoch, vae_loss))
                
                if m_old.group(4):
                    disc_loss = float(m_old.group(4))
                    losses['Disc Loss'].append((epoch, disc_loss))
                
                # Check if this is actually the new format (has more |-separated pairs)
                # by looking for Total/Recon/KL which old format doesn't have
                if 'Total:' in stripped or 'Recon:' in stripped or 'Noise Loss:' in stripped:
                    # Actually new format — reparse with kv_re
                    m_new = new_epoch_re.search(stripped)
                    if m_new:
                        # Clear old-format entry we just added
                        losses['VAE Loss'] = [
                            x for x in losses['VAE Loss'] if x[0] != epoch
                        ]
                        rest = m_new.group(3)
                        for kv in kv_re.finditer(rest):
                            key = kv.group(1).strip()
                            val = float(kv.group(2))
                            losses[key].append((epoch, val))
                continue
            
            # ── Try new format (covers diffusion "Noise Loss" too) ──
            m_new = new_epoch_re.search(stripped)
            if m_new:
                epoch = int(m_new.group(1))
                current_epoch = epoch
                in_per_layer_block = False
                
                rest = m_new.group(3)
                for kv in kv_re.finditer(rest):
                    key = kv.group(1).strip()
                    val = float(kv.group(2))
                    losses[key].append((epoch, val))
                continue
            
            # ── Per-layer block header ──
            if 'Per-layer losses:' in stripped:
                in_per_layer_block = True
                continue
            
            # ── Per-layer loss line ──
            if in_per_layer_block and current_epoch is not None:
                m_layer = layer_loss_re.match(line)
                if m_layer:
                    key = m_layer.group(1).strip()
                    val = float(m_layer.group(2))
                    losses[key].append((current_epoch, val))
                else:
                    # Non-matching line ends the block
                    if stripped:
                        in_per_layer_block = False
    
    return dict(losses)


def detect_model_type(log_path: str) -> str:
    """Detect training type from filename."""
    name = Path(log_path).stem.lower()
    if 'diffusion' in name:
        return 'Diffusion'
    elif 'cvae' in name:
        return 'CVAE'
    elif 'vae' in name:
        # Detect mode from filename
        if 'satellite' in name:
            return 'Satellite VAE'
        elif 'semantic' in name:
            return 'Semantic VAE'
        elif 'environmental' in name:
            return 'Environmental VAE'
        return 'VAE'
    return 'Training'


def plot_losses(
    losses: Dict[str, List[Tuple[int, float]]],
    title: str = 'Training Loss',
    save_path: Optional[str] = None,
    log_scale: bool = False,
) -> None:
    """
    Plot all parsed losses.
    
    Creates a main plot for the primary loss, and a secondary subplot
    for per-layer / component losses if present.
    """
    if not losses:
        print("No losses found in log file.")
        return
    
    # Separate primary losses from per-layer losses
    primary_keys = []
    layer_keys = []
    
    for key in losses:
        # Primary losses are things like VAE Loss, Total, Recon, KL, Disc, Noise Loss, etc.
        if key in ('VAE Loss', 'Total', 'Recon', 'KL', 'Disc', 'Disc Loss',
                    'Percep', 'Gen', 'Noise Loss'):
            primary_keys.append(key)
        else:
            layer_keys.append(key)
    
    has_layers = len(layer_keys) > 0
    
    if has_layers:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [2, 1.5]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    
    # ── Primary losses plot ──
    # Sample colors from rocket palette, spaced evenly across the range
    rocket_palette = sns.color_palette("rocket", n_colors=max(len(primary_keys), 6))
    
    # Map known loss types to fixed positions in the rocket palette + linestyles
    style_map = {
        'VAE Loss':   {'color_idx': 0, 'linewidth': 2.5, 'linestyle': '-'},
        'Total':      {'color_idx': 0, 'linewidth': 2.5, 'linestyle': '-'},
        'Recon':      {'color_idx': 1, 'linewidth': 2.0, 'linestyle': '-'},
        'KL':         {'color_idx': 2, 'linewidth': 1.5, 'linestyle': '--'},
        'Percep':     {'color_idx': 3, 'linewidth': 1.5, 'linestyle': '--'},
        'Gen':        {'color_idx': 4, 'linewidth': 1.5, 'linestyle': ':'},
        'Disc':       {'color_idx': 5, 'linewidth': 1.5, 'linestyle': ':'},
        'Disc Loss':  {'color_idx': 5, 'linewidth': 1.5, 'linestyle': ':'},
        'Noise Loss': {'color_idx': 0, 'linewidth': 2.5, 'linestyle': '-'},
    }
    
    for idx, key in enumerate(primary_keys):
        epochs, values = zip(*losses[key])
        style = style_map.get(key, {'color_idx': idx, 'linewidth': 1.5, 'linestyle': '-'})
        color = rocket_palette[style['color_idx'] % len(rocket_palette)]
        ax1.plot(epochs, values, label=key, 
                 color=color, linewidth=style['linewidth'],
                 linestyle=style['linestyle'], marker='.', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    if log_scale:
        ax1.set_yscale('log')
    
    # ── Per-layer losses plot ──
    if has_layers:
        # Use rocket palette for per-layer losses
        layer_palette = sns.color_palette("rocket", n_colors=max(len(layer_keys), 2))
        
        for idx, key in enumerate(sorted(layer_keys)):
            epochs, values = zip(*losses[key])
            color = layer_palette[idx % len(layer_palette)]
            ax2.plot(epochs, values, label=key, color=color,
                     linewidth=1.5, marker='.', markersize=3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Per-Layer Losses', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        if log_scale:
            ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    log_paths: List[str],
    save_path: Optional[str] = None,
    log_scale: bool = False,
) -> None:
    """
    Plot primary loss curves from multiple log files for comparison.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    compare_palette = sns.color_palette("rocket", n_colors=max(len(log_paths), 2))
    
    for idx, log_path in enumerate(log_paths):
        losses = parse_log_file(log_path)
        model_type = detect_model_type(log_path)
        job_id = Path(log_path).stem.split('-')[-1] if '-' in Path(log_path).stem else ''
        label = f'{model_type} ({job_id})' if job_id else model_type
        
        # Pick the primary loss key
        primary_key = None
        for candidate in ('VAE Loss', 'Total', 'Noise Loss'):
            if candidate in losses:
                primary_key = candidate
                break
        
        if primary_key is None:
            print(f"⚠ No recognized primary loss in {log_path}")
            continue
        
        epochs, values = zip(*losses[primary_key])
        ax.plot(epochs, values, label=f'{label} — {primary_key}',
                color=compare_palette[idx % len(compare_palette)], linewidth=2,
                marker='.', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(losses: Dict[str, List[Tuple[int, float]]]) -> None:
    """Print a summary table of parsed losses."""
    print(f"\n{'='*60}")
    print(f"{'Loss Component':<30s} {'Epochs':>8s} {'First':>10s} {'Last':>10s} {'Min':>10s}")
    print(f"{'='*60}")
    
    for key in sorted(losses.keys()):
        data = losses[key]
        epochs = [x[0] for x in data]
        values = [x[1] for x in data]
        print(f"{key:<30s} {len(data):>8d} {values[0]:>10.6f} {values[-1]:>10.6f} {min(values):>10.6f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot training losses from SLURM log files'
    )
    
    add_config_arguments(parser)
    
    parser.add_argument(
        'log_files', nargs='+', type=str,
        help='Path(s) to SLURM .out log file(s)'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save plot to file instead of showing'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare primary losses across multiple log files'
    )
    parser.add_argument(
        '--log_scale', action='store_true',
        help='Use log scale for y-axis'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print summary table of parsed losses'
    )
    
    args = parser.parse_args()
    
    config = load_configs()
    
    repo_dir = Path(config["repo_dir"])
    task_name = config.get('train_params', {}).get('task_name', 'urban_inpainting')
    results_dir = repo_dir / 'results' / task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare and len(args.log_files) > 1:
        save_path = args.save
        if save_path is None:
            save_path = str(results_dir / 'loss_comparison.png')
        plot_comparison(args.log_files, save_path=save_path, log_scale=args.log_scale)
    else:
        for log_path in args.log_files:
            print(f"\nParsing: {log_path}")
            losses = parse_log_file(log_path)
            
            if not losses:
                print(f"⚠ No epoch losses found in {log_path}")
                continue
            
            model_type = detect_model_type(log_path)
            job_id = Path(log_path).stem.split('-')[-1] if '-' in Path(log_path).stem else ''
            title = f'{model_type} Training Loss'
            if job_id:
                title += f' (Job {job_id})'
            
            if args.summary:
                print_summary(losses)
            
            # Default save path: repo_dir/results/<task_name>/<log_stem>.png
            save_path = args.save
            if save_path is None:
                save_path = str(results_dir / f'{Path(log_path).stem}.png')
            
            plot_losses(losses, title=title, save_path=save_path, log_scale=args.log_scale)

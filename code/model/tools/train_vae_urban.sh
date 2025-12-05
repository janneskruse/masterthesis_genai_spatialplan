#!/bin/bash

#SBATCH --time=2:30:00
#SBATCH --job-name="train_vae_urban"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a30:1
#SBATCH --mem=50G
#SBATCH --partition=paula
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.err-%j

# Create log directory if it doesn't exist
mkdir -p log

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print environment info
echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Print GPU info
nvidia-smi

echo "=================================================="
echo "Starting VAE training..."
echo "=================================================="

# Execute with unbuffered output
python3 -u train_vae_urban.py

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
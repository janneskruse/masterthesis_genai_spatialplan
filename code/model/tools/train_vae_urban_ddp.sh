#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --job-name="train_vae_urban_ddp"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # One task per GPU
#SBATCH --cpus-per-task=2    # 8 CPUs / 4 GPUs
#SBATCH --gpus=a30:4
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.err-%j

# Default config if not provided
if [ "$1" = "--config" ] && [ -n "$2" ]; then
    CONFIG_PATH=$2
else
    CONFIG_PATH=${1:-code/model/config/diffusion_1.yml}
fi

mkdir -p log

source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Install package in editable mode for proper imports
cd /home/sc.uni-leipzig.de/${USER}/masterthesis_genai_spatialplan
pip install -e . --quiet
cd - # return to previous directory

# Get IPv4 address explicitly
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Distributed training configuration
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "=================================================="
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Starting DDP training with $WORLD_SIZE GPUs"
echo "Passing config: $CONFIG_PATH"
echo "=================================================="

# Launch with srun and set CUDA_VISIBLE_DEVICES per process
srun bash -c "
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    python3 -u train_vae_urban_ddp.py --config $CONFIG_PATH
"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="

# hand in diffusion training shell script to slurm
sbatch train_vae_urban_ddp.sh --config $CONFIG_PATH
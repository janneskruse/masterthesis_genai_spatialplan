#!/bin/bash

#SBATCH --time=24:00:00
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

mkdir -p log

source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Set MASTER_ADDR and MASTER_PORT for rank 0
# SLURM will propagate these to all tasks via srun
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Distributed training configuration
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL configuration (--> for debugging)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "=================================================="
echo "Starting DDP training with $WORLD_SIZE GPUs"
echo "=================================================="

# Launch with srun (handles process spawning)
srun python3 -u train_vae_urban_ddp.py

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
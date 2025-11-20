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

# Set distributed environment
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export GLOO_SOCKET_IFNAME=eth0

# Force IPv4 with multiple fallback strategies
MASTER_ADDR=$(hostname -I | awk '{print $1}')  # -I instead of -i gets all addresses
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n 1)
fi
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR="127.0.0.1"  # Fallback for single node
fi

# log master address
echo "Using MASTER_ADDR: $MASTER_ADDR"

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=================================================="
echo "Starting DDP training with $WORLD_SIZE GPUs"
echo "=================================================="

# Launch with srun (handles process spawning)
srun python3 -u train_vae_urban_ddp.py

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --job-name="train_vae_environmental_ddp"
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

# Parse arguments
CONFIG_PATH="two_stage_14.yml"
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        *)
            # Assume first positional arg is config
            CONFIG_PATH="$1"
            shift
            ;;
    esac
done

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
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
else
    echo "Training from scratch (no checkpoint)"
fi
echo "=================================================="

# Build Python command dynamically
PYTHON_CMD="python3 -u ../train_vae_ddp.py --config $CONFIG_PATH --mode environmental"
if [ -n "$CHECKPOINT_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --load_checkpoint $CHECKPOINT_PATH"
fi

# Launch with srun and set CUDA_VISIBLE_DEVICES per process
srun bash -c "
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    $PYTHON_CMD
"

# Capture the exit code of srun/python
EXIT_CODE=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "Training exit code: $EXIT_CODE"
echo "=================================================="
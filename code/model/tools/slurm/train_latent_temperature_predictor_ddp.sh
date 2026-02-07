#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --job-name="train_latent_temperature_predictor"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a30:4
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.err-%j

# Parse arguments
CONFIG_PATH="two_stage_14.yml"
MODE="semantic"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

mkdir -p log

source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Install package
cd /home/sc.uni-leipzig.de/${USER}/masterthesis_genai_spatialplan
pip install -e . --quiet
cd -

# DDP setup
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "=================================================="
echo "Latent Temperature predictor Training"
echo "=================================================="
echo "Config: $CONFIG_PATH"
echo "Mode: $MODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "=================================================="

srun bash -c "
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    python3 -u ../train_latent_temperature_predictor_ddp.py --config $CONFIG_PATH --mode $MODE
"

EXIT_CODE=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
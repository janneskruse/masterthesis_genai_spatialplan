#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --job-name="Combine_Datasets"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/combine_datasets.%j.txt"

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print region to be processed
echo "Processing region: $REGION"

# Run the combination script for the given region
python3 -u ../combine_datasets.py --REGION ${REGION}
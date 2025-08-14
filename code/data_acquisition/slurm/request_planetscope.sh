#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --job-name="Request_PlanetScope"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL 
#SBATCH -o "outputs/request_planetscope.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print region to be processed
echo "Processing region: $REGION"

# Run the planetscop request script
python3 -u request_planetscope.py --REGION ${REGION} --LANDSAT_ZARR_NAME ${LANDSAT_ZARR_NAME}
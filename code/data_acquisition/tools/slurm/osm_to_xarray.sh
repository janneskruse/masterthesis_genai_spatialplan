#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --job-name="osm_to_xarray"
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/osm_to_xarray.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print region to be processed
echo "Processing region: $REGION"

# Execute the OSM to Xarray script
python3 -u ../osm_to_xarray.py --REGION ${REGION}

# Check if all upstream steps are done and submit combine job if so
echo "Checking whether combine_datasets can be submitted..."
python3 -u ../try_submit_combine.py --REGION ${REGION}
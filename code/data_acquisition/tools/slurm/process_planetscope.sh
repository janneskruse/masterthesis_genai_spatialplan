#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --job-name="Process_PlanetScope"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/process_planetscope.%j.txt"

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print region to be processed
echo "Processing region: $REGION"
echo "Filenames: $FILENAMES"

# Execute the process_planetscope.py script (handles multiprocessing + combine internally)
python3 -u ../process_planetscope.py --REGION ${REGION} --LANDSAT_ZARR_NAME ${LANDSAT_ZARR_NAME} --FILENAMES "${FILENAMES}"

# Check if all upstream steps are done and submit combine job if so
echo "Checking whether combine_datasets can be submitted..."
python3 -u ../try_submit_combine.py --REGION ${REGION}
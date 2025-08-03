#!/bin/bash

#SBATCH --time=0:01:00
#SBATCH --job-name="Landsat_to_xarray"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/landsat_to_xarray.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Execute the landsat_to_xarray.py script
python3 landsat_to_xarray.py --region=${region}
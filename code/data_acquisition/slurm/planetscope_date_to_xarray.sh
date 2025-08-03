#!/bin/bash

#SBATCH --time=0:01:00
#SBATCH --job-name="Planetscope_to_Xarray"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL 
#SBATCH -o "outputs/planetscope_to_xarray.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Run the planetscope_date_to_xarray.py script
python3 -u planetscope_date_to_xarray.py --region ${region} --landsat_zarr_name ${landsat_zarr_name} --filename ${filename}
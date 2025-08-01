#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --job-name="Process_PlanetScope"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/process_planetscope.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Find the repository root directory to locate the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
CONFIG_FILE="$REPO_ROOT/config.yml"

# Load the config variables
big_data_storage_path=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "big_data_storage_path")
regions=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "regions")

# Get the filenames from request_planetscope
# eval $(python ${REPO_ROOT}/code/helpers/get_timeid_filenames_planet.py "$landsat_zarr_name" "$region" "$big_data_storage_path")
# echo $filenames

# For each filename, submit a job for the planetscope processing
job_ids=()
for filename in $filenames; do
    echo "Processing file: $filename"
    file_job=$(sbatch --parsable --export=region="$region",filename="$filename",landsat_zarr_name="$landsat_zarr_name" ./planetscope_date_to_xarray.sh)
    job_ids+=($file_job)

# Add all jobs as dependency to finish on submitting the combine job
dependency_string=$(IFS=:; echo "${job_ids[*]}")
combine_job=$(sbatch --parsable --export=region="$region",filenames="$filenames",landsat_zarr_name="$landsat_zarr_name",region_filenames_json="$region_filenames_json" --dependency=afterok:$dependency_string ./combine_planetscope_xarrays.sh)
echo "Submitted combine job: $combine_job"

done

    


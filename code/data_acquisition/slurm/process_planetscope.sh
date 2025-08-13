#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --job-name="Process_PlanetScope"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/process_planetscope.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# if [ -n $SLURM_JOB_ID ];  then
#     # check the original location through scontrol and $SLURM_JOB_ID
#     SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}')
# else
#     # otherwise: started with bash. Get the real location.
#     SCRIPT_PATH=$(realpath $0)
# fi

# # Find the repository root directory to locate the config file
# SCRIPT_DIR=$( cd -- "--" "$(dirname -- "$SCRIPT_PATH")" &> /dev/null && pwd )
# REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
# CONFIG_FILE="$REPO_ROOT/config.yml"

# # Load the config variables
# big_data_storage_path=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "big_data_storage_path")
# regions=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "regions")

# # Get the filenames from request_planetscope
# # eval $(python ${REPO_ROOT}/code/helpers/get_timeid_filenames_planet.py "$landsat_zarr_name" "$region" "$big_data_storage_path")
# # echo $filenames

# Parse FILENAMES
echo "Filenames: $FILENAMES"
if [[ "$FILENAMES" == *:* ]]; then
    # colon-separated values
    IFS=':' read -r -a FILENAME_ARRAY <<< "$FILENAMES"
elif printf '%s' "$FILENAMES" | grep -q $'\n'; then
    # newline-separated values
    mapfile -t FILENAME_ARRAY < <(printf '%s' "$FILENAMES")
else
    # assume space-separated
    read -ra FILENAME_ARRAY <<< "$FILENAMES"
fi

# Print for debugging
echo "FILENAME_ARRAY: ${FILENAME_ARRAY[@]}"

job_ids=()
for filename in "${FILENAME_ARRAY[@]}"; do
    # trim whitespace
    filename=$(echo "$filename" | xargs)

    echo "Processing file: $filename"
    file_job=$(sbatch --parsable --export=REGION="$REGION",FILENAME="$filename",LANDSAT_ZARR_NAME="$LANDSAT_ZARR_NAME" ./planetscope_date_to_xarray.sh)
    job_ids+=($file_job)
done

# # For each filename, submit a job for the planetscope processing
# job_ids=()
# for filename in $FILENAMES; do
#     echo "Processing file: $filename"
#     file_job=$(sbatch --parsable --export=REGION="$REGION",FILENAME="$filename",LANDSAT_ZARR_NAME="$LANDSAT_ZARR_NAME" ./planetscope_date_to_xarray.sh)
#     job_ids+=($file_job)

# Add all jobs as dependency to finish on submitting the combine job
dependency_string=$(IFS=:; echo "${job_ids[*]}")
combine_job=$(sbatch --parsable --export=REGION="$REGION",FILENAMES="$FILENAMES",LANDSAT_ZARR_NAME="$LANDSAT_ZARR_NAME",REGION_FILENAMES_JSON="$REGION_FILENAMES_JSON" --dependency=afterok:$dependency_string ./combine_planetscope_xarrays.sh)
echo "Submitted combine job: $combine_job"
echo "Dependency string: $dependency_string"

done

    


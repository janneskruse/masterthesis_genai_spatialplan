#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name="Process_PlanetScope"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "process_planetscope.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

python3 -u combine_region_datasets.py --region "$region" --landsat_zarr_name "$landsat_zarr_name" --osm_zarr_name "$osm_zarr_name" --planet_zarr_name "$planet_zarr_name"

# Find the repository root directory to locate the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
CONFIG_FILE="$REPO_ROOT/config.yml"

# Load the config variables
regions=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "regions")

#check if for all regions the job has finished (the files exist)
for region in "${regions[@]}"; do
    echo "Processing region: $region"
    
    # Extract filename for the region
    processed_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".processed_zarr_name")

    # Check if the zarr files exist
    if [ -f "$processed_zarr_name" ]; then
        echo "Processed Zarr file for region $region exists"
    else
        echo "Missing processed Zarr file for region $region. Please check the processing steps."
        exit 1
    fi

# Submit the combine job
combine_job=$(sbatch --parsable --export=region_filenames_json="$region_filenames_json"  ./combine_datasets.sh)
echo "Submitted combine job for all regions: $combine_job"
done

    


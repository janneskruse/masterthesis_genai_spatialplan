#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --job-name="osm_to_xarray"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/osm_to_xarray_${REGION}.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Execute the OSM to Xarray script
python3 -u osm_to_xarray.py --REGION ${REGION}


#check if for all files of the region the jobs have finished (the files exist) and if so, submit the combine job
# Extract filenames for the region
landsat_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".landsat_zarr_name")
osm_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".osm_zarr_name")
planet_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".planet_zarr_name")
processed_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".processed_zarr_name")

# Check if the zarr files exist
if [[ -f "$landsat_zarr_name" && -f "$osm_zarr_name" && -f "$planet_zarr_name" ]]; then
    echo "All required files for region $REGION exist"
else
    echo "Missing files for region $REGION. Please check the processing steps."
    exit 1
fi

# If the processed zarr file already exists, skip the region
if [ -f "$processed_zarr_name" ]; then
    echo "Processed Zarr file for region $region already exists: $processed_zarr_name"
    continue
else
    echo "Processed Zarr file does not exist, proceeding with combination."
    
    # Check if a combine job is already running for this region
    existing_job=$(squeue -u $USER --name="combine_region_$REGION" --noheader --format="%i" 2>/dev/null)
    
    if [ -n "$existing_job" ]; then
        echo "Combine job already running for region $REGION (Job ID: $existing_job). Skipping submission."
    else
        # Submit the combine job for the region
        combine_job=$(sbatch --parsable --job-name="combine_region_$REGION" --export=REGION="$REGION",LANDSAT_ZARR_NAME="$landsat_zarr_name",OSM_ZARR_NAME="$osm_zarr_name",PLANET_ZARR_NAME="$planet_zarr_name",REGION_FILENAMES_JSON="$REGION_FILENAMES_JSON" ./combine_region_datasets.sh)
        echo "Submitted combine job for region $REGION: $combine_job"
    fi
fi

# Find the repository root directory to locate the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
CONFIG_FILE="$REPO_ROOT/config.yml"

# Load the config variables
regions=($(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "regions" | tr ',' ' '))

# Check if for all regions the job has finished (the files exist)
for region in "${regions[@]}"; do
    echo "Processing region: $region"
    
    # Extract filename for the region
    processed_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$region\".processed_zarr_name")

    # Check if the zarr files exist
    if [ -f "$processed_zarr_name" ]; then
        echo "Processed Zarr file for region $region exists"
    else
        echo "Missing processed Zarr file for region $region. Please check the processing steps."
        exit 1
    fi
done

# Submit the combine job
combine_job=$(sbatch --parsable --export=REGION_FILENAMES_JSON="$REGION_FILENAMES_JSON"  ./combine_datasets.sh)
echo "Submitted combine job for all regions: $combine_job"
done
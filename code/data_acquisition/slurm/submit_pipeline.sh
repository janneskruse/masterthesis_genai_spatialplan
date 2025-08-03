#!/bin/bash

#SBATCH --time=0:01:00
#SBATCH --job-name="SubmitPipeline"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL 
#SBATCH -o "outputs/submit_pipeline.%j.txt"

# Load Anaconda environment
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Find the repository root directory to locate the config file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
CONFIG_FILE="$REPO_ROOT/config.yml"

# Load vars from config.yaml
big_data_storage_path=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "big_data_storage_path")
min_temperature=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "temperature_day_filter.min")
max_cloud_cover=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "landsat_query.max_cloud_coverage")
start_year=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "temperature_day_filter.years.start")
end_year=$(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "temperature_day_filter.years.end")
regions=($(python ${REPO_ROOT}/code/helpers/read_yaml.py "$CONFIG_FILE" "regions" | tr ',' ' '))

# Construct the input filename
input_filename="${big_data_storage_path}/processed/input_config_ge${min_temperature}_cc${max_cloud_cover}_${start_year}_${end_year}.zarr"
echo "Input filename: $input_filename"

# Check if the input file already exists
if [ -f "$input_filename" ]; then
    echo "Input file already exists: $input_filename"
    exit 0
else
    echo "Input file does not exist, proceeding with data acquisition."
fi

region_filenames_json=$(python ${REPO_ROOT}/code/helpers/get_region_filenames.py "$CONFIG_FILE")

# Submit jobs for each region
for region in "${regions[@]}"; do
    echo "Processing region: $region"
    
    # Extract filenames for the region
    landsat_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".landsat_zarr_name")
    osm_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".osm_zarr_name")
    planet_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".planet_zarr_name")
    processed_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".processed_zarr_name")

    # If the processed zarr file already exists, skip the region
    if [ -f "$processed_zarr_name" ]; then
        echo "Processed Zarr file for region $region already exists: $processed_zarr_name"
        continue
    fi

    # Submit Landsat job
    if [ ! -f "$landsat_zarr_name" ]; then
        echo "Submitting Landsat job for $region (file: $landsat_zarr_name)"
        landsat_job_id=$(sbatch --parsable --export=region="$region",landsat_zarr_name="$landsat_zarr_name" ./landsat_to_xarray.sh)
        echo "Landsat job ID: $landsat_job_id"
    else
        echo "Landsat file already exists for $region: $landsat_zarr_name"
        landsat_job_id=""
    fi

    # Submit OSM job
    if [ ! -f "$osm_zarr_name" ]; then
        echo "Submitting OSM job for $region (file: $osm_zarr_name)"
        osm_job_id=$(sbatch --parsable --export=region="$region",osm_zarr_name="$osm_zarr_name",region_filenames_json="$region_filenames_json" ./osm_to_xarray.sh)
        echo "OSM job ID: $osm_job_id"
    else
        echo "OSM file already exists for $region: $osm_zarr_name"
        osm_job_id=""
    fi

    # Submit PlanetScope job with dependency on Landsat job
    if [ ! -f "$planet_zarr_name" ]; then
        echo "Submitting PlanetScope job for $region (file: $planet_zarr_name)"
        
        # Only add dependency if Landsat job was actually submitted
        if [ -n "$landsat_job_id" ]; then
            planet_request_job_id=$(sbatch --dependency=afterok:$landsat_job_id --export=region="$region",landsat_zarr_name="$landsat_zarr_name",planet_zarr_name="$planet_zarr_name",region_filenames_json="$region_filenames_json" ./request_planetscope.sh)
        else
            planet_request_job_id=$(sbatch --export=region="$region",landsat_zarr_name="$landsat_zarr_name",planet_zarr_name="$planet_zarr_name",region_filenames_json="$region_filenames_json" ./request_planetscope.sh)
        fi
        echo "PlanetScope job ID: $planet_request_job_id"
    else
        echo "PlanetScope file already exists for $region: $planet_zarr_name"
        planet_request_job_id=""
    fi

    # if all exists but not the processed zarr, submit the combine job
    if [ -z "$landsat_job_id" ] && [ -z "$osm_job_id" ] && [ -z "$planet_request_job_id" ] && [ ! -f "$processed_zarr_name" ]; then
        echo "Submitting combine job for $region (file: $processed_zarr_name)"
    
        # Check if a combine job is already running for this region
        existing_job=$(squeue -u $USER --name="combine_region_$region" --noheader --format="%i" 2>/dev/null)
        
        if [ -n "$existing_job" ]; then
            echo "Combine job already running for region $region (Job ID: $existing_job). Skipping submission."
        else
            # Submit the combine job for the region
            combine_job=$(sbatch --parsable --job-name="combine_region_$region" --export=region="$region",landsat_zarr_name="$landsat_zarr_name",osm_zarr_name="$osm_zarr_name",planet_zarr_name="$planet_zarr_name",region_filenames_json="$region_filenames_json" ./combine_region_datasets.sh)
            echo "Submitted combine job for region $region: $combine_job"
        fi
    
    echo "---"
done
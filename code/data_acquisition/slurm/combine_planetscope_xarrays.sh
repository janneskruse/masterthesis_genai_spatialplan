#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name="Combine_PlanetScope_Xarrays"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "outputs/combine_planetscope_xarrays.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Print region to be processed
echo "Processing region: $REGION"

# Run the combine job for the planetscope datasets
python3 -u combine_planetscope_xarrays.py --region "$REGION" --landsat_zarr_name "$LANDSAT_ZARR_NAME" --filenames="$FILENAMES"

# Print Region Filenames JSON
echo "Region Filenames JSON: $REGION_FILENAMES_JSON"

# Extract filenames for the region
landsat_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".landsat_zarr_name")
osm_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".osm_zarr_name")
planet_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".planet_zarr_name")
processed_zarr_name=$(echo "$REGION_FILENAMES_JSON" | jq -r ".\"$REGION\".processed_zarr_name")

# Print the filenames for debugging
echo "Landsat Zarr Name: $landsat_zarr_name"
echo "OSM Zarr Name: $osm_zarr_name"
echo "Planet Zarr Name: $planet_zarr_name"
echo "Processed Zarr Name: $processed_zarr_name"

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
    existing_job=$(squeue -u $USER --name="combine_region_$region" --noheader --format="%i" 2>/dev/null)
    
    if [ -n "$existing_job" ]; then
        echo "Combine job already running for region $region (Job ID: $existing_job). Skipping submission."
    else
        # Submit the combine job for the region
        combine_job=$(sbatch --parsable --job-name="combine_region_$region" --export=ALL ./combine_region_datasets.sh)
        echo "Submitted combine job for region $region: $combine_job"
    fi
fi

    


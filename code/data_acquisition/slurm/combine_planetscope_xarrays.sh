#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --job-name="Process_PlanetScope"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=paul
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL
#SBATCH -o "process_planetscope.%j.txt" #j for the job id

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate genaiSpatialplan

# Run the combine job for the planetscope datasets
python3 -u combine_planetscope_xarrays.py --region "$region" --landsat_zarr_name "$landsat_zarr_name" --planet_zarr_name "$planet_zarr_name" --filenames="$filenames"


#check if for all files of the region the jobs have finished (the files exist) and if so, submit the combine job
# Extract filenames for the region
landsat_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".landsat_zarr_name")
osm_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".osm_zarr_name")
planet_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".planet_zarr_name")
processed_zarr_name=$(echo "$region_filenames_json" | jq -r ".\"$region\".processed_zarr_name")

# Check if the zarr files exist
if [[ -f "$landsat_zarr_name" && -f "$osm_zarr_name" && -f "$planet_zarr_name" ]]; then
    echo "All required files for region $region exist"
else
    echo "Missing files for region $region. Please check the processing steps."
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
        combine_job=$(sbatch --parsable --job-name="combine_region_$region" --export=region="$region",landsat_zarr_name="$landsat_zarr_name",osm_zarr_name="$osm_zarr_name",planet_zarr_name="$planet_zarr_name",region_filenames_json="$region_filenames_json" ./combine_region_datasets.sh)
        echo "Submitted combine job for region $region: $combine_job"
        done
    fi
fi

    


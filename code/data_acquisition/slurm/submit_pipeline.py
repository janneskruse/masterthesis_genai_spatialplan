#!/usr/bin/env python3
# Import necessary libraries
import os
import sys
import json
import yaml

# Define the repository root
p = os.popen('git rev-parse --show-toplevel')
repo_dir = p.read().strip()
p.close()

# Import helper functions
sys.path.append(f"{repo_dir}/code/helpers")
from submit_job import submit_job_with_dependency, check_existing_job
from get_region_filenames import get_region_filenames

with open(f"{repo_dir}/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

def main():
    # Load configuration values directly from config dict
    print("Loading configuration...")
    
    big_data_storage_path = config['big_data_storage_path']
    min_temperature = config['temperature_day_filter']['min']
    max_cloud_cover = config['landsat_query']['max_cloud_coverage']
    start_year = config['temperature_day_filter']['years']['start']
    end_year = config['temperature_day_filter']['years']['end']
    
    # Construct the input filename
    input_filename = f"{big_data_storage_path}/processed/input_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"
    print(f"Input filename: {input_filename}")
    
    # Check if the input file already exists
    if os.path.exists(input_filename):
        print(f"Input file already exists: {input_filename}")
        sys.exit(0)
    else:
        print("Input file does not exist, proceeding with data acquisition.")
    
    # Get region filenames
    region_filenames_json = get_region_filenames(config_path=f"{repo_dir}/config.yml")
    print("Region filenames JSON loaded successfully: ", region_filenames_json)
    
    # Get list of regions from the JSON
    regions = list(region_filenames_json.keys()) if region_filenames_json else []
    
    # Submit jobs for each region
    for region in regions:
        print(f"Processing region: {region}")
        
        # Extract filenames for the region
        region_data = region_filenames_json.get(region, {})
        landsat_zarr_name = region_data.get("landsat_zarr_name")
        osm_zarr_name = region_data.get("osm_zarr_name")
        planet_zarr_name = region_data.get("planet_zarr_name")
        processed_zarr_name = region_data.get("processed_zarr_name")
        
        # If the processed zarr file already exists, skip the region
        if os.path.exists(processed_zarr_name):
            print(f"Processed Zarr file for region {region} already exists: {processed_zarr_name}")
            continue
        
        landsat_job_id = ""
        osm_job_id = ""
        planet_request_job_id = ""
        
        # Submit Landsat job
        if not os.path.exists(landsat_zarr_name):
            print(f"Submitting Landsat job for {region} (file: {landsat_zarr_name})")
            landsat_job_id = submit_job_with_dependency(
                "./landsat_to_xarray.sh",
                region=region,
                landsat_zarr_name=landsat_zarr_name
            )
            print(f"Landsat job ID: {landsat_job_id}")
        else:
            print(f"Landsat file already exists for {region}: {landsat_zarr_name}")
        
        # Submit OSM job
        if not os.path.exists(osm_zarr_name):
            print(f"Submitting OSM job for {region} (file: {osm_zarr_name})")
            osm_job_id = submit_job_with_dependency(
                "./osm_to_xarray.sh",
                region=region,
                osm_zarr_name=osm_zarr_name,
                region_filenames_json=json.dumps(region_filenames_json)
            )
            print(f"OSM job ID: {osm_job_id}")
        else:
            print(f"OSM file already exists for {region}: {osm_zarr_name}")
        
        # Submit PlanetScope job with dependency on Landsat job
        if not os.path.exists(planet_zarr_name):
            print(f"Submitting PlanetScope job for {region} (file: {planet_zarr_name})")
            
            # Only add dependency if Landsat job was actually submitted
            dependency = landsat_job_id if landsat_job_id else None
            planet_request_job_id = submit_job_with_dependency(
                "./request_planetscope.sh",
                dependency_job_id=dependency,
                region=region,
                landsat_zarr_name=landsat_zarr_name,
                planet_zarr_name=planet_zarr_name,
                region_filenames_json=json.dumps(region_filenames_json)
            )
            print(f"PlanetScope job ID: {planet_request_job_id}")
        else:
            print(f"PlanetScope file already exists for {region}: {planet_zarr_name}")
        
        # If all files exist but not the processed zarr, submit the combine job
        if (not landsat_job_id and not osm_job_id and not planet_request_job_id and 
            not os.path.exists(processed_zarr_name)):
            print(f"Submitting combine job for {region} (file: {processed_zarr_name})")
            
            # Check if a combine job is already running for this region
            existing_job = check_existing_job(f"combine_region_{region}")
            
            if existing_job:
                print(f"Combine job already running for region {region} (Job ID: {existing_job}). Skipping submission.")
            else:
                # Submit the combine job for the region using submit_job_with_dependency for consistency
                combine_job = submit_job_with_dependency(
                    "./combine_region_datasets.sh",
                    region=region,
                    landsat_zarr_name=landsat_zarr_name,
                    osm_zarr_name=osm_zarr_name,
                    planet_zarr_name=planet_zarr_name,
                    region_filenames_json=json.dumps(region_filenames_json)
                )
                print(f"Submitted combine job for region {region}: {combine_job}")
        
        print("---")

if __name__ == "__main__":
    main()
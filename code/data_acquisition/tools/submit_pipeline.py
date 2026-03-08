"""
================================================================================
Pipeline orchestrator — submits SLURM jobs for each region with proper
dependency ordering and CSV-based job-status tracking.
================================================================================
"""
# Import necessary libraries
import os
import sys
import json
import yaml

# Local imports
from helpers.load_configs import load_configs
from helpers.submit_job import submit_job_with_dependency, check_existing_job
from helpers.get_region_filenames import get_region_filenames
from helpers.job_tracker import get_job_csv_path, get_latest_job_status, STATUS_COMPLETED, STATUS_IN_PROGRESS

def main():
    # Load configuration values directly from config dict
    print("Loading configuration...")
    
    config = load_configs()
    repo_dir = config.get("repo_dir", ".")
    config = config.get("data_config", {})
    
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
    region_filenames_json = get_region_filenames(config_path=f"{repo_dir}/code/data_acquisition/config.yml")
    print("Region filenames JSON loaded successfully: ", region_filenames_json)
    
    # Get list of regions from the JSON
    regions = list(region_filenames_json.keys()) if region_filenames_json else []
    
    # Submit jobs for each region
    for region in regions:
        print(f"\n{'='*60}")
        print(f"Processing region: {region}")
        print(f"{'='*60}")
        
        # Extract filenames for the region
        region_data = region_filenames_json.get(region, {})
        landsat_zarr_name = region_data.get("landsat_zarr_name")
        osm_zarr_name = region_data.get("osm_zarr_name")
        planet_zarr_name = region_data.get("planet_zarr_name")
        processed_zarr_name = region_data.get("processed_zarr_name")
        
        # ── CSV paths for each script ──
        landsat_csv = get_job_csv_path(big_data_storage_path, region, "landsat_to_xarray")
        osm_csv = get_job_csv_path(big_data_storage_path, region, "osm_to_xarray")
        request_ps_csv = get_job_csv_path(big_data_storage_path, region, "request_planetscope")
        process_ps_csv = get_job_csv_path(big_data_storage_path, region, "process_planetscope")
        combine_csv = get_job_csv_path(big_data_storage_path, region, "combine_datasets")

        # ── Check status via CSV ──
        # Possible values: None (no CSV), "in_progress", "completed", "failed"
        landsat_status = get_latest_job_status(landsat_csv)
        osm_status = get_latest_job_status(osm_csv)
        request_ps_status = get_latest_job_status(request_ps_csv)
        process_ps_status = get_latest_job_status(process_ps_csv)
        combine_status = get_latest_job_status(combine_csv)

        # Helper: a script is "open" (needs submission) when it has never run or failed
        def _needs_submit(status: str | None) -> bool:
            return status is None or status not in (STATUS_COMPLETED, STATUS_IN_PROGRESS)

        print(f"  landsat_to_xarray   : {landsat_status or 'NOT STARTED'}")
        print(f"  osm_to_xarray       : {osm_status or 'NOT STARTED'}")
        print(f"  request_planetscope : {request_ps_status or 'NOT STARTED'}")
        print(f"  process_planetscope : {process_ps_status or 'NOT STARTED'}")
        print(f"  combine_datasets    : {combine_status or 'NOT STARTED'}")

        # If everything is already done for the region, skip
        if combine_status == STATUS_COMPLETED:
            print(f"All steps completed for region {region}, skipping.")
            continue
        
        landsat_job_id = ""
        osm_job_id = ""
        
        # Submit Landsat job (no dependency)
        if _needs_submit(landsat_status):
            print(f"Submitting Landsat job for {region}")
            landsat_job_id = submit_job_with_dependency(
                "./landsat_to_xarray.sh",
                region=region,
                landsat_zarr_name=landsat_zarr_name
            )
            print(f"  Landsat job ID: {landsat_job_id}")
        else:
            print(f"  Landsat {landsat_status} for {region}, skipping.")
        
        # Submit OSM job (no dependency)
        if _needs_submit(osm_status):
            print(f"Submitting OSM job for {region}")
            osm_job_id = submit_job_with_dependency(
                "./osm_to_xarray.sh",
                region=region,
                osm_zarr_name=osm_zarr_name,
            )
            print(f"  OSM job ID: {osm_job_id}")
        else:
            print(f"  OSM {osm_status} for {region}, skipping.")
        
        # Submit PlanetScope request (depends on Landsat)
        #    request_planetscope will itself submit process_planetscope on success
        if _needs_submit(request_ps_status):
            print(f"Submitting PlanetScope request job for {region}")
            dependency = landsat_job_id if landsat_job_id else None
            planet_request_job_id = submit_job_with_dependency(
                "./request_planetscope.sh",
                dependency_job_id=dependency,
                region=region,
                landsat_zarr_name=landsat_zarr_name,
                planet_zarr_name=planet_zarr_name,
                region_filenames_json=json.dumps(region_filenames_json)
            )
            print(f"  PlanetScope request job ID: {planet_request_job_id}")
        else:
            print(f"  PlanetScope request {request_ps_status} for {region}, skipping.")
        
        # Submit Combine job (depends on landsat + osm + process_planetscope) ──
        #    Only if all three upstream steps are completed. If any is still running
        #    or pending, skip — combine will be triggered by osm/process_planetscope
        #    shell scripts or on the next invocation of submit_pipeline.py.
        all_upstream_done = (
            landsat_status == STATUS_COMPLETED
            and osm_status == STATUS_COMPLETED
            and process_ps_status == STATUS_COMPLETED
        )
        if all_upstream_done and _needs_submit(combine_status):
            print(f"Submitting combine job for {region}")
            
            existing_job = check_existing_job("Combine_Datasets")
            if existing_job:
                print(f"  Combine job already running for {region} (Job ID: {existing_job}), skipping.")
            else:
                combine_job_id = submit_job_with_dependency(
                    "./combine_datasets.sh",
                    region=region,
                )
                print(f"  Combine job ID: {combine_job_id}")
        elif combine_status == STATUS_IN_PROGRESS:
            print(f"  Combine already in-progress for {region}, skipping.")
        elif not all_upstream_done:
            print(f"  Combine not ready yet for {region} — waiting for upstream steps to complete.")
        
        print("---")

if __name__ == "__main__":
    main()
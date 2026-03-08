"""
================================================================================
Script combine all datasets (Landsat, Planet, OSM) 
for a given region into a single xarray Dataset and save it as a Zarr file. 
=================================================================================
"""
## Import libraries
# system
import os
import time
import argparse
import traceback

# data manipulation 
import numpy as np
import rasterio as rio # (rio imports needed for rio to work on xarray)
import rioxarray as rxr
import xarray as xr

# local imports
from helpers.load_configs import add_config_arguments, load_configs
from helpers.get_region_filenames import get_region_filenames
from helpers.job_tracker import (
    get_job_csv_path,
    record_job_start,
    record_job_complete,
    record_job_failure,
    is_script_completed,
)
from data_acquisition.cube.combine import combine_region_datasets

SCRIPT_NAME = "combine_datasets"

#### Function to exit on error ######
def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)

def main(args):
    
    try: 
        ####### Get the region to process #######
        region = args.REGION
        
        # ensure sentencase case for region name to match GHSL data
        region = region.title()

        ###### setup config variables #######
        config = load_configs()
        repo_dir = config.get("repo_dir", ".")
        config = config.get("data_config", {})
        region_filenames_json = get_region_filenames(config)
        
        # setup folders
        big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
        processed_region_folder = f"{big_data_storage_path}/processed/{region.lower()}"
        os.makedirs(processed_region_folder, exist_ok=True)

        # setup folders
        filenames = region_filenames_json[region]

        ###### Job tracking setup ######
        job_csv = get_job_csv_path(big_data_storage_path, region, SCRIPT_NAME)
        job_id = os.environ.get("SLURM_JOB_ID", "local")

        # Skip if already completed
        if is_script_completed(job_csv):
            print(f"[{SCRIPT_NAME}] Already completed for region {region}, skipping.")
            exit(0)

        record_job_start(job_csv, job_id, SCRIPT_NAME)

        ###### Combine datasets for the region #######
        combined_ds = combine_region_datasets(
            region=region,
            repo_dir=repo_dir,
            big_data_storage_path=big_data_storage_path,
            filenames=filenames
        )

        record_job_complete(job_csv, job_id)

    except Exception as e:
        print(f"An error occurred: {e}")
        
        # print full stack trace for debugging
        traceback.print_exc()

        record_job_failure(job_csv, job_id, str(e))
        
        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE DDP for Urban Inpainting')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    # parser.add_argument('--TOTAL_CPUS', type=int, default=1, help='Total number of CPUs available for processing (default: 1)')
    
    args = parser.parse_args()
    
    main(args)
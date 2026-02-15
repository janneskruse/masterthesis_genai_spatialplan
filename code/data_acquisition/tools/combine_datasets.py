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
from data_acquisition.cube.combine import combine_region_datasets

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

        ###### Combine datasets for the region #######
        combined_ds = combine_region_datasets(
            region=region,
            repo_dir=repo_dir,
            big_data_storage_path=big_data_storage_path,
            filenames=filenames
        )


    except Exception as e:
        print(f"An error occurred: {e}")
        
        # print full stack trace for debugging
        traceback.print_exc()
        
        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE DDP for Urban Inpainting')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    # parser.add_argument('--TOTAL_CPUS', type=int, default=1, help='Total number of CPUs available for processing (default: 1)')
    
    args = parser.parse_args()
    
    main(args)
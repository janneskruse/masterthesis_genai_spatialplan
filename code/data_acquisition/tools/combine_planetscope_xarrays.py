"""
================================================================================
Script to combine individual PlanetScope date Zarr files into a single
multi-temporal Zarr cube for a region.
================================================================================
"""
# Planetscope images are high resolution (3m) satellite images from Planet Labs
# Planet lab's has a rest api for metadata based search: https://developers.planet.com/docs/apis/data/reference/#tag/Item-Search
# More information on search filters etc. can be found here: https://developers.planet.com/docs/apis/data/searches-filtering/
# From the results, the images then can be downloaded like indicated here:
# https://developers.planet.com/docs/planetschool/downloading-imagery-with-data-api/

##### Import libraries ######
# system
import argparse
import os
import time
import traceback

# data manipulation
import geopandas as gpd
import xarray as xr

# local imports
from helpers.load_configs import add_config_arguments, load_configs


##### Function to exit on error ######
def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)


def main(args):

    ###### setup config variables #######
    config = load_configs()
    repo_dir = config.get("repo_dir", ".")
    config = config.get("data_config", {})

    ####### Get the region to process #######
    region = args.REGION

    # ensure title case for region name to match GHSL data
    region = region.title()

    # setup folders
    big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    planet_region_folder = f"{big_data_storage_path}/planet_scope/{region.lower()}"
    os.makedirs(planet_region_folder, exist_ok=True)

    ##### get the landsat zarr file name ######
    landsat_zarr_name = args.LANDSAT_ZARR_NAME

    ##### get the config variables from the landsat zarr name ######
    try:
        landsat_zarr_name_noext = os.path.splitext(os.path.basename(landsat_zarr_name.split("/").pop()))[0]
        parts = landsat_zarr_name_noext.split("_")
        min_temperature = int([x for x in parts if x.startswith("ge")][0].replace("ge", ""))
        max_cloud_cover = int([x for x in parts if x.startswith("cc")][0].replace("cc", ""))
        years = [x for x in parts if x.isdigit() and len(x) == 4]
        start_year = years[0]
        end_year = years[1]
        
        if not min_temperature or not max_cloud_cover or not start_year or not end_year:
            exit_with_error(f"Landsat Zarr name does not contain all required parts (min_temperature, max_cloud_cover, start_year, end_year), finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print("Error parsing landsat zarr name:", e)
        exit_with_error(f"Could not parse Landsat Zarr name, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    planet_zarr_name = f"{planet_region_folder}/planet_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

    ######## Try except Planet data processing ########
    try:
        if os.path.exists(planet_zarr_name):
            print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
            exit(0)
        
        filenames = args.FILENAMES

        print(f"Processing PlanetScope data for region {region} using metadata from files: {filenames} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # exit(0) # for testing purposes
        
        folderpath = f"{planet_region_folder}/planet_tmp"
        planet_zarr_filenames = []
        for filename in filenames.split(":"):
            collection = gpd.read_parquet(filename)
            scene_date = collection.date_id.iloc[0]
            scene_date = scene_date.replace("-", "")
            planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"
            planet_zarr_filenames.append(planet_date_zarr_name)

        print(f"Found PlanetScope Zarr files: {planet_zarr_filenames} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        xr_ds_list = [xr.open_zarr(filename) for filename in planet_zarr_filenames if os.path.exists(filename)]

        if not xr_ds_list:
            exit_with_error(f"No valid xarray datasets found in the provided filenames, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # concat along time dimension
        print("Concatenating xarray datasets at", time.strftime("%Y-%m-%d %H:%M:%S"))
        xds = xr.concat(xr_ds_list, dim="time")
        
        # rechunk the data to avoid memory issues
        print("Rechunking data at", time.strftime("%Y-%m-%d %H:%M:%S"))
        xds = xds.chunk({'time': 1, 'y': 1024, 'x': 1024})
        
        # write to zarr
        print("Writing to zarr at", time.strftime("%Y-%m-%d %H:%M:%S"))
        xds.to_zarr(planet_zarr_name, mode='w', consolidated=True)
        print(f"PlanetScope data written to {planet_zarr_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"An error occurred: {e}")

        # print full stack trace for debugging
        traceback.print_exc()

        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine individual PlanetScope date Zarr files into a single multi-temporal Zarr cube')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    parser.add_argument('--LANDSAT_ZARR_NAME', type=str, required=True, help='Path to the Landsat zarr file (used to derive config parameters)')
    parser.add_argument('--FILENAMES', type=str, required=True, help='Colon-separated list of parquet file paths for each date collection')
    
    args = parser.parse_args()
    
    main(args)
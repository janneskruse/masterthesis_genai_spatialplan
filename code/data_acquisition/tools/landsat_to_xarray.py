"""
================================================================================
Script to acquire and pre-process Landsat LST data for high temperature days.
Orchestrates the DWD temperature filtering and Landsat data acquisition pipeline,
saved as a Zarr file.
================================================================================
"""

##### Import libraries ######
# system
import argparse
import os
import time
import traceback

# data manipulation
import json
import geopandas as gpd
from dotenv import load_dotenv

# local imports
from helpers.landsat_config import get_landsat_config_vars
from helpers.load_configs import add_config_arguments, load_configs
from data_acquisition.cube.metropolitan_regions import get_region_bbox
from data_acquisition.lst.dwd import get_high_temperature_dates
from data_acquisition.lst.landsat import (
    query_stac_for_dates,
    get_landsat_temperature_products,
    download_all_products,
    build_landsat_zarr,
)


#### Function to exit on error ######
def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)


def main(args):

    ###### setup config variables #######
    config = load_configs()
    repo_dir = config.get("repo_dir", ".")

    # Load .env file
    load_dotenv(dotenv_path=f"{repo_dir}/.env")

    ####### Get the region to process #######
    region = args.REGION

    # ensure title case for region name to match GHSL data
    region = region.title()

    print(f"Processing region: {region} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # exit(0)  # Exit early for testing purposes

    ######## Try except Landsat data processing ########
    try:
        ############ Define the bbox ############ 
        bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)
        bbox_polygon = json.loads(bbox_gdf.to_json())['features'][0]['geometry']

        ####### Get/Define the config parameters ########
        config_vars = get_landsat_config_vars(os.path.join(repo_dir, "code/data_acquisition/config.yml"), region)
        
        landsat_region_folder = config_vars["landsat_region_folder"]
        landsat_zarr_name = config_vars["landsat_zarr_name"]
        stac_filename = config_vars["stac_filename"]

        min_temperature = config_vars["min_temperature"]
        max_cloud_cover = config_vars["max_cloud_cover"]
        start_year = config_vars["start_year"]
        end_year = config_vars["end_year"]
        consecutive_days = config_vars["consecutive_days"]
        collections = config_vars["collections"]
        max_dates_per_year = config_vars["max_dates_per_year"]

        print("Processing region:", region, "at", time.strftime("%Y-%m-%d %H:%M:%S"), "to produce zarr file:", landsat_zarr_name)
        
        if not os.path.exists(landsat_zarr_name):
            print(f"Creating Landsat zarr dataset at {landsat_zarr_name} at", time.strftime("%Y-%m-%d %H:%M:%S"))
        
            if not os.path.exists(stac_filename):
                ########### Get consecutive high temperatures from DWD #########
                station_temp_max_gt_dates = get_high_temperature_dates(
                    region=region,
                    repo_dir=repo_dir,
                    bbox_gdf=bbox_gdf,
                    start_year=start_year,
                    end_year=end_year,
                    min_temperature=min_temperature,
                    consecutive_days=consecutive_days
                )

                ################## Get LST Data from Landsat #################
                query_gdf = query_stac_for_dates(
                    dates=station_temp_max_gt_dates,
                    bbox_polygon=bbox_polygon,
                    collections=collections,
                    max_cloud_cover=max_cloud_cover
                )

                # save as geoparquet
                print(f"Saving STAC query results to {stac_filename}")
                query_gdf.to_parquet(stac_filename)
                print(f"Saved STAC query results to {stac_filename}")
            else:
                print(f"STAC query results already exist at {stac_filename}")

            query_gdf = gpd.read_parquet(stac_filename)

            ####### Get the images for the requested collection information #######
            products = get_landsat_temperature_products(query_gdf)

            ### Download the images using AWS CLI
            download_all_products(products, landsat_region_folder)

            ######### Create a pre-processed/cleaned zarr for the Landsat data #########
            build_landsat_zarr(
                landsat_region_folder=landsat_region_folder,
                query_gdf=query_gdf,
                bbox_gdf=bbox_gdf,
                max_cloud_cover=max_cloud_cover,
                max_dates_per_year=max_dates_per_year,
                landsat_zarr_name=landsat_zarr_name
            )

            print(f"Saved Landsat dataset to {landsat_zarr_name} at", time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print(f"Landsat zarr dataset already exists at {landsat_zarr_name}, skipping creation at", time.strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        print(f"An error occurred: {e}")
        
        # print full stack trace for debugging
        traceback.print_exc()
        
        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Acquire and pre-process Landsat LST data for high temperature days')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    
    args = parser.parse_args()
    
    main(args)


"""
================================================================================
Script to acquire and pre-process Planet Lab's (PlanetScope) data to an Xarray 
cube for a single date, saved as a Zarr file.
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
import json
import geopandas as gpd
import utm
from pyproj import CRS

# local imports
from helpers.load_configs import add_config_arguments, load_configs
from data_acquisition.cube.metropolitan_regions import get_region_bbox
from data_acquisition.planet_scope.process import (
    create_reference_da_from_bounds,
    build_planetscope_date_zarr,
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

    print("Processing region:", region, "at", time.strftime("%Y-%m-%d %H:%M:%S"))

    ######## Try except Planet data processing ########
    try:
        
        if os.path.exists(planet_zarr_name):
            print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
            exit(0)
        
        filename = args.FILENAME

        print("Processing file:", filename, "at", time.strftime("%Y-%m-%d %H:%M:%S"))
        # exit(0)  # Exit early for testing purposes

        folderpath = f"{planet_region_folder}/planet_tmp"
        collection = gpd.read_parquet(filename)
        scene_date = collection.date_id.iloc[0]
        scene_date = scene_date.replace("-", "")
        collection_folder = f"{folderpath}/psscene_{scene_date}"
        collection_files = os.listdir(collection_folder)
        collection_files = [f"{collection_folder}/{file}" for file in collection_files]
            
        planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"
        
        if os.path.exists(planet_date_zarr_name):
            print(f"PlanetScope data for date {scene_date} already exists at {planet_date_zarr_name}, skipping processing.")
            exit(0)
            
        ############ Define the bbox ############ 
        bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)

        # reproject gdfs to utm zone
        easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
        is_south = zone_letter < 'N'  # True for southern hemisphere
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
        print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")
        bbox_gdf = bbox_gdf.to_crs(utm_crs)

        ###### Prepare reference dataset ##########
        utm_bounds_gdf = bbox_gdf.to_crs(utm_crs)
        bounds = utm_bounds_gdf.total_bounds  # minx, miny, maxx, maxy
        res_m = 3.0
        ref = create_reference_da_from_bounds(bounds, res_m, crs=utm_crs.to_string())
        # ref = ref.rio.reproject("EPSG:4326")

        ######### Build the zarr for this date #########
        build_planetscope_date_zarr(
            collection_files=collection_files,
            bbox_gdf=bbox_gdf,
            ref=ref,
            scene_date=scene_date,
            output_path=planet_date_zarr_name
        )

        print(f"Saved PlanetScope dataset to {planet_date_zarr_name} at", time.strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        print(f"An error occurred: {e}")
        
        # print full stack trace for debugging
        traceback.print_exc()
        
        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PlanetScope data for a single date into a Zarr cube')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    parser.add_argument('--LANDSAT_ZARR_NAME', type=str, required=True, help='Path to the Landsat zarr file (used to derive config parameters)')
    parser.add_argument('--FILENAME', type=str, required=True, help='Path to the collection parquet file for the date to process')
    
    args = parser.parse_args()
    
    main(args)
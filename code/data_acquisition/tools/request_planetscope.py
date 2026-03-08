"""
================================================================================
Script to request and download PlanetScope satellite imagery from the Planet API
for a given region, using Landsat time ranges as reference dates.
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
from dotenv import load_dotenv

# data manipulation
import json
import pandas as pd
import geopandas as gpd
import xarray as xr
import utm
from pyproj import CRS

# local imports
from helpers.load_configs import add_config_arguments, load_configs
from helpers.job_tracker import (
    get_job_csv_path,
    record_job_start,
    record_job_complete,
    record_job_failure,
    is_script_completed,
)
from data_acquisition.cube.metropolitan_regions import get_region_bbox
from data_acquisition.planetscope.request import (
    search_planet_scenes_for_dates,
    get_planetscope_scenes_cover_for_date,
    download_all_collections,
)
from helpers.submit_job import submit_job_with_dependency

SCRIPT_NAME = "request_planetscope"


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

    # Load .env file
    load_dotenv(dotenv_path=f"{repo_dir}/.env")

    # planet lab
    base_url = "https://api.planet.com/data/v1"
    planet_api_key = (os.getenv("PLANET_API_KEY"), "")
    request_path = "/quick-search"
    url = f"{base_url}{request_path}"

    ####### Get the region to process #######
    region = args.REGION

    # ensure title case for region name to match GHSL data
    region = region.title()

    # setup folders
    big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    planet_region_folder = f"{big_data_storage_path}/planet_scope/{region.lower()}"
    os.makedirs(planet_region_folder, exist_ok=True)

    ###### Job tracking setup ######
    job_csv = get_job_csv_path(big_data_storage_path, region, SCRIPT_NAME)
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Skip if already completed
    if is_script_completed(job_csv):
        print(f"[{SCRIPT_NAME}] Already completed for region {region}, skipping.")
        exit(0)

    record_job_start(job_csv, job_id, SCRIPT_NAME)

    # get planet config settings
    planet_query_config = config.get("planetscope_query", {})
    cloud_cover_limit = planet_query_config.get("max_cloud_coverage", 0.1)  # max 10% cloud cover
    asset_names = planet_query_config.get("asset_names", ["ortho_analytic_4b_sr", "ortho_udm2"])
    max_scenes_per_region = planet_query_config.get("max_scenes_per_region", 1)

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

    region_filenames_json = args.REGION_FILENAMES_JSON

    planet_zarr_name = f"{planet_region_folder}/planet_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

    print(f"Requesting PlanetScope with Landsat Zarr file: {landsat_zarr_name} for region: {region} at {time.strftime('%Y-%m-%d %H:%M:%S')} to store at {planet_zarr_name}")

    # comment this out for testing:
    # test_folderpath=f"{planet_region_folder}/planet_tmp"
    # test_filenames=[f"{test_folderpath}/planet_scope_cover_{i.replace('-','')}.parquet" for i in ["2023-01-01", "2023-02-01", "2023-03-01"]]
    # submit_job_with_dependency("./process_planetscope.sh", region=region, landsat_zarr_name=landsat_zarr_name, filenames=test_filenames, region_filenames_json=region_filenames_json)
    # exit(0)  # Exit early for testing purposes

    ######## Planet data processing ########
    try:
        
        if os.path.exists(planet_zarr_name):
            print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
            exit(0)
        
        ############ Define the bbox ############
        bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)
        coordinates = json.loads(bbox_gdf.geometry.to_json())["features"][0]["geometry"]["coordinates"]

        ########## get unique dates from landsat zarr file ##########
        # import landsat xarray dataset
        landsat_xr_ds = xr.open_zarr(landsat_zarr_name, consolidated=True)

        # Get time values from xarray dataset
        time_ranges = landsat_xr_ds.time
        time_ranges = [pd.to_datetime(timestamp).strftime("%Y-%m-%d") for timestamp in time_ranges.values]
        
        # Remove duplicates and sort
        time_ranges = sorted(set(time_ranges))
        time_ids = [(i, t) for i, t in enumerate(time_ranges)]

        ###### get planet scenes for the bbox and time ranges ######
        planet_bydate_gdf = search_planet_scenes_for_dates(
            time_ranges=time_ranges,
            coordinates=coordinates,
            planet_api_key=planet_api_key,
            url=url,
            cloud_cover_limit=cloud_cover_limit
        )

        # reproject gdfs to utm zone
        easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
        is_south = zone_letter < 'N'  # True for southern hemisphere
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
        print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")

        planet_bydate_gdf = planet_bydate_gdf.to_crs(utm_crs)
        bbox_gdf = bbox_gdf.to_crs(utm_crs)

        # Save scene metadata as geoparquet
        meta_filename = f"{planet_region_folder}/planet_ge{min_temperature}_{start_year}_{end_year}_meta.parquet"
        planet_bydate_gdf.to_parquet(meta_filename)
        
        ########## compute scene covers for all dates ##########
        folderpath = f"{planet_region_folder}/planet_tmp"
        os.makedirs(folderpath, exist_ok=True)

        filenames = []
        planet_scope_cover_df_list = []

        for time_id, date_value in time_ids:
            filename = f"{folderpath}/planet_scope_cover_{date_value.replace('-','')}.parquet"
            filenames.append(filename)

            if os.path.exists(filename):
                print(f"Loading existing cover file: {filename}")
                try:
                    df = gpd.read_parquet(filename)
                except Exception as e:
                    print(f"Failed to read {filename} ({e}), recomputing...")
                    df = get_planetscope_scenes_cover_for_date(time_id, landsat_xr_ds, planet_bydate_gdf, bbox_gdf)
                    df = df.to_crs("EPSG:4326")
                    df.to_parquet(filename)
                    print(f"Saved: {filename}")
            else:
                print(f"No existing file for {date_value}, computing cover...")
                df = get_planetscope_scenes_cover_for_date(time_id, landsat_xr_ds, planet_bydate_gdf, bbox_gdf)
                df = df.to_crs("EPSG:4326")
                df.to_parquet(filename)
                print(f"Saved: {filename}")

            planet_scope_cover_df_list.append((date_value, df))

        ######### Download all scenes #########
        download_all_collections(
            filenames=filenames,
            folderpath=folderpath,
            planet_api_key=planet_api_key,
            asset_names=asset_names,
            max_scenes_per_region=max_scenes_per_region
        )
            
        print("Finished processing PlanetScope data at", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Submitting job to process PlanetScope data with dependency on downloaded files at", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Filenames for processing:", filenames)

        record_job_complete(job_csv, job_id)

        submit_job_with_dependency("./process_planetscope.sh", region=region, landsat_zarr_name=landsat_zarr_name, filenames=filenames, region_filenames_json=region_filenames_json)
        exit(0)

    except Exception as e:
        print(f"An error occurred: {e}")

        # print full stack trace for debugging
        traceback.print_exc()

        record_job_failure(job_csv, job_id, str(e))

        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request and download PlanetScope imagery for a region')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    parser.add_argument('--LANDSAT_ZARR_NAME', type=str, required=True, help='Path to the Landsat zarr file (used to derive config parameters and time ranges)')
    parser.add_argument('--REGION_FILENAMES_JSON', type=str, required=True, help='Path to the region filenames JSON file')
    
    args = parser.parse_args()
    
    main(args)
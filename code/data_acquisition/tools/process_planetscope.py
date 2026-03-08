"""
================================================================================
Script to process all PlanetScope date collections into per-date Zarr files
(using multiprocessing) and then combine them into a single multi-temporal
Zarr cube for a region.
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
from multiprocessing import Pool, cpu_count
from functools import partial

# data manipulation
import utm

# visualization
from tqdm.auto import tqdm
from pyproj import CRS

# local imports
from helpers.load_configs import add_config_arguments, load_configs
from data_acquisition.cube.metropolitan_regions import get_region_bbox
from data_acquisition.planet_scope.process import (
    create_reference_da_from_bounds,
    process_single_date,
)
from data_acquisition.planet_scope.combine import combine_planetscope_zarrs


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

    print(f"Processing PlanetScope for region: {region} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ######## Try except Planet data processing ########
    try:
        
        if os.path.exists(planet_zarr_name):
            print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
            exit(0)

        # parse filenames (colon-separated from the request script)
        filenames = args.FILENAMES.split(":")
        filenames = [f.strip() for f in filenames if f.strip()]

        if not filenames:
            exit_with_error(f"No filenames provided, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Processing {len(filenames)} date collections into per-date Zarr files at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        ############ Compute bbox and reference grid once for the region ############
        bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)

        # reproject to utm zone
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            bbox_gdf.geometry.centroid.y.values[0],
            bbox_gdf.geometry.centroid.x.values[0]
        )
        is_south = zone_letter < 'N'  # True for southern hemisphere
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
        print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")
        bbox_gdf = bbox_gdf.to_crs(utm_crs)

        # prepare reference dataset
        bounds = bbox_gdf.total_bounds  # minx, miny, maxx, maxy
        res_m = 3.0
        ref = create_reference_da_from_bounds(bounds, res_m, crs=utm_crs.to_string())

        ########## Step 1: Process each date in parallel ##########
        n_workers = min(len(filenames), max(1, cpu_count() - 1))
        print(f"Using {n_workers} parallel workers for date processing")

        # use partial to fix the shared arguments
        process_fn = partial(
            process_single_date,
            bbox_gdf=bbox_gdf,
            ref=ref,
            planet_region_folder=planet_region_folder
        )

        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, filenames),
                total=len(filenames),
                desc="Processing dates",
                unit="date"
            ))

        # check results
        successful = [r for r in results if r is not None]
        failed = [f for f, r in zip(filenames, results) if r is None]

        print(f"Successfully processed {len(successful)}/{len(filenames)} dates")
        if failed:
            print(f"Failed dates: {failed}")

        if not successful:
            exit_with_error(f"All date processing failed, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        ########## Step 2: Combine all date zarrs into one ##########
        print(f"Combining {len(successful)} date zarrs into {planet_zarr_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        combine_planetscope_zarrs(
            filenames=filenames,
            planet_region_folder=planet_region_folder,
            planet_zarr_name=planet_zarr_name
        )

        print(f"Finished processing PlanetScope data at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"An error occurred: {e}")

        # print full stack trace for debugging
        traceback.print_exc()

        exit_with_error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process and combine PlanetScope data for a region (multiprocessing per-date, then combine)'
    )
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    parser.add_argument('--LANDSAT_ZARR_NAME', type=str, required=True, help='Path to the Landsat zarr file (used to derive config parameters)')
    parser.add_argument('--FILENAMES', type=str, required=True, help='Colon-separated list of collection parquet file paths for each date')
    
    args = parser.parse_args()
    
    main(args)

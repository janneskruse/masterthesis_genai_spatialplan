## Script to aquire and pre-process Planet Lab's (Planetscope) data to an Xarray cube
# Planetscope images are high resolution (3m) satellite images from Planet Labs
# Planet lab's has a rest api for metadata based search: https://developers.planet.com/docs/apis/data/reference/#tag/Item-Search
# More information on search filters etc. can be found here: https://developers.planet.com/docs/apis/data/searches-filtering/
# From the results, the images then can be downloaded like indicated here:
# https://developers.planet.com/docs/planetschool/downloading-imagery-with-data-api/

## Import libraries
# system
import os
import time
import calendar
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib
from dotenv import load_dotenv

# data manipulation
import json
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio # needed for xarray.rio to work
import xarray as xr
import rioxarray as rxr
from skimage.exposure import match_histograms
from rioxarray.merge import merge_arrays
from shapely.geometry import box, shape

# visualization
from tqdm import tqdm

##### Function to exit on error ######
def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)

###### setup config variables #######
repo_name = 'masterthesis_genai_spatialplan'
if not repo_name in os.getcwd():
    os.chdir(repo_name)

p=os.popen('git rev-parse --show-toplevel')
repo_dir = p.read().strip()
p.close()

with open(f"{repo_dir}/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Load .env file
load_dotenv(dotenv_path=f"{repo_dir}/.env")

# planet lab
base_url="https://api.planet.com/data/v1"
planet_api_key=(os.getenv("PLANET_API_KEY"), "")
request_path="/quick-search"
url=f"{base_url}{request_path}"


####### Get the region to process #######
try:
    if "REGION" in os.environ:
        region = os.environ["REGION"] 
    else:
        exit_with_error("Region not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting region from environment:", e)
    exit_with_error(f"Region not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# setup folders
big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
planet_region_folder = f"{big_data_storage_path}/planet_scope/{region.lower()}"
os.makedirs(planet_region_folder, exist_ok=True)

##### get the landsat zarr file name ######
try:
    if "LANDSAT_ZARR_NAME" in os.environ:
        landsat_zarr_name = os.environ["LANDSAT_ZARR_NAME"]
    else:
        exit_with_error(f"Landsat Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting landsat zarr name from environment:", e)
    exit_with_error(f"Landsat Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
    exit_with_error(f"Landsat Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

planet_zarr_name = f"{planet_region_folder}/planet_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

######## Try except Planet data processing ########
try:
    if os.path.exists(planet_zarr_name):
        print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
        exit(0)
    
    if "FILENAMES" in os.environ:
        filenames = os.environ["FILENAMES"]
    else:
        exit_with_error(f"Filenames not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Processing PlanetScope data for region {region} using metadata from files: {filenames} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # exit(0) # for testing purposes
    
    folderpath=f"{planet_region_folder}/planet_tmp"
    planet_zarr_filenames=[]
    for filename in filenames.split(","):
        collection=gpd.read_parquet(filename)
        scene_date=collection.date_id.iloc[0]
        scene_date=scene_date.replace("-","")
        planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"
        planet_zarr_filenames.append(planet_date_zarr_name)

    xr_ds_list = [xr.open_zarr(filename) for filename in planet_zarr_filenames if os.path.exists(filename)]

    if not xr_ds_list:
        exit_with_error(f"No valid xarray datasets found in the provided filenames, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    #concat along time dimension
    xds = xr.concat(xr_ds_list, dim="time")
    
    # write to zarr
    xds.to_zarr(planet_zarr_name, mode='w', consolidated=True)
    print(f"PlanetScope data written to {planet_zarr_name}")

except Exception as e:
    print(f"An error occurred: {e}")
    exit_with_error(f"An error occurred: {e}")
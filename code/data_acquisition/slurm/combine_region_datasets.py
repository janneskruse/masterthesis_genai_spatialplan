## Combine all datasets of the region to a single zarr file

## Import libraries
# system
import os
import sys
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

# import helper functions
# sys.path.append(f"{repo_dir}/code/helpers")

with open(f"{repo_dir}/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

####### Get the region to process #######
try:
    if "REGION" in os.environ:
        region = os.environ["REGION"] 
    else:
        exit_with_error("Region not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting region from environment:", e)
    exit_with_error("Region not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))

# setup folders
big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
processed_region_folder = f"{big_data_storage_path}/processed/{region.lower()}"
os.makedirs(processed_region_folder, exist_ok=True)

##### get the landsat zarr file name ######
try:
    if "LANDSAT_ZARR_NAME" in os.environ:
        landsat_zarr_name = os.environ["LANDSAT_ZARR_NAME"]
    else:
        exit_with_error("Landsat Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting landsat zarr name from environment:", e)
    exit_with_error("Landsat Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))

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
        exit_with_error("Landsat Zarr name does not contain all required parts (min_temperature, max_cloud_cover, start_year, end_year), finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error parsing landsat zarr name:", e)
    exit_with_error("Landsat Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))


####### get the planet zarr file name ######
try:
    if "PLANET_ZARR_NAME" in os.environ:
        planet_zarr_name = os.environ["PLANET_ZARR_NAME"]
    else:
        exit_with_error("Planet Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting planet zarr name from environment:", e)
    exit_with_error("Planet Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
    
####### get the OSM zarr file name ######
try:
    if "OSM_ZARR_NAME" in os.environ:
        osm_zarr_name = os.environ["OSM_ZARR_NAME"]
    else:
        exit_with_error("OSM Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting OSM zarr name from environment:", e)
    exit_with_error("OSM Zarr name not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
    
processed_zarr_name = f"{processed_region_folder}/input_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

print(f"Combining datasets for region: {region} at", time.strftime("%Y-%m-%d %H:%M:%S"), "to store at", processed_zarr_name)
exit(0)  # Exit early for testing purposes

if os.path.exists(processed_zarr_name):
    print(f"Processed data already exists at {processed_zarr_name}, skipping processing.")
    exit(0)

####### read the zarr files #######
xr_landsat = xr.open_zarr(landsat_zarr_name)
xr_planet = xr.open_zarr(planet_zarr_name)
xr_osm = xr.open_zarr(osm_zarr_name)
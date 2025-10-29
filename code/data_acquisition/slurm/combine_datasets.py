## Combine all datasets of all region to a single zarr file

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

with open(f"{repo_dir}/code/data_acquisition/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# setup folders
big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
processed_folder = f"{big_data_storage_path}/processed"
os.makedirs(processed_folder, exist_ok=True)

try:
    if "REGION_FILENAMES_JSON" in os.environ:
        region_filenames_json = os.environ["REGION_FILENAMES_JSON"]
    else:
        exit_with_error(f"Region filenames JSON not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting region filenames JSON from environment:", e)
    exit_with_error(f"Region filenames JSON not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
# Parse the JSON string to a Python dictionary
try:
    region_filenames_json = json.loads(region_filenames_json)
except json.JSONDecodeError as e:
    print("Error decoding JSON from environment variable REGION_FILENAMES_JSON:", e)
    exit_with_error(f"Invalid JSON format for region filenames, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# retrieve landsat zarr name from first entry
try:
    landsat_zarr_name = list(region_filenames_json.values())[0]["landsat_zarr_name"]
except KeyError as e:
    print("Error retrieving landsat zarr name from region filenames JSON:", e)
    exit_with_error(f"Landsat Zarr name not found in region filenames JSON, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

processed_zarr_name = f"{processed_folder}/input_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

print("Combining datasets... at", time.strftime("%Y-%m-%d %H:%M:%S"), "to store at", processed_zarr_name)
# exit(0)  # Exit early for testing purposes

if os.path.exists(processed_zarr_name):
    print(f"Processed data already exists at {processed_zarr_name}, skipping processing.")
    exit(0)

####### read the zarr files from all regions #######
print("Reading zarr files from all regions...")

processed_zarr_names = []
for region, filenames in region_filenames_json.items():
    processed_zarr_names.append(filenames.get("processed_zarr_name"))
    
xds_list = []
for zarr_name in processed_zarr_names:
    print("Reading", zarr_name)
    try:
        xr_data = xr.open_zarr(zarr_name, consolidated=True)
        xds_list.append(xr_data)
    except Exception as e:
        print("Error reading", zarr_name, ":", e)
        
        
if not xds_list:
    exit_with_error(f"No valid xarray datasets found in the provided filenames, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
# Concatenate along time dimension
print("Concatenating datasets along time dimension...")
xds = xr.concat(xds_list, dim="time")

# Write to zarr
print("Writing combined dataset to", processed_zarr_name)
xds.to_zarr(processed_zarr_name, mode='w', consolidated=True)

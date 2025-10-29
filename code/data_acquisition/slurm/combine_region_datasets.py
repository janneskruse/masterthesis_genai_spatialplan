## Combine all datasets of the region to a single zarr file

## Import libraries
# system
import os
import time

# data manipulation 
import yaml
import numpy as np
import pandas as pd
import rasterio as rio # (rio imports needed for rio to work on xarray)
from rasterio.enums import Resampling
import rioxarray as rxr
import xarray as xr

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

####### Get the region to process #######
try:
    if "REGION" in os.environ:
        region = os.environ["REGION"] 
    else:
        exit_with_error(f"Region not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting region from environment:", e)
    exit_with_error(f"Region not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# setup folders
big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
processed_region_folder = f"{big_data_storage_path}/processed/{region.lower()}"
os.makedirs(processed_region_folder, exist_ok=True)

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


####### get the planet zarr file name ######
try:
    if "PLANET_ZARR_NAME" in os.environ:
        planet_zarr_name = os.environ["PLANET_ZARR_NAME"]
    else:
        exit_with_error(f"Planet Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting planet zarr name from environment:", e)
    exit_with_error(f"Planet Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
####### get the OSM zarr file name ######
try:
    if "OSM_ZARR_NAME" in os.environ:
        osm_zarr_name = os.environ["OSM_ZARR_NAME"]
    else:
        exit_with_error(f"OSM Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting OSM zarr name from environment:", e)
    exit_with_error(f"OSM Zarr name not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

processed_zarr_name = f"{processed_region_folder}/input_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

print(f"Combining datasets for region: {region} at {time.strftime('%Y-%m-%d %H:%M:%S')} to store at {processed_zarr_name}")
# exit(0)  # Exit early for testing purposes

if os.path.exists(processed_zarr_name):
    print(f"Processed data already exists at {processed_zarr_name}, skipping processing.")
    exit(0)
    

# helper: reproject/resample every variable in a Dataset
def reproject_ds(ds: xr.Dataset, template_da: xr.DataArray, resampling: Resampling) -> xr.Dataset:
    out_vars = {}
    for name, da in ds.data_vars.items():
        # carry nodata forward
        if da.rio.nodata is None and getattr(da, "_FillValue", None) is not None:
            da = da.rio.write_nodata(da._FillValue, encoded=True)
        out_vars[name] = da.rio.reproject_match(template_da, resampling=resampling)
    out = xr.Dataset(out_vars)
    # copy non-spatial coords
    for c in ds.coords:
        if c not in ("x", "y"):
            out = out.assign_coords({c: ds[c]})
    out.attrs = ds.attrs
    return out

####### read the zarr files #######
xr_planet = xr.open_zarr(planet_zarr_name, consolidated=True)
xr_planet = xr_planet.set_coords("spatial_ref")
xr_osm = xr.open_zarr(osm_zarr_name, consolidated=True)

drop_vars = ['qa_pixel',
 'stac_id',
 'view_sun_azimuth',
 'surface_temp_b10',
 'view_off_nadir',
 'view_sun_elevation']

xr_landsat = xr.open_zarr(landsat_zarr_name, consolidated=True, decode_times=False, drop_variables=drop_vars)

print("Landsat bounds:", xr_landsat.rio.bounds())
print("Planet bounds:", xr_planet.rio.bounds())
print("OSM bounds:", xr_osm.rio.bounds())


# rename surface_temp_b10 to land_surface_temp
lst_name = "landsat_surface_temp_b10_masked"
xr_landsat = xr_landsat.rename({"surface_temp_b10_masked": lst_name})

# convert time to minute precision
xr_landsat['time'] = xr_landsat['time'].astype('datetime64[m]')
xr_planet['time'] = xr_planet['time'].astype('datetime64[m]')

# find nearest landsat time for each planet time and modify planet time to match
planet_times = xr_planet['time'].values
landsat_times = xr_landsat['time'].values
nearest_landsat_times = np.array([landsat_times[np.abs(landsat_times - pt).argmin()] for pt in planet_times])
xr_planet = xr_planet.assign_coords(time=nearest_landsat_times)

#add spatial_ref coordsinate to landsat and planet
print("CRS of OSM dataset:", xr_osm.rio.crs)
xr_landsat = xr_landsat.rio.write_crs(xr_osm.rio.crs, inplace=True)
xr_landsat = xr_landsat.rio.write_coordinate_system(inplace=True)

# rechunk to common chunk size
common_chunks = {'x': xr_planet.chunks['x'][0], 'y': xr_planet.chunks['y'][0]}
xr_landsat = xr_landsat.chunk(common_chunks)
xr_osm = xr_osm.chunk(common_chunks)
xr_planet = xr_planet.chunk(common_chunks)

template = (
    xr_planet.isel(time=0)
    .rio.write_crs(xr_planet.rio.crs, inplace=False)
)

# ensure spatial dims are x/y for all three
for ds in [xr_planet, xr_landsat, xr_osm]:
    if {"x","y"} - set(ds.dims):
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

## resample/reproject datasets to common grid
print("Reprojecting and resampling datasets to common grid...")
print(f"Target CRS: {template.rio.crs}")
#resample/reproject osm
osm_on_planet = reproject_ds(xr_osm, template, Resampling.nearest)

# resample/reproject landsat lst
if xr_landsat[lst_name].rio.nodata is None:
    xr_landsat[lst_name] = xr_landsat[lst_name].rio.write_nodata(-9999, encoded=True)

landsat_lst_on_planet = xr_landsat[lst_name].rio.reproject_match(
    template, resampling=Resampling.bilinear
)

# convert no data values back to NaN
landsat_lst_on_planet = landsat_lst_on_planet.where(landsat_lst_on_planet != -9999, np.nan)

# rechunk to align with xr_planet
target_chunks = {"y": xr_planet.chunks["y"][0], "x": xr_planet.chunks["x"][0]}
osm_on_planet = osm_on_planet.chunk(target_chunks)
if "time" in landsat_lst_on_planet.dims:
    target_chunks = {"time": 1, **target_chunks}
landsat_lst_on_planet = landsat_lst_on_planet.chunk(target_chunks)

# merge all three datasets
print("Merging datasets...")
merged_xs = xr.merge(
    [xr_planet, osm_on_planet, landsat_lst_on_planet.to_dataset(name=lst_name)],
    compat="override",
    join="outer",
    fill_value=np.nan
)

# remove chunk encoding to prevent errors
for var in merged_xs.data_vars:
    if 'chunks' in merged_xs[var].encoding:
        del merged_xs[var].encoding['chunks']

# save to zarr
print("Saving processed Zarr file...")
merged_xs.to_zarr(
    processed_zarr_name,
    mode="w",
    consolidated=True,
    compute=True,
)
print(f"Processed Zarr file saved at {processed_zarr_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}.")


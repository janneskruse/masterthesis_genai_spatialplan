"""
================================================================================
Module to combine individual PlanetScope date Zarr files into a single
multi-temporal Zarr cube for a region.
================================================================================
"""

##### Import libraries ######
# system
import os
import time

# data manipulation
import geopandas as gpd
import xarray as xr


def combine_planetscope_zarrs(
    filenames: list[str],
    planet_region_folder: str,
    planet_zarr_name: str
) -> None:
    """
    Combine individual PlanetScope date Zarr files into a single multi-temporal Zarr cube.
    
    Reads collection parquet files to derive per-date zarr paths, opens them,
    concatenates along the time dimension, rechunks for efficiency, and writes
    the combined dataset to a single zarr store.
    
    Args:
        filenames: Colon-separated string or list of parquet file paths for each date collection.
        planet_region_folder: Folder containing the per-date zarr files.
        planet_zarr_name: Output path for the combined zarr store.
        
    Raises:
        ValueError: If no valid xarray datasets are found.
    """
    if os.path.exists(planet_zarr_name):
        print(f"PlanetScope data already exists at {planet_zarr_name}, skipping combine.")
        return

    # support both list and colon-separated string
    if isinstance(filenames, str):
        filenames = filenames.split(":")

    print(f"Combining PlanetScope data from {len(filenames)} date files at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    planet_zarr_filenames = []
    for filename in filenames:
        collection = gpd.read_parquet(filename)
        scene_date = collection.date_id.iloc[0]
        scene_date = scene_date.replace("-", "")
        planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"
        planet_zarr_filenames.append(planet_date_zarr_name)
        collection = None  # free memory

    print(f"Found PlanetScope Zarr files: {planet_zarr_filenames} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    xr_ds_list = []
    for zf in planet_zarr_filenames:
        if os.path.exists(zf):
            xr_ds_list.append(xr.open_zarr(zf))
        else:
            print(f"Warning: Zarr file {zf} does not exist, skipping.")

    if not xr_ds_list:
        raise ValueError("No valid xarray datasets found in the provided filenames")

    try:
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
    finally:
        # close all opened zarr datasets
        for ds in xr_ds_list:
            try:
                ds.close()
            except Exception:
                pass

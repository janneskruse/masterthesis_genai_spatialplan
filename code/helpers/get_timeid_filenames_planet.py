########## get unique dates from landsat zarr file and convert to filenames ##########
import sys
import xarray as xr
import pandas as pd

def get_timeid_filenames(landsat_zarr_name, region, big_data_storage_path="/work/zt75vipu-master/data"):
    
    #import landsat xarray dataset
    landsat_xr_ds=xr.open_zarr(landsat_zarr_name, consolidated=True)

    # Get time values from xarray dataset
    time_ranges = landsat_xr_ds.time
    time_ranges = [pd.to_datetime(timestamp).strftime("%Y-%m-%d") for timestamp in time_ranges.values]

    # Remove duplicates and sort
    time_ranges = sorted(set(time_ranges))
    time_ids=[(i, t) for i, t in enumerate(time_ranges)]

    filenames= []
    planet_region_folder = f"{big_data_storage_path}/planet_scope/{region.lower()}"
    folderpath=f"{planet_region_folder}/planet_tmp"
    for i, t in time_ids:
        filename=f"{folderpath}/planet_scope_cover_{t.replace('-','')}.parquet"
        filenames.append(filename)

    return filenames

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: get_timeid_filenames_planet.py <landsat_zarr_name> <region>")
        sys.exit(1)
    landsat_zarr_name = sys.argv[1]
    region = sys.argv[2]
    filenames = get_timeid_filenames(landsat_zarr_name, region)
    print(filenames)
        
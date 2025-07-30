# to-do: sh file that sets up a list of jobs for all regions specified in the yaml config file
# it should check wether in big_data_storage_path/processed a region folder exists and contains model_input.zarr
# if not, it should submit a job to acquire the landsat data first, then a job to setup the planetscope request with few resources that finishes when the products are ready
# then it should submit a job to acquire the planetscope data once its ready
# also, from the beginning a job to acquire the osm data
# finally when all three are finished, a job to combine the data into a single zarr dataset for the region
# when all regions are done, a job to combine all regions into a single zarr dataset



########## get unique dates from landsat zarr file ##########
#import landsat xarray dataset
landsat_xr_ds=xr.open_zarr(landsat_zarr_name, consolidated=True)

# Get time values from xarray dataset
time_ranges = landsat_xr_ds.time
time_ranges = [pd.to_datetime(timestamp).strftime("%Y-%m-%d") for timestamp in time_ranges.values]

# Remove duplicates and sort
time_ranges = sorted(set(time_ranges))
time_ids=[(i, t) for i, t in enumerate(time_ranges)]

filenames= []
folderpath=f"{planet_region_folder}/planet_tmp"
for i, t in time_ids:
    filename=f"{folderpath}/planet_scope_cover_{t.replace('-','')}.parquet"
    filenames.append(filename)
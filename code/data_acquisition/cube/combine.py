## Import libraries
# system
import os

# data manipulation
import yaml
import numpy as np
import geopandas as gpd
from rasterio.enums import Resampling
import rioxarray as rxr # adds rioxarray capabilities to xarray
import xarray as xr
import utm
from pyproj import CRS

# local imports
from data_acquisition.cube.urban_zones import define_urban_areas
from data_acquisition.cube.reproject import reproject_ds
from data_acquisition.cube.metropolitan_regions import get_region_bbox

# combine function
def combine_region_datasets(region, big_data_storage_path, repo_dir, region_filenames_json) -> xr.Dataset:
    """
    Combine Planet, OSM, and Landsat datasets into a single xarray Dataset.
    Aligns datasets spatially and temporally.
    Saves the combined dataset as a Zarr file.
    
    Args:
        region (str): Region name to process. Defaults to the first region in the config.
        big_data_storage_path (str): Path to the big data storage directory.
        repo_dir (str): Path to the repository directory.
        region_filenames_json (dict): JSON object containing filenames for the region.
    
    Returns:
        xr.Dataset: Combined dataset with Planet, OSM, and Landsat data.
    """
    
    # setup folders
    processed_region_folder = f"{big_data_storage_path}/processed/{region.lower()}"
    os.makedirs(processed_region_folder, exist_ok=True)
    filenames = region_filenames_json[region]
    processed_zarr_name = filenames['processed_zarr_name']
    processed_zarr_name_clipped = processed_zarr_name.replace(".zarr", "_clipped.zarr")

    if os.path.exists(processed_zarr_name) and os.path.exists(processed_zarr_name_clipped):
        print(f"Processed Zarr file already exists at {processed_zarr_name}. Skipping processing.")
        return xr.open_zarr(processed_zarr_name, consolidated=True)
    
    # Get region bounding box
    bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)

    # reproject gdfs to utm zone
    easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
    is_south = zone_letter < 'N'  # True for southern hemisphere
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
    print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")
    bbox_gdf = bbox_gdf.to_crs(utm_crs)

    # define urban areas
    urban_mask = define_urban_areas(region=region, utm_crs=utm_crs, max_distance=20, bbox_gdf=bbox_gdf)

    drop_vars = ['qa_pixel',
    'stac_id',
    'view_sun_azimuth',
    'surface_temp_b10',
    'view_off_nadir',
    'view_sun_elevation']

    planet_xs = xr.open_zarr(filenames['planet_zarr_name'], consolidated=True)
    planet_xs = planet_xs.set_coords("spatial_ref")
    
    osm_xs = xr.open_zarr(filenames['osm_zarr_name'], consolidated=True)
    landsat_xs = xr.open_zarr(filenames['landsat_zarr_name'], consolidated=True, drop_variables=drop_vars)

    # rename surface_temp_b10 to land_surface_temp
    lst_name = "landsat_surface_temp_b10_masked"
    landsat_xs = landsat_xs.rename({"surface_temp_b10_masked": lst_name})

    # convert time to miunute precision
    landsat_xs['time'] = landsat_xs['time'].astype('datetime64[m]')
    planet_xs['time'] = planet_xs['time'].astype('datetime64[m]')

    # find nearest landsat time for each planet time and modify planet time to match
    planet_times = planet_xs['time'].values
    landsat_times = landsat_xs['time'].values
    nearest_landsat_times = np.array([landsat_times[np.abs(landsat_times - pt).argmin()] for pt in planet_times])
    planet_xs = planet_xs.assign_coords(time=nearest_landsat_times) 

    #add spatial_ref coordinate to landsat
    print("CRS of OSM dataset:", osm_xs.rio.crs)
    landsat_xs = landsat_xs.rio.write_crs(osm_xs.rio.crs, inplace=True)
    landsat_xs = landsat_xs.rio.write_coordinate_system(inplace=True)

    # rechunk to common chunk size
    common_chunks = {'x': planet_xs.chunks['x'][0], 'y': planet_xs.chunks['y'][0]}
    landsat_xs = landsat_xs.chunk(common_chunks)
    osm_xs = osm_xs.chunk(common_chunks)
    planet_xs = planet_xs.chunk(common_chunks)
    
    # templates for reprojection
    template = (
        planet_xs.isel(time=0)
        .rio.write_crs(planet_xs.rio.crs, inplace=False)
    )

    # ensure spatial dims are x/y for all three
    datasets = {"planet_xs": planet_xs, "landsat_xs": landsat_xs, "osm_xs": osm_xs}
    for key, ds in datasets.items():
        if {"x", "y"} - set(ds.dims):
            ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        datasets[key] = ds  # keep reference
    planet_xs, landsat_xs, osm_xs = datasets["planet_xs"], datasets["landsat_xs"], datasets["osm_xs"]


    ## resample/reproject datasets to common grid
    print("Reprojecting and resampling datasets to common grid...")
    print(f"Target CRS: {template.rio.crs}")
    #resample/reproject osm
    osm_on_planet = reproject_ds(osm_xs, template, Resampling.nearest)

    # resample/reproject landsat lst
    if landsat_xs[lst_name].rio.nodata is None:
        landsat_xs[lst_name] = landsat_xs[lst_name].rio.write_nodata(-9999, encoded=True)

    landsat_lst_on_planet = landsat_xs[lst_name].rio.reproject_match(
        template, resampling=Resampling.bilinear
    )

    # convert no data values back to NaN
    landsat_lst_on_planet = landsat_lst_on_planet.where(landsat_lst_on_planet != -9999, np.nan)

    # rechunk to align with planet_xs
    target_chunks = {"y": planet_xs.chunks["y"][0], "x": planet_xs.chunks["x"][0]}
    osm_on_planet = osm_on_planet.chunk(target_chunks)
    if "time" in landsat_lst_on_planet.dims:
        target_chunks = {"time": 1, **target_chunks}
    landsat_lst_on_planet = landsat_lst_on_planet.chunk(target_chunks)

    # merge all three datasets
    print("Merging datasets...")
    merged_xs = xr.merge(
        [planet_xs,osm_on_planet, landsat_lst_on_planet.to_dataset(name=lst_name)],
        compat="override",
        join="outer",
        fill_value=np.nan
    )
    
    
    # set crs
    merged_xs = merged_xs.rio.write_crs(planet_xs.rio.crs, inplace=True)
    
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
    print(f"Processed Zarr file saved at {processed_zarr_name}.")
    
    # reproject urban mask
    urban_mask = urban_mask.rio.reproject_match(merged_xs.isel(time=0), resampling=Resampling.nearest)
    
    # clip to urban areas
    print("Clipping merged dataset to urban areas...")
    merged_xs_clipped = merged_xs.where(urban_mask == 1)
    
    
    # per-dimension chunk-size mapping
    clipped_chunk_sizes = {}
    for dim in merged_xs_clipped.dims:
        if dim in merged_xs.chunks and merged_xs.chunks[dim]:
            clipped_chunk_sizes[dim] = merged_xs.chunks[dim][0]  # use size, not full tuple
        else:
            # default sizes
            clipped_chunk_sizes[dim] = 1 if dim == "time" else merged_xs_clipped.sizes[dim]

    if "time" in merged_xs_clipped.dims:
        clipped_chunk_sizes["time"] = 1

    merged_xs_clipped = merged_xs_clipped.chunk(clipped_chunk_sizes)

    
    for var in merged_xs_clipped.data_vars:
        if 'chunks' in merged_xs_clipped[var].encoding:
            del merged_xs_clipped[var].encoding['chunks']

    merged_xs_clipped.to_zarr(
        processed_zarr_name_clipped,
        mode="w",
        consolidated=True,
        compute=True,
    )
    print(f"Clipped processed Zarr file saved at {processed_zarr_name_clipped}.")
    
    # close datasets
    planet_xs.close()
    landsat_xs.close()
    osm_xs.close()
    merged_xs.close()
    merged_xs_clipped.close()
    urban_mask.close()
    
    return
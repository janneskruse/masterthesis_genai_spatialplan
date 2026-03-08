"""
================================================================================
Module to process PlanetScope satellite imagery tiles into xarray datasets.
Provides functions for reading tifs, quality scoring, histogram matching,
reference grid creation, and zarr assembly.
================================================================================
"""

##### Import libraries ######
# data manipulation
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio as rio  # needed for xarray.rio to work
import xarray as xr
import rioxarray as rxr
from skimage.exposure import match_histograms
from rioxarray.merge import merge_arrays


###### Prepare reference dataset ##########
def create_reference_da_from_bounds(
    bounds: tuple,
    res: float,
    crs: str = "EPSG:4326"
) -> xr.DataArray:
    """
    Create an empty DataArray template covering bounds = (minx, miny, maxx, maxy)
    with resolution res (units of CRS) and CRS string.
    """
    minx, miny, maxx, maxy = bounds
    # x from left to right, y from top to bottom (descending)
    xs = np.arange(minx + res / 2, maxx, res)
    ys = np.arange(maxy - res / 2, miny, -res)
    arr = np.zeros((ys.size, xs.size), dtype="int16")
    da = xr.DataArray(arr, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da = da.rio.write_crs(crs)
    return da


def read_planetscope_to_xarray_ds(
    filepath: str,
    bbox_gdf: gpd.GeoDataFrame
) -> xr.Dataset:
    """
    Read PlanetScope tif files to xarray dataset and attach metadata.
    
    Args:
        filepath: str, path to the tif file
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        
    Returns:
        xarray Dataset with planetscope_sr_4band and metadata variables.
    """
    # Open with chunking for memory efficiency
    xda = rxr.open_rasterio(filepath, chunks={'x': 1024, 'y': 1024})
    xda = xda.astype("int16")
    # xda = xda.rio.reproject("EPSG:4326")

    # clip to bbox
    xda = xda.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)

    # rename bands to ['blue', 'green', 'red', 'nir']
    xda = xda.rename({"band": "channel"})
    xda = xda.assign_coords(channel=["blue", "green", "red", "nir"])

    # remove spatial_ref coords
    # xda=xda.drop_vars(["spatial_ref"])

    # add attributes
    xda = xda.assign_attrs(
        scale_factor=0.0001,
        offset=0.0,
        units='reflectance',
        description='Analysis-Ready PlanetScope Surface Reflectance'
    )

    # rename variable
    xda = xda.rename("planetscope_sr_4band")

    # process datetime from attributes
    tiff_datetime = xda.attrs["TIFFTAG_DATETIME"]  # "2019:07:27 08:10:45"
    tiff_datetime = tiff_datetime.replace(":", "-", 2)
    xda = xda.expand_dims(time=[np.datetime64(tiff_datetime)])
    
    # remove unneeded attrs
    xda.attrs.pop("TIFFTAG_DATETIME", None)
    
    
    # add another variable from TIFFTAG_IMAGEDESCRIPTION
    image_description = xda.attrs["TIFFTAG_IMAGEDESCRIPTION"]
    xda.attrs.pop("TIFFTAG_IMAGEDESCRIPTION", None)
    xds = xda.to_dataset()
    xds["meta_planetscope_sr_4band"] = xr.DataArray([image_description], dims=["time"])
    
    # apply scale factor
    xda = xda * xda.scale_factor + xda.offset

    return xds


def extract_quality_score(xds: xr.Dataset) -> float:
    """
    Extract a quality score from PlanetScope metadata for sorting tiles.
    
    Args:
        xds: xarray Dataset with meta_planetscope_sr_4band variable.
        
    Returns:
        Quality score (higher is better).
    """
    try:
        attrs = json.loads(xds["meta_planetscope_sr_4band"].values[0])
        ac = attrs["atmospheric_correction"]
        aot = ac["aot_used"]
        zenith = ac["solar_zenith_angle"]
        return 1 / ((1 + aot) * (1 + zenith))
    except Exception:
        return 0  # fallback


def histogram_match(source: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """
    Performs histogram matching between a source and a reference DataArray,
    basing the matching on the overlapping area between them.
    """
    source_overlap = None
    ref_overlap = None
    
    try:
        # get geometry of both datasets
        source_geom = box(*source.rio.bounds())
        ref_geom = box(*reference.rio.bounds())

        # calculate the intersection
        overlap_geom = source_geom.intersection(ref_geom)

        if not overlap_geom.is_empty:
            # clip source and reference to overlap area
            source_overlap = source.rio.clip([overlap_geom], source.rio.crs, drop=False, invert=False)
            ref_overlap = reference.rio.clip([overlap_geom], source.rio.crs, drop=False, invert=False)
        else:
            print("No geometric overlap found. Falling back to full image histogram matching.")

    except Exception as e:
        print(f"Could not create overlap area: {e}. Falling back to full image histogram matching.")

    # use the full images as fallback
    if source_overlap is None or ref_overlap is None:
        source_overlap = source
        ref_overlap = reference

    matched_bands = []
    for b in range(source.shape[0]):
        src_band = source[b].values
        src_overlap_band = source_overlap[b].values
        ref_overlap_band = ref_overlap[b].values

        # mask out invalid values (nan or 0)
        valid_src_mask = np.isfinite(src_overlap_band) & (src_overlap_band > 0)
        valid_ref_mask = np.isfinite(ref_overlap_band) & (ref_overlap_band > 0)

        matched_band = src_band.copy().astype("float32")
        
        # only match if valid pixels in the overlap
        if np.any(valid_src_mask) and np.any(valid_ref_mask):
            match_ref = ref_overlap_band[valid_ref_mask]

            # full source band values
            full_src_valid_mask = np.isfinite(src_band) & (src_band > 0)
            
            if full_src_valid_mask.any():
                # perform histogram matching
                matched_valid_pixels = match_histograms(
                    src_band[full_src_valid_mask],
                    match_ref,
                )
                matched_band[full_src_valid_mask] = matched_valid_pixels
        
        # set nodata values to nan
        matched_band[~np.isfinite(src_band) | (src_band == 0)] = np.nan
        matched_bands.append(matched_band)
    
    # clean up resources
    source.close()
    source_overlap.close()
    ref_overlap.close()
        
    return xr.DataArray(
        np.stack(matched_bands),
        dims=source.dims,
        coords=source.coords,
        attrs=source.attrs,
    ).rio.write_crs(source.rio.crs)


def build_planetscope_date_zarr(
    collection_files: list[str],
    bbox_gdf: gpd.GeoDataFrame,
    ref: xr.DataArray,
    scene_date: str,
    output_path: str
):
    """
    Build a PlanetScope zarr dataset for a single date from collection tif files.
    
    Reads all tifs, sorts by quality, histogram-matches tiles, merges them,
    resamples to the reference grid, computes NDVI, and saves as zarr.
    
    Args:
        collection_files: List of paths to PlanetScope tif files.
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        ref: Reference DataArray for reprojection matching.
        scene_date: Date string in YYYYMMDD format.
        output_path: Path to save the output zarr dataset.
        
    Raises:
        ValueError: If no valid xarray datasets could be created.
    """
    # read all files
    # xds_list=[readPlanetScopetoXarrayDS(file) for file in collection_files]
    xds_list = []
    for file in collection_files:
        try:
            xds = read_planetscope_to_xarray_ds(file, bbox_gdf)
            if xds is not None:
                xds_list.append(xds)
        except Exception as e:
            print(f"    Failed to read/convert {file}: {e} -- skipping this file")

    if not xds_list:
        raise ValueError(f"No valid xarray datasets for date {scene_date}")

    # Sort by quality
    xds_list.sort(key=extract_quality_score, reverse=True)

    dataarrays = [
        ds["planetscope_sr_4band"].squeeze("time").transpose("channel", "y", "x")
        for ds in xds_list
    ]

    # set crs for all dataarrays
    # for da in dataarrays:
    #     da.rio.write_crs("EPSG:32633", inplace=True)

    # as float for rio merge later
    dataarrays = [
        da.astype("float32") for da in dataarrays
    ]

    reference = dataarrays[0]
    matched_dataarrays = [reference]

    print(f"Merging {len(dataarrays)} tiles...")
    if len(dataarrays) > 1:
        for da in dataarrays[1:]:
            try:
                matched_dataarrays.append(histogram_match(da, reference))
            except Exception as e:
                print(f"    Histogram matching failed for one tile: {e} -- using original tile")
                matched_dataarrays.append(da)

        merged = merge_arrays(
            matched_dataarrays,
            method="first",
            nodata=np.nan,
            res=None,
        )
    else:
        # single tile -> no merge needed
        merged = reference

    # back to int
    # merged = (merged * 1).astype("int16")
    
    # resample to reference dataset
    merged = merged.rio.reproject_match(ref)
    
    # drop nan coords
    merged = merged.dropna("x", how="all").dropna("y", how="all")

    # Add time dimension and rechunk
    scene_date_np = np.datetime64(pd.to_datetime(scene_date))
    merged = merged.expand_dims(time=[scene_date_np])
    merged = merged.rio.write_nodata(np.nan)
    merged = merged.chunk({'y': 1024, 'x': 1024, 'time': 1, 'channel': 4})

    # Derive NDVI dataarray from the planetscope data
    # create dataset from merged
    merged = merged.to_dataset(name="planetscope_sr_4band")

    # create ndvi
    merged["ndvi"] = (
        (merged.planetscope_sr_4band.isel(channel=3) - merged.planetscope_sr_4band.isel(channel=2)) / 
        (merged.planetscope_sr_4band.isel(channel=3) + merged.planetscope_sr_4band.isel(channel=2))
    )

    # this also applies all the transformations (mean() etc. and therefore might take some time)
    merged.to_zarr(output_path, mode='w', consolidated=True)

    # merged=xr.open_zarr(f"{planet_region_folder}/planet_scope_{scene_date}.zarr")

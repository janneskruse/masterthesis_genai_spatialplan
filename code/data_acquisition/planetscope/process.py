"""
================================================================================
Module to process PlanetScope satellite imagery tiles into xarray datasets.
Provides functions for reading tifs, quality scoring, histogram matching,
reference grid creation, zarr assembly, and single-date processing pipeline.
================================================================================
"""

##### Import libraries ######
# system
import os
import time

# data manipulation
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.enums import Resampling
import rasterio as rio  # needed for xarray.rio to work
import xarray as xr
import rioxarray as rxr
from skimage.exposure import match_histograms
from rioxarray.merge import merge_arrays
import utm
from pyproj import CRS


##### Quality bands mapping for UDM2 #####
QUALITY_BANDS_DICT = {
    'clear': 0,
    'snow': 1,
    'shadow': 2,
    'light_haze': 3,
    'heavy_haze': 4,
    'cloud': 5,
    'confidence': 6,
    'unusable': 7,
}


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


##### Quality / validation helpers #####

def calculate_quality_statistics(
    udm: xr.DataArray,
    bbox_geom,
    quality_bands: list,
    quality_bands_dict: dict | None = None,
) -> dict:
    """
    Calculate UDM2 coverage statistics for quality assessment.

    Parameters
    ----------
    udm : xarray.DataArray
        UDM2 data with 8 bands.
    bbox_geom : shapely.geometry
        Bounding box geometry for area calculation.
    quality_bands : list
        List of quality band names to include in the statistics.
    quality_bands_dict : dict, optional
        Dictionary mapping quality band names to their corresponding band indices.

    Returns
    -------
    dict
        Statistics including percentages for each quality indicator.
    """
    if quality_bands_dict is None:
        quality_bands_dict = QUALITY_BANDS_DICT

    total_pixels = udm.isel(band=0).size

    stats = {
        'total_pixels': total_pixels,
        'aoi_area_km2': bbox_geom.area / 1e6,  # Convert m² to km²
    }

    # Band statistics
    for name in quality_bands:
        band_data = udm.isel(band=quality_bands_dict[name])
        coverage_pixels = (band_data == 1).sum().compute().item()
        coverage_percentage = (coverage_pixels / total_pixels) * 100
        stats[f'{name}_percentage'] = coverage_percentage
        stats[f'{name}_pixels'] = coverage_pixels

    # Calculate usable percentage (inverse of not clear)
    stats['usable_percentage'] = 100 - stats.get('clear_percentage', 0)

    return stats


def create_quality_mask(
    udm_file_path: str,
    bbox_gdf: gpd.GeoDataFrame,
    quality_bands: list,
    quality_bands_dict: dict | None = None,
    confidence_threshold: int = 50,
) -> tuple[xr.DataArray | None, dict | None]:
    """
    Create a quality mask from a UDM2 file.

    Parameters
    ----------
    udm_file_path : str
        Path to the UDM2 file.
    bbox_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the bounding box geometry.
    quality_bands : list
        List of quality band names to include in the mask.
    quality_bands_dict : dict, optional
        Dictionary mapping quality band names to their corresponding band indices.
    confidence_threshold : int
        Threshold for confidence band to consider a pixel as low confidence.

    Returns
    -------
    tuple[xr.DataArray | None, dict | None]
        Quality mask where 1 = unusable pixel, 0 = usable pixel; and statistics dict.
    """
    if quality_bands_dict is None:
        quality_bands_dict = QUALITY_BANDS_DICT

    try:
        # Open UDM2 file
        udm = rxr.open_rasterio(udm_file_path, chunks={'x': 1024, 'y': 1024}, masked=False)
        udm = udm.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)

        tile_stats = {}

        # UDM2 Band Mapping
        # Band 0 (1): Clear map [0, 1] - 0: not clear, 1: clear
        # Band 1 (2): Snow map [0, 1] - 0: no snow, 1: snow
        # Band 2 (3): Shadow map [0, 1] - 0: no shadow, 1: shadow
        # Band 3 (4): Light haze map [0, 1] - 0: no light haze, 1: light haze
        # Band 4 (5): Heavy haze map [0, 1] - 0: no heavy haze, 1: heavy haze
        # Band 5 (6): Cloud map [0, 1] - 0: no cloud, 1: cloud
        # Band 6 (7): Confidence map [0-100] - percentage confidence
        # Band 7 (8): Unusable pixels - equivalent to UDM asset

        # Extract quality masks (True = unusable pixel)
        is_not_clear = udm.isel(band=0) == 0       # Band 1: inverted (0 = not clear)
        is_snow = udm.isel(band=1) == 1             # Band 2: 1 = snow
        is_shadow = udm.isel(band=2) == 1           # Band 3: 1 = shadow
        is_light_haze = udm.isel(band=3) == 1       # Band 4: 1 = light haze
        is_heavy_haze = udm.isel(band=4) == 1       # Band 5: 1 = heavy haze
        is_cloud = udm.isel(band=5) == 1            # Band 6: 1 = cloud
        is_low_confidence = udm.isel(band=6) < confidence_threshold  # Band 7 (confidence)
        is_unusable_band8 = udm.isel(band=7) != 0   # Band 8 (unusable pixels)

        # Combine quality masks based on quality_bands
        quality_mask = xr.zeros_like(udm.isel(band=0), dtype=np.uint8)
        mask_lookup = {
            'clear': is_not_clear,
            'snow': is_snow,
            'shadow': is_shadow,
            'light_haze': is_light_haze,
            'heavy_haze': is_heavy_haze,
            'cloud': is_cloud,
            'low_confidence': is_low_confidence,
            'unusable': is_unusable_band8,
        }
        for name in quality_bands:
            if name in mask_lookup:
                quality_mask = quality_mask | mask_lookup[name]

        # Convert bool to binary (1 = unusable, 0 = usable)
        quality_mask = quality_mask.where(quality_mask == 0, 1)
        quality_mask = quality_mask.rio.write_crs(udm.rio.crs)

        return quality_mask, tile_stats

    except Exception as e:
        print(f"    Failed to read/convert {udm_file_path}: {e} -- skipping this file")
        return None, None


def validate_tiff_file(filepath: str, sample_fraction: float = 0.1) -> bool:
    """
    Validate if a TIFF file is readable and not corrupted.

    Parameters
    ----------
    filepath : str
        Path to the TIFF file.
    sample_fraction : float
        Fraction of the file to sample for validation (0.1 = 10%).

    Returns
    -------
    bool
        True if file is valid, False otherwise.
    """
    try:
        with rxr.open_rasterio(filepath, masked=False) as test_read:
            # Get dimensions
            y_size = test_read.y.size
            x_size = test_read.x.size

            # Sample multiple regions across the file
            sample_points = [
                (0, 0),                                                  # top-left
                (y_size // 2, x_size // 2),                              # center
                (max(0, y_size - 100), max(0, x_size - 100)),            # bottom-right
            ]

            for y_start, x_start in sample_points:
                y_end = min(y_start + 100, y_size)
                x_end = min(x_start + 100, x_size)

                if y_end <= y_start or x_end <= x_start:
                    continue

                _ = test_read.isel(
                    y=slice(y_start, y_end),
                    x=slice(x_start, x_end),
                ).values

        return True
    except Exception as e:
        print(f"    Validation failed for {os.path.basename(filepath)}: {e}")
        return False


def check_overlap(da1: xr.DataArray, da2: xr.DataArray, threshold: float = 100) -> bool:
    """
    Check if two DataArrays overlap by at least *threshold* % of da1's area.

    Parameters
    ----------
    da1 : xr.DataArray
        First raster (reference for area fraction).
    da2 : xr.DataArray
        Second raster to check overlap against.
    threshold : float
        Minimum overlap percentage (0-100).

    Returns
    -------
    bool
        True if overlap >= threshold.
    """
    box1 = box(*da1.rio.bounds())
    box2 = box(*da2.rio.bounds())
    intersection = box1.intersection(box2)
    if intersection.is_empty:
        return False
    return intersection.area / box1.area * 100 >= threshold


def check_pixel_overlap(tile_da: xr.DataArray, ref_da: xr.DataArray, threshold: float = 70) -> bool:
    """
    Check how much of the reference area is covered by non-NaN pixels in the tile.

    Parameters
    ----------
    tile_da : xr.DataArray
        Single-band tile DataArray.
    ref_da : xr.DataArray
        Reference DataArray for total pixel count.
    threshold : float
        Minimum coverage percentage (0-100).

    Returns
    -------
    bool
        True if coverage >= threshold.
    """
    not_nan_mask = np.isfinite(tile_da.values)
    covered_pixels = np.sum(not_nan_mask)
    total_pixels = ref_da.size
    cover_frac = (covered_pixels / total_pixels) * 100
    print(f"    Tile covers {cover_frac:.2f}% of the reference area.")
    return cover_frac >= threshold


##### Read PlanetScope tif + UDM to xarray #####

def read_planetscope_to_xarray_ds(
    filepair: tuple,
    bbox_gdf: gpd.GeoDataFrame,
    quality_bands: list,
    quality_bands_dict: dict | None = None,
    confidence_threshold: int = 50,
) -> xr.Dataset:
    """
    Read PlanetScope tif files to xarray dataset and attach metadata.

    Args:
        filepair: Tuple of (ortho_file, udm_file). udm_file may be None.
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        quality_bands: List of quality band names to include in the quality mask.
        quality_bands_dict: Dictionary mapping quality band names to band indices.
        confidence_threshold: Confidence threshold for UDM2.

    Returns:
        xarray Dataset with planetscope_sr_4band and metadata variables.
    """
    if quality_bands_dict is None:
        quality_bands_dict = QUALITY_BANDS_DICT

    ortho_file = filepair[0]
    udm_file = filepair[1]

    # Open with chunking for memory efficiency
    xda = rxr.open_rasterio(ortho_file, chunks={'x': 1024, 'y': 1024})
    xda = xda.astype("int16")

    # clip to bbox
    xda = xda.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)

    # rename bands to ['blue', 'green', 'red', 'nir']
    xda = xda.rename({"band": "channel"})
    xda = xda.assign_coords(channel=["blue", "green", "red", "nir"])

    # UDM/quality processing
    if udm_file:
        udm_xda, udm_stats = create_quality_mask(
            udm_file_path=udm_file,
            bbox_gdf=bbox_gdf,
            quality_bands=quality_bands,
            quality_bands_dict=quality_bands_dict,
            confidence_threshold=confidence_threshold,
        )

        # filter ortho data with quality mask
        if udm_xda is not None:
            if udm_xda.rio.nodata is None and getattr(udm_xda, "_FillValue", None) is not None:
                udm_xda = udm_xda.rio.write_nodata(udm_xda._FillValue, encoded=True)

            # match ortho resolution / extent
            udm_xda_matched = udm_xda.rio.reproject_match(
                xda.isel(channel=0),
                resampling=Resampling.nearest,
            )
            udm_xda_matched = udm_xda_matched.where(
                udm_xda_matched != udm_xda_matched._FillValue, np.nan
            )

            xda = xda.where(udm_xda_matched == 0)  # set unusable pixels to nan

            # close udm resources
            udm_xda.close()
            udm_xda_matched.close()

    # add attributes
    xda = xda.assign_attrs(
        scale_factor=0.0001,
        offset=0.0,
        units='reflectance',
        description='Analysis-Ready PlanetScope Surface Reflectance',
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

    return xds


##### Quality scoring #####

def extract_quality_score(xds: xr.Dataset, ref_da: xr.DataArray) -> float:
    """
    Extract a quality score from PlanetScope metadata for sorting tiles.

    Args:
        xds: xarray Dataset with meta_planetscope_sr_4band variable.
        ref_da: Reference DataArray used to check pixel overlap.

    Returns:
        Quality score (higher is better).
    """
    try:
        attrs = json.loads(xds["meta_planetscope_sr_4band"].values[0])
        ac = attrs["atmospheric_correction"]
        aot = ac["aot_used"]
        zenith = ac["solar_zenith_angle"]
        score = 1 / ((1 + aot) * (1 + zenith))

        # check how much of the bbox is covered by the tile
        tile_da = xds["planetscope_sr_4band"].isel(time=0, channel=0)
        if not check_pixel_overlap(tile_da, ref_da, threshold=50):
            print("    Less than 50% overlap with reference area, reducing quality score.")
            score *= 0.5  # reduce score if less than 50% of the region is covered
        else:
            print("    Sufficient overlap with reference area.")

        return score
    except Exception:
        return 0  # fallback


##### Histogram matching #####

def histogram_match_global(source: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Fallback global histogram matching when no overlap exists."""
    matched_bands = []

    for b in range(source.shape[0]):
        src_band = source[b].values.copy()
        ref_band = reference[b].values

        valid_src = np.isfinite(src_band) & (src_band > 0)
        valid_ref = np.isfinite(ref_band) & (ref_band > 0)

        matched_band = src_band.astype('float32')

        if valid_src.any() and valid_ref.any():
            matched_band[valid_src] = np.maximum(
                match_histograms(src_band[valid_src], ref_band[valid_ref]),
                0,
            )  # reflectance >= 0

        matched_band[~valid_src] = np.nan
        matched_bands.append(matched_band)

    return xr.DataArray(
        np.stack(matched_bands),
        dims=source.dims,
        coords=source.coords,
        attrs=source.attrs,
    ).rio.write_crs(source.rio.crs)


def histogram_match(source: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """
    Performs histogram matching between a source and a reference DataArray,
    using percentile-based linear scaling on the overlapping area before
    applying histogram matching.
    """
    # Find overlap region
    source_geom = box(*source.rio.bounds())
    ref_geom = box(*reference.rio.bounds())
    overlap_geom = source_geom.intersection(ref_geom)

    if overlap_geom.is_empty:
        print("No overlap - using global histogram matching")
        return histogram_match_global(source, reference)

    # Clip to overlap
    source_overlap = source.rio.clip([overlap_geom], source.rio.crs, drop=False)
    ref_overlap = reference.rio.clip([overlap_geom], source.rio.crs, drop=False)

    matched_bands = []

    for b in range(source.shape[0]):
        src_band = source[b].values.copy()
        src_overlap_band = source_overlap[b].values
        ref_overlap_band = ref_overlap[b].values

        # Valid masks
        valid_src = np.isfinite(src_overlap_band) & (src_overlap_band > 0)
        valid_ref = np.isfinite(ref_overlap_band) & (ref_overlap_band > 0)

        if not (valid_src.any() and valid_ref.any()):
            matched_bands.append(src_band)
            continue

        # percentile-based normalization factors
        src_percentiles = np.percentile(src_overlap_band[valid_src], [2, 98])
        ref_percentiles = np.percentile(ref_overlap_band[valid_ref], [2, 98])

        # linear scaling based on percentiles
        scale = (ref_percentiles[1] - ref_percentiles[0]) / (src_percentiles[1] - src_percentiles[0])
        offset = ref_percentiles[0] - (src_percentiles[0] * scale)

        # apply
        matched_band = src_band.astype('float32')
        valid_full = np.isfinite(src_band) & (src_band > 0)

        # linear scaling with histogram matching
        linear_corrected = src_band[valid_full] * scale + offset

        # Apply histogram matching on linearly corrected data
        hist_matched = match_histograms(
            linear_corrected,
            ref_overlap_band[valid_ref],
        )

        matched_band[valid_full] = np.maximum(hist_matched, 0)  # reflectance >= 0
        matched_band[~valid_full] = np.nan

        matched_bands.append(matched_band)

    # Clean up
    source_overlap.close()
    ref_overlap.close()

    return xr.DataArray(
        np.stack(matched_bands),
        dims=source.dims,
        coords=source.coords,
        attrs=source.attrs,
    ).rio.write_crs(source.rio.crs)


def build_planetscope_date_zarr(
    collection_folder: str,
    bbox_gdf: gpd.GeoDataFrame,
    ref: xr.DataArray,
    scene_date: str,
    output_path: str,
    quality_bands: list | None = None,
    quality_bands_dict: dict | None = None,
    confidence_threshold: int = 50,
):
    """
    Build a PlanetScope zarr dataset for a single date from collection tif files.

    Separates ortho and UDM2 files, validates TIFFs, matches UDM files to ortho
    files, reads with quality masking, sorts by quality, histogram-matches tiles
    incrementally, resamples to the reference grid, computes NDVI, and saves as zarr.

    Args:
        collection_folder: Folder containing the downloaded tif files for this date.
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        ref: Reference DataArray for reprojection matching.
        scene_date: Date string in YYYYMMDD format.
        output_path: Path to save the output zarr dataset.
        quality_bands: List of quality band names to mask. Defaults to all bands.
        quality_bands_dict: Quality band name -> index mapping.
        confidence_threshold: Confidence threshold for UDM2 quality masking.

    Raises:
        ValueError: If no valid xarray datasets could be created.
    """
    if quality_bands is None:
        quality_bands = ['clear', 'snow', 'shadow', 'light_haze', 'heavy_haze', 'cloud', 'unusable']
    if quality_bands_dict is None:
        quality_bands_dict = QUALITY_BANDS_DICT

    if not os.path.exists(collection_folder):
        raise ValueError(f"Collection folder {collection_folder} does not exist")

    collection_files = [
        os.path.join(collection_folder, f)
        for f in os.listdir(collection_folder)
        if f.lower().endswith(".tif")
    ]
    if not collection_files:
        raise ValueError(f"No .tif files found in {collection_folder}")

    ortho_files = [f for f in collection_files if "ortho_analytic_4b_sr" in f]
    if not ortho_files:
        raise ValueError(f"No Ortho Analytic files found in {collection_folder}")

    udm_files = [f for f in collection_files if "ortho_udm2" in f]
    if not udm_files:
        print(f"  No UDM2 files found in {collection_folder} -- processing without quality masking")

    # Build file pairs (ortho, udm) with validation
    file_pairs = []
    print(f"Processing PlanetScope data for date {scene_date} with {len(ortho_files)} tiles...")
    for ortho_file in ortho_files:
        if not validate_tiff_file(ortho_file):
            print(f"    Corrupted ortho file: {os.path.basename(ortho_file)} -> skipping")
            continue

        matching_udm_file = None
        if udm_files:
            udm_file_candidate = ortho_file.replace("ortho_analytic_4b_sr", "ortho_udm2")
            if os.path.exists(udm_file_candidate):
                try:
                    with rxr.open_rasterio(ortho_file, masked=True) as da_ortho:
                        with rxr.open_rasterio(udm_file_candidate, masked=True) as da_udm:
                            if check_overlap(da_ortho, da_udm, threshold=100):
                                if validate_tiff_file(udm_file_candidate):
                                    matching_udm_file = udm_file_candidate
                                else:
                                    print(f"    Corrupted UDM file: {os.path.basename(udm_file_candidate)} -> skipping")
                            else:
                                print(f"  UDM2 file does not overlap >100% with ortho, searching for another...")
                                # find udm file that overlaps > 100%
                                for other_udm in udm_files:
                                    if not os.path.exists(other_udm):
                                        continue
                                    try:
                                        with rxr.open_rasterio(other_udm, masked=True) as da_udm2:
                                            if check_overlap(da_ortho, da_udm2, threshold=100):
                                                matching_udm_file = other_udm
                                                break
                                    except Exception as e:
                                        print(f"  Error checking overlap for {other_udm}: {e}")
                                        continue
                except Exception as e:
                    print(f"  Error checking UDM overlap for {ortho_file}: {e}")

        if matching_udm_file is None and udm_files:
            print(f"  No matching UDM2 file found for {os.path.basename(ortho_file)}")

        file_pairs.append((ortho_file, matching_udm_file))

    # Read all file pairs
    xds_list = []
    for file_pair in file_pairs:
        try:
            xds = read_planetscope_to_xarray_ds(
                file_pair,
                bbox_gdf,
                quality_bands,
                quality_bands_dict,
                confidence_threshold,
            )
            if xds is not None:
                xds_list.append(xds)
        except Exception as e:
            print(f"    Failed to read/convert {file_pair}: {e} -- skipping these files")

    if not xds_list:
        raise ValueError(f"No valid xarray datasets for date {scene_date}")

    try:
        # sort by quality (with pixel overlap check against ref)
        print(f"Sorting {len(xds_list)} tiles by quality for histogram matching...")
        xds_list.sort(key=lambda ds: extract_quality_score(ds, ref), reverse=True)

        dataarrays = [
            ds["planetscope_sr_4band"].squeeze("time").transpose("channel", "y", "x")
            for ds in xds_list
        ]

        # as float for rio merge later
        dataarrays = [da.astype("float32") for da in dataarrays]

        # use the cleanest scene as reference for histogram matching
        reference = dataarrays[0]

        # Apply incremental histogram matching + merging
        print(f"Merging {len(dataarrays)} tiles...")
        if len(dataarrays) > 1:
            for da in dataarrays[1:]:
                try:
                    # Validate tile has valid spatial dimensions
                    if da.x.size == 0 or da.y.size == 0:
                        print(f"Skipping tile with invalid dimensions: x={da.x.size}, y={da.y.size}")
                        continue

                    matched = histogram_match(da, reference)

                    # Validate matched result
                    if matched.x.size == 0 or matched.y.size == 0:
                        print("Matched tile has invalid dimensions, skipping")
                        continue

                    reference = merge_arrays(
                        [reference, matched],
                        method="first",
                        nodata=np.nan,
                        res=None,
                    )

                    # close matched to free resources
                    matched.close()

                except Exception as e:
                    print(f"Histogram matching failed for one tile: {e} -- using original tile")

            # close all dataarrays to free resources
            for da in dataarrays:
                da.close()

        # final merged dataarray
        merged = reference

        # resample to reference dataset
        merged = merged.rio.reproject_match(ref)

        # drop nan coords
        merged = merged.dropna("x", how="all").dropna("y", how="all")

        # Add time dimension and rechunk
        scene_date_np = np.datetime64(pd.to_datetime(scene_date))
        merged = merged.expand_dims(time=[scene_date_np])

        merged = merged.rio.write_nodata(np.nan)
        merged = merged.chunk({'y': 1024, 'x': 1024, 'time': 1, 'channel': 4})

        print("CRS before calling to dataset:", merged.rio.crs)

        # Derive NDVI dataarray from the planetscope data
        merged = merged.to_dataset(name="planetscope_sr_4band")

        # create ndvi and clip to natural range (histogram matching can push reflectance
        # values beyond valid bounds, producing NDVI outside [-1, 1])
        ndvi = (
            (merged.planetscope_sr_4band.isel(channel=3) - merged.planetscope_sr_4band.isel(channel=2))
            / (merged.planetscope_sr_4band.isel(channel=3) + merged.planetscope_sr_4band.isel(channel=2))
        )
        merged["ndvi"] = ndvi.clip(-1, 1)

        print(f"Shape of merged dataset for date {scene_date}: {merged.dims} with CRS {merged.rio.crs}")

        # write to zarr (zarr_format=3 for latest spec)
        merged.to_zarr(output_path, mode="w", zarr_format=3, consolidated=True)
        print(f"Saved merged PlanetScope dataset to {output_path}")

        merged.close()  # free resources
    finally:
        # close all opened xarray datasets to free memory/file handles
        for xds in xds_list:
            try:
                xds.close()
            except Exception:
                pass


def process_single_date(
    filename: str,
    bbox_gdf: gpd.GeoDataFrame,
    ref: xr.DataArray,
    planet_region_folder: str,
    quality_bands: list | None = None,
    quality_bands_dict: dict | None = None,
    confidence_threshold: int = 50,
) -> str | None:
    """
    Process a single PlanetScope date from a collection parquet file into a Zarr file.

    This is a self-contained function suitable for multiprocessing. It reads the
    collection parquet, derives the scene date, then calls build_planetscope_date_zarr
    with the pre-computed bbox and reference grid to produce the output zarr.

    Args:
        filename: Path to the collection parquet file for this date.
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        ref: Reference DataArray for reprojection matching.
        planet_region_folder: Folder for this region's PlanetScope data.
        quality_bands: List of quality band names to mask.
        quality_bands_dict: Quality band name -> index mapping.
        confidence_threshold: Confidence threshold for UDM2 quality masking.

    Returns:
        Path to the output zarr file on success, or None on failure.
    """
    try:
        folderpath = f"{planet_region_folder}/planet_tmp"
        collection = gpd.read_parquet(filename)
        scene_date = collection.date_id.iloc[0]
        scene_date = scene_date.replace("-", "")
        collection_folder = f"{folderpath}/psscene_{scene_date}"

        if not os.path.exists(collection_folder):
            print(f"  Collection folder {collection_folder} does not exist -> skipping date {scene_date}")
            return None

        collection = None  # free memory

        planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"

        if os.path.exists(planet_date_zarr_name):
            print(f"PlanetScope data for date {scene_date} already exists at {planet_date_zarr_name}, skipping processing.")
            return planet_date_zarr_name

        ######### Build the zarr for this date #########
        build_planetscope_date_zarr(
            collection_folder=collection_folder,
            bbox_gdf=bbox_gdf,
            ref=ref,
            scene_date=scene_date,
            output_path=planet_date_zarr_name,
            quality_bands=quality_bands,
            quality_bands_dict=quality_bands_dict,
            confidence_threshold=confidence_threshold,
        )

        print(f"[{scene_date}] Saved PlanetScope dataset to {planet_date_zarr_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return planet_date_zarr_name

    except Exception as e:
        print(f"[process_single_date] Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

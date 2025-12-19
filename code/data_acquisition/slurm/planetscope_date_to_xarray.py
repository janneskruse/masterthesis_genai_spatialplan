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
from dotenv import load_dotenv

# data manipulation
import json
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio as rio # needed for xarray.rio to work
import xarray as xr
import rioxarray as rxr
from skimage.exposure import match_histograms
from rioxarray.merge import merge_arrays
import utm
from pyproj import CRS

# local imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_acquisition.cube.metropolitan_regions import get_region_bbox

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

with open(f"{repo_dir}/code/data_acquisition/config.yml", 'r') as stream:
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
        exit_with_error(f"Region not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    if "FILENAME" in os.environ:
        filename = os.environ["FILENAME"]
    else:
        exit_with_error(f"Filename not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Processing file:", filename, "at", time.strftime("%Y-%m-%d %H:%M:%S"))
    # exit(0)  # Exit early for testing purposes


    folderpath=f"{planet_region_folder}/planet_tmp"
    collection=gpd.read_parquet(filename)
    scene_date=collection.date_id.iloc[0]
    scene_date=scene_date.replace("-","")
    collection_folder=f"{folderpath}/psscene_{scene_date}"
    collection_files=os.listdir(collection_folder)
    collection_files=[f"{collection_folder}/{file}" for file in collection_files]
        
    planet_date_zarr_name = f"{planet_region_folder}/planet_scope_{scene_date}.zarr"
    
    if os.path.exists(planet_date_zarr_name):
        print(f"PlanetScope data for date {scene_date} already exists at {planet_date_zarr_name}, skipping processing.")
        exit(0)
        
    ############ Define the bbox ############ 
    bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)
    bbox_polygon=json.loads(bbox_gdf.to_json())['features'][0]['geometry']
    coordinates=json.loads(bbox_gdf.geometry.to_json())["features"][0]["geometry"]["coordinates"]

    # reproject gdfs to utm zone
    easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
    is_south = zone_letter < 'N'  # True for southern hemisphere
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
    print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")
    bbox_gdf = bbox_gdf.to_crs(utm_crs)

    ###### Prepare reference dataset ##########
    def create_reference_da_from_bounds(bounds, res, crs="EPSG:4326"):
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

    utm_bounds_gdf = bbox_gdf.to_crs(utm_crs)
    bounds = utm_bounds_gdf.total_bounds  # minx, miny, maxx, maxy
    res_m = 3.0
    ref = create_reference_da_from_bounds(bounds, res_m, crs=utm_crs.to_string())
    # ref = ref.rio.reproject("EPSG:4326")


    def readPlanetScopetoXarrayDS(filepath:str):
        """
        Read PlanetScope tif files to xarray dataset and attach metadata.
        
        args:
        filepath: str, path to the tif file    
        """
        # Open with chunking for memory efficiency
        xda=rxr.open_rasterio(filepath, chunks={'x': 1024, 'y': 1024})
        xda = xda.astype("int16")
        # xda = xda.rio.reproject("EPSG:4326")

        #clip to bbox
        xda = xda.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)

        # rename bands to ['blue', 'green', 'red', 'nir']
        xda=xda.rename({"band": "channel"})
        xda=xda.assign_coords(channel=["blue", "green", "red", "nir"])

        #remove spatial_ref coords
        # xda=xda.drop_vars(["spatial_ref"])

        #add attributes
        xda=xda.assign_attrs(
            scale_factor=0.0001,
            offset=0.0,
            units= 'reflectance',
            description= 'Analysis-Ready PlanetScope Surface Reflectance'
        )

        #rename variable
        xda=xda.rename("planetscope_sr_4band")

        #process datetime from attributes
        tiff_datetime = xda.attrs["TIFFTAG_DATETIME"]  # "2019:07:27 08:10:45"
        tiff_datetime = tiff_datetime.replace(":", "-", 2)
        xda = xda.expand_dims(time=[np.datetime64(tiff_datetime)])
        
        #remove unneeded attrs
        xda.attrs.pop("TIFFTAG_DATETIME", None)
        
        
        #add another variable from TIFFTAG_IMAGEDESCRIPTION
        image_description = xda.attrs["TIFFTAG_IMAGEDESCRIPTION"]
        xda.attrs.pop("TIFFTAG_IMAGEDESCRIPTION", None)
        xds=xda.to_dataset()
        xds["meta_planetscope_sr_4band"] = xr.DataArray([image_description], dims=["time"])
        
        # apply scale factor
        xda = xda * xda.scale_factor + xda.offset
    
        return xds

    def extract_quality_score(xds):
        import json
        try:
            attrs = json.loads(xds["meta_planetscope_sr_4band"].values[0])
            ac = attrs["atmospheric_correction"]
            aot = ac["aot_used"]
            zenith = ac["solar_zenith_angle"]
            return 1 / ((1 + aot) * (1 + zenith))
        except:
            return 0  # fallback
    
    def histogram_match(source, reference):
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

    #read all files
    # xds_list=[readPlanetScopetoXarrayDS(file) for file in collection_files]
    xds_list = []
    for file in collection_files:
        try:
            xds = readPlanetScopetoXarrayDS(file)
            if xds is not None:
                xds_list.append(xds)
        except Exception as e:
            print(f"    Failed to read/convert {file}: {e} -- skipping this file")

    if not xds_list:
        print(f"No valid xarray datasets for date {scene_date} -> skipping")
        exit_with_error(f"No valid xarray datasets for date {scene_date}, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
    scene_date_np=np.datetime64(pd.to_datetime(scene_date))
    merged = merged.expand_dims(time=[scene_date_np])
    merged = merged.rio.write_nodata(np.nan)
    merged=merged.chunk({'y': 1024, 'x': 1024, 'time': 1, 'channel': 4})

    # Derive NDVI dataarray from the planetscope data
    #create dataset from merged
    merged = merged.to_dataset(name="planetscope_sr_4band")

    # create ndvi
    merged["ndvi"] = (merged.planetscope_sr_4band.isel(channel=3) - merged.planetscope_sr_4band.isel(channel=2)) / (merged.planetscope_sr_4band.isel(channel=3) + merged.planetscope_sr_4band.isel(channel=2))

    # this also applies all the transformations (mean() etc. and therefore might take some time)
    merged.to_zarr(planet_date_zarr_name, mode='w', consolidated=True)

    # merged=xr.open_zarr(f"{planet_region_folder}/planet_scope_{scene_date}.zarr")

except Exception as e:
    print(f"An error occurred: {e}")
    exit_with_error(f"An error occurred: {e}")
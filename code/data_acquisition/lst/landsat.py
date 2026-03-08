"""
================================================================================
Module to acquire and pre-process Landsat LST data.
Provides functions for STAC queries, S3 downloads, tif-to-xarray conversion,
and zarr dataset assembly.
================================================================================
"""

##### Import libraries ######
# system
import os
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from dotenv import load_dotenv

# downloading and website scraping
import requests

# aws bucket access
import boto3

# data manipulation
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio  # needed to load for xarray conversions
import xarray as xr
import rioxarray as rxr

# visualization
from tqdm.auto import tqdm


################## Get LST Data from Landsat #################
# How to query the stac api: https://code.usgs.gov/eros-user-services/quick-guides/querying-the-stac-api-with-geojson-objects/-/blob/main/querying_with_geojson_objects_v3.ipynb?ref_type=heads
# Further information on the post requests can be found on the node api documentation:
# https://github.com/stac-utils/stac-server

########## Define the query functions ##########
# Function to query the stac server for features with boundary geolocation
def fetch_stac_server(query: dict) -> gpd.GeoDataFrame | None:
    '''
    Queries the stac-server (STAC) backend.
    query is a python dictionary to pass as json to the request.
    '''
    
    search_url = f"https://landsatlook.usgs.gov/stac-server/search"
    query_return = requests.post(search_url, json=query).json()
    error = query_return.get("message", "")
    if error:
        raise Exception(f"STAC-Server failed and returned: {error}")
        
    if 'code' in query_return:  # if query fails, return failure code
        print(query_return)   
    else:
        features = query_return['features']
        #print(f"{len(features)} STAC items found")
        if len(features) > 0:
            #print(f"first feature: {features[0]}")
            
            query_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            query_gdf['assets'] = [ 
                feature["assets"]
                for feature in features
            ]
            query_gdf['description'] = [feature["description"] for feature in features]
            query_gdf['stac_id'] = [feature["id"] for feature in features]

            return query_gdf
        else:
            #print("No features found")
            return None


# Function to send a filtered request to the stac server using the function above:
def send_STAC_query(
    limit: int = 200,
    collections: str = 'landsat-c2l2-st',
    intersects: dict | None = None,
    year: str | None = None,
    month: str | None = None,
    date_list: list[str] | None = None,
    max_cloud_cover: int | None = None
) -> gpd.GeoDataFrame | None:
    '''
    This function helps to create a simple parameter dictionary for querying 
    the Landsat Collection 2 Level 2 Surface Reflectance feature in the STAC Server.
    It prints the parameter dictionary and returns the query results.
    
    args:
    limit: int, default 200, number of items to return
    collections: str, default 'landsat-c2l2-st', collection to query
    intersects: dict, default None, geometry to intersect with
    year: str, default None, year to filter by
    month: str, default None, month to filter by in format '01'
    date_list: list, default None, list of dates (YYYY-MM-DD) to filter by
    '''
    params = {}
    if limit is not None:
        params['limit'] = limit
    
    if collections is not None:
        params['collections'] = collections
        
    if intersects is not None:
        params['intersects'] = intersects
        
    if max_cloud_cover is not None:
        params['query'] = {
            "eo:cloud_cover": {
                "lte": max_cloud_cover
            }
        }
        
    # filter by date
    if date_list is not None:
        formatted_dates = [f"{date}T00:00:00Z" for date in date_list]
        params["datetime"] = ",".join(formatted_dates)

        all_results = []

        for date in date_list:
            params["datetime"] = f"{date}T00:00:00Z/{date}T23:59:59Z"
            
            #print(f"Querying STAC for date: {date}")
            result = fetch_stac_server(params)

            if result is not None:
                all_results.append(result)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return None
        
    else:
        max_day = 31
        
        if year is not None:
            params['datetime'] = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
        if month is not None:
            # set last day for month
            max_day = calendar.monthrange(int(year), int(month))[1]
            
            params['datetime'] = f"1970-{month}-01T00:00:00Z/2024-{month}-{max_day}T23:59:59Z"
        if year is not None and month is not None:        
            params['datetime'] = f"{year}-{month}-01T00:00:00Z/{year}-{month}-{max_day}T23:59:59Z"
        
        print(params) 
        
        return fetch_stac_server(params)


def query_stac_for_dates(
    dates: list[str],
    bbox_polygon: dict,
    collections: list[str],
    max_cloud_cover: int,
    chunk_size: int = 20,
    max_workers: int = 8
) -> gpd.GeoDataFrame:
    """
    Query STAC server for Landsat products on the given dates using multithreaded requests.
    
    Args:
        dates: List of date strings in YYYY-MM-DD format.
        bbox_polygon: GeoJSON polygon geometry for spatial intersection.
        collections: STAC collection identifiers.
        max_cloud_cover: Maximum cloud cover percentage.
        chunk_size: Number of dates per query chunk.
        max_workers: Maximum number of parallel workers.
        
    Returns:
        GeoDataFrame with STAC query results.
    """
    # Query for the dates using multithreaded requests
    chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]

    query_gdf = pd.DataFrame()

    # multithread requests using tqdm progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                send_STAC_query, 
                intersects=bbox_polygon, 
                limit=1, 
                date_list=chunk, 
                collections=collections, 
                max_cloud_cover=max_cloud_cover
            ) 
            for chunk in chunks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                if len(result) > 0:
                    query_gdf = pd.concat([query_gdf, result], ignore_index=True)

    return query_gdf


####### Get the images for the requested collection information #######
# For information on the asset links:
# https://landsat.usgs.gov/stac/LC09_L2SP_095022_20220625_20220627_02_T1_ST_stac.json
def get_landsat_temperature_products(query_gdf: gpd.GeoDataFrame) -> list[dict]:
    '''
    This function retrieves the Landsat 8 Surface Temperature products from the query results.
    It returns a list of the product urls.
    '''
    products = []
    for index, row in query_gdf.iterrows():
        assets = row.assets
        # lwir11 is the Surface Temperature Band (B10), see: https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l2-st/items/LC09_L2SP_095022_20220625_20230409_02_T1_ST
        if 'lwir11' in assets and assets['lwir11'] is not None:
            products.append({"stac_id": row.stac_id, "datetime": row.datetime,
                            "thermal": {
                            "url": assets['lwir11']['href'],
                            "alternate": assets['lwir11']['alternate']},
                            "qa_pixel": {"url": assets['qa_pixel']['href'], "alternate": assets['qa_pixel']['alternate']}
                            })
                            
        elif 'lwir' in assets and assets['lwir'] is not None:
            print(f"found only B6 from Landsat 7 for {row.stac_id}")
            # products.append({"stac_id": row.stac_id, "datetime": row.datetime, 
                            #  "url": assets['lwir']['href'], "alternate": assets['lwir']['alternate']})
            
        else:
            print(f"No lwir11 asset for {row.stac_id}")

    return products


### Download the images using AWS CLI

def _setup_s3_client() -> boto3.client:
    """
    Setup the boto3 S3 client using environment variables.
    
    Returns:
        Configured boto3 S3 client.
    """
    # get aws credentials
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # Setup the boto3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='us-west-2'
    )
    return s3


def _parse_s3_url(s3_url: str) -> tuple[str, str]:
    if s3_url.startswith("s3://"):
        s3_url = s3_url[5:]
    bucket, key = s3_url.split("/", 1)
    return bucket, key


class TqdmFileWrapper:
    def __init__(self, fileobj, tqdm_bar):
        self.fileobj = fileobj
        self.tqdm_bar = tqdm_bar

    def write(self, data):
        self.fileobj.write(data)
        self.tqdm_bar.update(len(data))

    def close(self):
        self.fileobj.close()


def _download_tif(s3: boto3.client, s3_url: str, local_path: str, position: int = 0):
    bucket, key = _parse_s3_url(s3_url)

    # Find out the file size
    head = s3.head_object(Bucket=bucket, Key=key, RequestPayer='requester')
    file_size = head['ContentLength']

    # Setup tqdm progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(local_path)}", position=position) as pbar:
        with open(local_path, 'wb') as f:
            wrapped_file = TqdmFileWrapper(f, pbar)
            s3.download_fileobj(
                Bucket=bucket,
                Key=key,
                Fileobj=wrapped_file,
                ExtraArgs={'RequestPayer': 'requester'}
            )


def download_product(
    product: dict,
    landsat_region_folder: str,
    s3: boto3.client,
    position: int = 0
):
    '''
    This function retrieves the products from the USGS server and saves it to the output path.
    It stores both the thermal and the qa_pixel tif files.
    '''
    
    output_path_base = f"{landsat_region_folder}/landsat_temperature/{product['stac_id']}"
    output_path_thermal = f"{output_path_base}_thermal.tif"
    output_path_qa_pixel = f"{output_path_base}_qa_pixel.tif"
    
    if not os.path.exists(output_path_thermal):
        os.makedirs(os.path.dirname(output_path_thermal), exist_ok=True)
        s3_url = product["thermal"]['alternate']['s3']['href']
        _download_tif(s3, s3_url, output_path_thermal, position)

    if not os.path.exists(output_path_qa_pixel):
        os.makedirs(os.path.dirname(output_path_qa_pixel), exist_ok=True)
        s3_url = product["qa_pixel"]['alternate']['s3']['href']
        _download_tif(s3, s3_url, output_path_qa_pixel, position + 1)

    return


def download_all_products(
    products: list[dict],
    landsat_region_folder: str,
    max_workers: int = 8
):
    """
    Thread download all the datasets.
    
    Args:
        products: List of product dicts from get_landsat_temperature_products.
        landsat_region_folder: Path to the region folder for saving tifs.
        max_workers: Maximum number of parallel download workers.
    """
    # Load .env and setup S3 client
    s3 = _setup_s3_client()
    
    # Setup tqdm lock to prevent corruption of output
    tqdm.set_lock(RLock())
    
    # Download first product
    download_product(products[0], landsat_region_folder, s3)
    
    # Thread download all the datasets
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_product, product, landsat_region_folder, s3, i * 2): product
            for i, product in enumerate(products, start=1)
        }


######### Create a pre-processed/cleaned zarr for the Landsat data #########
def read_landsat_tifs_to_xarray_ds(
    ds_file_path_set: list[str],
    query_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame
) -> xr.Dataset | None:
    """
    Read Landsat tif files to xarray dataset and attach metadata.
    
    Args:
        ds_file_path_set: List of [qa_pixel_path, thermal_path].
        query_gdf: GeoDataFrame with STAC query results for metadata lookup.
        bbox_gdf: GeoDataFrame with the bounding box geometry for clipping.
        
    Returns:
        xarray Dataset with thermal, qa_pixel, and masked variables, or None if file not found.
    """
    
    thermal_file_path = ds_file_path_set[1]
    qa_pixel_file_path = ds_file_path_set[0]
    
    # find file in query_gdf
    stac_id = thermal_file_path.split("/")[-1].split(".")[0].split("_thermal")[0]
    file = query_gdf[query_gdf['stac_id'] == stac_id]
    if len(file) > 0:
        file = file.iloc[0]
    else:
        print(f"File not found in query_gdf: {stac_id}")
        return None
    
    if not os.path.exists(thermal_file_path):
        print(f"File not found: {thermal_file_path}")
        return None
    if not os.path.exists(qa_pixel_file_path):
        print(f"File not found: {qa_pixel_file_path}")
        return None
    
    #####create thermal data array######
    # read file
    xda = rxr.open_rasterio(thermal_file_path, masked=True)
    xda = xda.rio.reproject("EPSG:4326")
    
    # clip to bbox
    xda = xda.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)
    
    # apply scaling factor to degrees celsius
    scale_factor = 0.00341802
    add_offset = 149.0 - 273.15
    #xda=xda*scale_factor+add_offset
    xda.attrs['scale_factor'] = scale_factor
    xda.attrs['add_offset'] = add_offset
    
    # rename data array
    xda = xda.rename('surface_temp_b10')
    
    
    #####create qa pixel data array#######
    # read file
    xda_qa = rxr.open_rasterio(qa_pixel_file_path)
    
    # fill nodata values with 0
    xda_qa = xda_qa.rio.write_nodata(0, inplace=True)
    xda_qa = xda_qa.rio.reproject("EPSG:4326")
    
    # clip to bbox
    xda_qa = xda_qa.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)
    
    # mask cloud and cloud shadow
    # Define bit positions for cloud and cloud shadow
    CLOUD_SHADOW_BIT = 3  # Bit 3 = cloud shadow
    CLOUD_BIT = 5         # Bit 5 = cloud

    # Create masks
    cloud_shadow_mask = (xda_qa & (1 << CLOUD_SHADOW_BIT)) == 0  # True = no shadow
    cloud_mask = (xda_qa & (1 << CLOUD_BIT)) == 0                 # True = no cloud

    clear_mask = cloud_shadow_mask & cloud_mask
    xda_qa = clear_mask.astype(np.uint8)
    
    # rename data array
    xda_qa = xda_qa.rename('qa_pixel')
    
    
    #####create masked array #####
    # create masked array
    xda_mask = xda.where(xda_qa)
    
    # rename data array
    xda_mask = xda_mask.rename('surface_temp_b10_masked')
    
    
    #####combine data arrays####
    xds = xr.merge([xda, xda_qa, xda_mask])
    
    
    #####add metadata#####
    # get stac data
    date = file.datetime
    
    # add general metadata
    xds.attrs['title'] = "Landsat 8 Surface Temperature"
    xds.attrs['description'] = "Landsat 8 Surface Temperature data from USGS for specific hot days (3 continuous >30C° days)"
    xds.attrs['source'] = "USGS"
    xds.attrs['crs'] = "EPSG:4326"
    xds.attrs['bbox'] = bbox_gdf.to_json()
    xds.attrs['variables'] = {"surface_temp_b10": "Surface Temperature Band (B10)",
                                "qa_pixel": "Quality Assessment Pixel",
                                "surface_temp_b10_masked": "Surface Temperature Band (B10) Masked"}
    xds.attrs['units'] = {"surface_temp_b10": "°C", "qa_pixel": "1", "surface_temp_b10_masked": "°C"}
    
    # remove scale_factor and add_offset from attrs
    xds.attrs.pop('scale_factor', None)
    xds.attrs.pop('add_offset', None)
    
    # remove spatial_ref and band coords
    xds = xds.drop_vars(["spatial_ref", "band"])
    
    # squeeze band from variables
    xds = xds.squeeze("band", drop=True)
    
    # add time coordinate
    xds = xds.expand_dims(time=[date])
    # xds=xds.expand_dims(stac_id=[stac_id])
    # xds=xds.expand_dims(view_sun_elevation=[file['view:sun_elevation']])
    # xds=xds.expand_dims(view_sun_azimuth=[file['view:sun_azimuth']])
    # xds=xds.expand_dims(view_off_nadir=[file['view:off_nadir']])

    # add metadata as variables over time
    xds['stac_id'] = xr.DataArray([stac_id], dims=['time'])
    xds['view_sun_elevation'] = xr.DataArray([file['view:sun_elevation']], dims=['time'])
    xds['view_sun_azimuth'] = xr.DataArray([file['view:sun_azimuth']], dims=['time'])
    xds['view_off_nadir'] = xr.DataArray([file['view:off_nadir']], dims=['time'])

    return xds


def build_landsat_zarr(
    landsat_region_folder: str,
    query_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    max_cloud_cover: int,
    max_dates_per_year: int | None,
    landsat_zarr_name: str
):
    """
    Create a pre-processed/cleaned zarr dataset for the Landsat data.
    
    Args:
        landsat_region_folder: Path to the region folder containing downloaded tifs.
        query_gdf: GeoDataFrame with STAC query results.
        bbox_gdf: GeoDataFrame with the bounding box geometry.
        max_cloud_cover: Maximum cloud cover percentage for filtering.
        max_dates_per_year: Maximum number of dates to keep per year (None to skip).
        landsat_zarr_name: Output path for the zarr dataset.
    """
    # read files
    files = os.listdir(f"{landsat_region_folder}/landsat_temperature")
    files = [file for file in files if file.startswith("LC08")]
    file_paths = [f"{landsat_region_folder}/landsat_temperature/{file}" for file in files]
    thermal_file_paths = [file for file in file_paths if file.endswith("thermal.tif")]
    qa_pixel_file_paths = [file for file in file_paths if file.endswith("qa_pixel.tif")]

    # find qa pixel files for thermal files and create set of file paths
    file_sets = []
    for thermal_file in thermal_file_paths:
        qa_pixel_file = thermal_file.replace("thermal", "qa_pixel")
        if qa_pixel_file in qa_pixel_file_paths:
            file_sets.append([qa_pixel_file, thermal_file])

    # read all files
    xds_list = [
        ds for file_set in file_sets
        if (ds := read_landsat_tifs_to_xarray_ds(file_set, query_gdf, bbox_gdf)) is not None
    ]

    ###### reindex to common grid ######
    # Therefore, the coordinates should be reindexed to a template grid before concatenating the data, so the data aligns.

    common_x = xds_list[0].x
    common_y = xds_list[0].y

    xds_list = [
        ds.reindex(x=common_x, y=common_y, method="nearest")  # or method="pad"
        for ds in xds_list
    ]

    # combine datasets
    landsat_xr_ds = xr.concat(xds_list, dim='time')
    
    # ensure time coordinate is in timezone-naive datetime64[ns] format 
    landsat_xr_ds = landsat_xr_ds.assign_coords(
        time=pd.to_datetime(landsat_xr_ds.time.values).tz_localize(None)
    )

    ###### filter data a second time ######
    # Somehow the stac filtering did not work on all files, so a second filter is applied here to remove all files where the qa_pixel is not null more than the allowed configuration percentage.
    # filter out all timesteps where no data values are present
    mask = landsat_xr_ds.surface_temp_b10_masked.notnull().compute()
    landsat_xr_ds = landsat_xr_ds.where(mask, drop=True)

    # filter where qa_pixel not more than max_cloud_cover percentage
    valid_pixel_percentage = landsat_xr_ds.qa_pixel.notnull().mean(dim=['x', 'y']).compute()
    landsat_xr_ds = landsat_xr_ds.where(valid_pixel_percentage >= (100 - max_cloud_cover) / 100, drop=True)

    if max_dates_per_year:
        # group by year and take the first max_dates_per_year dates
        landsat_xr_ds = landsat_xr_ds.groupby('time.year').apply(
            lambda x: x.isel(time=np.arange(min(max_dates_per_year, len(x.time))))
        )
    else:
        print("No max_dates_per_year configured, skipping this step.")

    # save as zarr dataset
    landsat_xr_ds.to_zarr(landsat_zarr_name, mode='w')

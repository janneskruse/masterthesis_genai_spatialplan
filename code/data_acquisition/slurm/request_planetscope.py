## Script to aquire and pre-process Planet Lab's (Planetscope) data to an Xarray cube
# Planetscope images are high resolution (3m) satellite images from Planet Labs
# Planet lab's has a rest api for metadata based search: https://developers.planet.com/docs/apis/data/reference/#tag/Item-Search
# More information on search filters etc. can be found here: https://developers.planet.com/docs/apis/data/searches-filtering/
# From the results, the images then can be downloaded like indicated here:
# https://developers.planet.com/docs/planetschool/downloading-imagery-with-data-api/

## Import libraries
# system
import os
import sys
import time
import calendar
import requests
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
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
import utm
from pyproj import CRS

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
sys.path.append(f"{repo_dir}/code/helpers")
from submit_job import submit_job_with_dependency

with open(f"{repo_dir}/config.yml", 'r') as stream:
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

try:
    if "REGION_FILENAMES_JSON" in os.environ:
        region_filenames_json = os.environ["REGION_FILENAMES_JSON"]
    else:
        exit_with_error(f"Region filenames JSON not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    print("Error getting region filenames JSON from environment:", e)
    exit_with_error(f"Region filenames JSON not set in environment, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

planet_zarr_name = f"{planet_region_folder}/planet_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

print(f"Requesting PlanetScope with Landsat Zarr file: {landsat_zarr_name} for region: {region} at {time.strftime('%Y-%m-%d %H:%M:%S')} to store at {planet_zarr_name}")

# comment this out for testing:
# test_folderpath=f"{planet_region_folder}/planet_tmp"
# test_filenames=[f"{test_folderpath}/planet_scope_cover_{i.replace('-','')}.parquet" for i in ["2023-01-01", "2023-02-01", "2023-03-01"]]
# submit_job_with_dependency("./process_planetscope.sh", region=region, landsat_zarr_name=landsat_zarr_name, filenames=test_filenames, region_filenames_json=region_filenames_json)
# exit(0)  # Exit early for testing purposes

######## Planet data processing ########
try:
    
    if os.path.exists(planet_zarr_name):
        print(f"PlanetScope data already exists at {planet_zarr_name}, skipping processing.")
        exit(0)
    
    ############ Define the bbox ############ 
    ghsl_df_new = gpd.read_parquet(f"{repo_dir}/data/processed/ghsl_regions.parquet")
    bbox_gdf = gpd.GeoDataFrame(geometry=ghsl_df_new[ghsl_df_new["region_name"]==region].bbox, crs="EPSG:4326")
    bbox_polygon=json.loads(bbox_gdf.to_json())['features'][0]['geometry']
    coordinates=json.loads(bbox_gdf.geometry.to_json())["features"][0]["geometry"]["coordinates"]

    ########## get unique dates from landsat zarr file ##########
    #import landsat xarray dataset
    landsat_xr_ds=xr.open_zarr(landsat_zarr_name, consolidated=True)

    # Get time values from xarray dataset
    time_ranges = landsat_xr_ds.time
    time_ranges = [pd.to_datetime(timestamp).strftime("%Y-%m-%d") for timestamp in time_ranges.values]
    
    # Remove duplicates and sort
    time_ranges = sorted(set(time_ranges))
    time_ids=[(i, t) for i, t in enumerate(time_ranges)]
    

    ###### get planet scenes for the bbox and time ranges ######
    def requestPlanetItemInfo(item_types:list=["PSScene"], date:str=None, cloud_cover_limit:float=None, download:bool=False, further_filters:dict=None):
    
        # Define filters
        filters=[
                {
                "type":"GeometryFilter",
                "field_name":"geometry",
                "config":{
                    "type":"Polygon",
                    "coordinates": coordinates,
                }
                },
                # {
                #    "type":"AssetFilter",
                #    "config":[
                #       "ortho_analytic_8b"
                #    ]
            ]
        
        if cloud_cover_limit is not None:
            cloud_cover_filter={
                "type":"RangeFilter",
                "config":{
                    "gte":0,
                    "lte":cloud_cover_limit #0.6
                },
                "field_name":"cloud_cover"
            }
            filters.append(cloud_cover_filter)
            
        if download:
            download_filter={
                "type":"PermissionFilter",
                "config":[
                    "assets:download"
                ]
            }
            filters.append(download_filter)
            
        if further_filters is not None:
            filters.append(further_filters)

        if date is not None:
            year=date.split("-")[0]
            month=int(date.split("-")[1])
            
                    
            # define a date range of plus and minus 1 month
            start_month=str(month-1 if month != 1 else 12).zfill(2)
            end_month=str(month+1 if month !=12 else 1).zfill(2)
            end_day=calendar.monthrange(int(year), int(end_month))[1]
            
            local_start_year=year if month != 1 else str(int(year)-1)
            local_end_year=year if month != 12 else str(int(year)+1)

            start=f"{local_start_year}-{start_month}-01"
            end=f"{local_end_year}-{end_month}-{end_day}"
            
            date_range_filter={
                "type":"DateRangeFilter",
                "field_name":"acquired",
                "config":{
                    "gte":f"{start}T00:00:00Z",
                    "lte":f"{end}T00:00:00Z"
                }
            }
            
            filters.append(date_range_filter)


        payload = {
                "item_types": item_types,
                "filter": {
                    "type": "AndFilter",
                    "config": filters
                }
            }
        
        print(f"Requesting Planet items for date: {date}", payload)

        # Send POST request
        response=requests.post(url, auth=planet_api_key, json=payload)
        response=response.json()
        if "features" not in response:
            #print(response)
            return None
        else:
            #print(response)
            features_df=pd.DataFrame(response["features"])
            
            return features_df

    # planet scope ("PSScene")
    item_types=["PSScene"]
    cloud_cover_limit=0.1 #max 10% cloud cover

    #thread collect for all time ranges
    with ThreadPoolExecutor(max_workers=8) as executor:
        planet_bydate_list = list(tqdm(executor.map(lambda date: requestPlanetItemInfo(item_types=item_types, date=date, cloud_cover_limit=cloud_cover_limit), time_ranges), total=len(time_ranges)))
        
    #merge all dataframes
    planet_bydate_list=[df for df in planet_bydate_list if df is not None]
    if not planet_bydate_list:
        print("No PlanetScope items found")
        exit_with_error(f"No PlanetScope items found, finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        planet_bydate_df=pd.concat(planet_bydate_list, ignore_index=True)
        planet_bydate_df.head(2)

    planet_bydate_gdf=gpd.GeoDataFrame(planet_bydate_df, geometry=[shape(geom) for geom in planet_bydate_df["geometry"]], crs="EPSG:4326")

    # reproject gdfs to utm zone
    easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
    is_south = zone_letter < 'N'  # True for southern hemisphere
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
    print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")

    planet_bydate_gdf = planet_bydate_gdf.to_crs(utm_crs)
    bbox_gdf = bbox_gdf.to_crs(utm_crs)

    # Save scene metadata as geoparquet
    meta_filename=f"{planet_region_folder}/planet_ge{min_temperature}_{start_year}_{end_year}_meta.parquet"
    planet_bydate_gdf.to_parquet(meta_filename)
    
    def mergeNearestRows(df, bbox_gdf, max_distance=0.01):
        '''
        Merges nearest rows of a GeoDataFrame until the merged geometry fully covers a reference bbox_gdf.
        
        Parameters
        ----------
        df : GeoDataFrame
            Input tiles or geometries, sorted by some priority (e.g., temporal closeness)
        bbox_gdf : GeoDataFrame
            Contains the target bounding box (1 row with 1 Polygon/Multipolygon)
        max_distance : float
            Maximum allowed distance for adding new geometries (same units as CRS)

        Returns
        -------
        merged_gdf : GeoDataFrame
            Merged rows that together cover the full bbox_gdf
        '''
        if df.empty:
            return gpd.GeoDataFrame(columns=df.columns, crs=df.crs)
        
        merged_gdf = gpd.GeoDataFrame()
        bbox_geom = bbox_gdf.union_all()

        while not df.empty:
            row = df.iloc[0]
            df = df.iloc[1:] # exclude the first row
            merged_gdf = pd.concat([merged_gdf, gpd.GeoDataFrame([row], crs=df.crs)], ignore_index=True)

            # Update the merged geometry
            merged_geom = merged_gdf.union_all()

            if merged_geom.covers(bbox_geom):
                # Success: fully covered the bbox
                break
            
            inter_area = merged_geom.intersection(bbox_geom).area
            bbox_area = bbox_geom.area if bbox_geom is not None else 0
            cover_frac = inter_area / bbox_area if bbox_area > 0 else 0
            
            if cover_frac >= 1:
                break

            # Find the nearest geometry to the current merged geometry
            distances = df.distance(merged_geom)
            if distances.empty:
                break
            nearest_idx = distances.idxmin()

            if distances[nearest_idx] < max_distance:
                nearest_row = df.loc[[nearest_idx]]
                df = df.drop(nearest_idx)
                merged_gdf = pd.concat([merged_gdf, nearest_row], ignore_index=True)
            else:
                # If no nearby geometry is available, stop (optional - could also continue and allow gaps)
                break

        inter_area = merged_geom.intersection(bbox_geom).area
        bbox_area = bbox_geom.area if bbox_geom is not None else 0
        cover_frac = inter_area / bbox_area if bbox_area > 0 else 0

        print(f"Coverage fraction of the first date's merged geometries over the bbox: {cover_frac:.2%}")

        # Final check
        if not merged_gdf.union_all().covers(bbox_geom) and cover_frac < 1:
            raise ValueError("Failed to fully cover the target bbox with available geometries.")

        return merged_gdf


    def getPlanetscopeScenesCoverForDate(time_id):
        global landsat_xr_ds, planet_bydate_gdf, bbox_gdf
        
        # Get time values from xarray dataset
        time_stamp = landsat_xr_ds.isel(time=time_id).time.values
        time_stamp = pd.to_datetime(time_stamp).strftime("%Y-%m-%d")
        #time_stamp_flat=time_stamp.replace("-", "")
        time_stamp_flat_month=time_stamp.replace("-", "")[:-2]
        month=int(time_stamp_flat_month[-2:])
        previous_month=str(month-1 if month != 1 else 12).zfill(2)
        previous_month_year=time_stamp_flat_month[:-2] if month != 1 else str(int(time_stamp_flat_month[:-2])-1)
        previous_month_time_stamp_flat=f"{previous_month_year}{previous_month}"
        next_month=str(month+1 if month !=12 else 1).zfill(2)
        next_month_year=time_stamp_flat_month[:-2] if month != 12 else str(int(time_stamp_flat_month[:-2])+1)
        next_month_time_stamp_flat=f"{next_month_year}{next_month}"

        # filter planet_bydate_gdf by id of time_stamp_flat_month, previous_month_time_stamp_flat, next_month_time_stamp_flat
        planet_bydate_gdf_filtered= planet_bydate_gdf[
            planet_bydate_gdf['id'].str.contains(time_stamp_flat_month) | 
            planet_bydate_gdf['id'].str.contains(previous_month_time_stamp_flat) | 
            planet_bydate_gdf['id'].str.contains(next_month_time_stamp_flat)].copy()

        #create date id
        planet_bydate_gdf_filtered['date_id']=planet_bydate_gdf_filtered['id'].str[0:8]
        
        planet_bydate_gdf_filtered_clipped = planet_bydate_gdf_filtered.clip(bbox_gdf)
        
        #get the nearest ids for time_stamp_flat
        planet_bydate_gdf_filtered_clipped.loc[:, 'date_id_dt'] = pd.to_datetime(planet_bydate_gdf_filtered_clipped['date_id'])
        planet_bydate_gdf_filtered_clipped.loc[:, 'time_stamp_dt'] = pd.to_datetime(time_stamp)
        planet_bydate_gdf_filtered_clipped.loc[:, 'time_diff'] = (planet_bydate_gdf_filtered_clipped['date_id_dt'] - planet_bydate_gdf_filtered_clipped['time_stamp_dt']).dt.days
        planet_bydate_gdf_filtered_clipped.loc[:, 'time_diff'] = planet_bydate_gdf_filtered_clipped['time_diff'].abs()

        #sort the dataframe by diff
        planet_bydate_gdf_filtered_clipped.sort_values('time_diff', inplace=True)
        
        planet_scenes_cover_df=mergeNearestRows(planet_bydate_gdf_filtered_clipped, bbox_gdf)
        
        return planet_scenes_cover_df

    # Run for all timestamps
    # planet_scope_cover_df_list=[(date_value,getPlanetscopeScenesCoverForDate(date_id)) for date_id, date_value in time_ids]

    # # Save fully covered scene meta as geoparquet
    # filenames= []
    # folderpath=f"{planet_region_folder}/planet_tmp"
    # os.makedirs(folderpath, exist_ok=True)
    # for time_id, df in planet_scope_cover_df_list:
    #     filename=f"{folderpath}/planet_scope_cover_{time_id.replace('-','')}.parquet"
    #     filenames.append(filename)
        
    #     # reproject to original crs
    #     df = df.to_crs("EPSG:4326")
        
    #     df.to_parquet(filename)
    #     print(f"Saved: {filename}")

    folderpath = f"{planet_region_folder}/planet_tmp"
    os.makedirs(folderpath, exist_ok=True)

    filenames = []
    planet_scope_cover_df_list = []

    for time_id, date_value in time_ids:
        filename = f"{folderpath}/planet_scope_cover_{date_value.replace('-','')}.parquet"
        filenames.append(filename)

        if os.path.exists(filename):
            print(f"Loading existing cover file: {filename}")
            try:
                df = gpd.read_parquet(filename)
            except Exception as e:
                print(f"Failed to read {filename} ({e}), recomputing...")
                df = getPlanetscopeScenesCoverForDate(time_id)
                df = df.to_crs("EPSG:4326")
                df.to_parquet(filename)
                print(f"Saved: {filename}")
        else:
            print(f"No existing file for {date_value}, computing cover...")
            df = getPlanetscopeScenesCoverForDate(time_id)
            df = df.to_crs("EPSG:4326")
            df.to_parquet(filename)
            print(f"Saved: {filename}")

        planet_scope_cover_df_list.append((date_value, df))

    ######### Order items for download #########
    # orders_base_url="https://api.planet.com/compute/ops/orders/v2"

    # def orderPlanetItem(date:str, ids:list, product:str='analytic_sr_udm2', clip_aoi:dict=None):
    #     """
    #     Request Planet item by ids and product.
        
    #     args:
    #     date: str, flat date like "20200813"
    #     ids: list of ids like ["20201129_150531_19_1065"]
    #     product: str, product name like "analytic_sr_udm2" for "ortho_analytic_4b_sr"
    #     clip_aoi: dict, clip area of interest as polygon coordinates
    #     """
    #     payload={
    #         "name": f"clip_{date}",
    #         "source_type": "scenes",
    #         "products": [
    #             {
    #             "item_ids": ids,
    #             "item_type": "PSScene",
    #             "product_bundle": product,
    #             }
    #         ],
    #         "tools": [
    #             {
    #             "clip": {
    #                 "aoi": {
    #                 "type": "Polygon",
    #                 "coordinates": clip_aoi,
    #                 }
    #             }
    #             }
    #         ]
    #     }
        
    #     response=requests.post(orders_base_url, auth=planet_api_key, json=payload)
        
    #     return response.json()


    # # Get all assets for the file
    # item=gpd.read_parquet(filenames[0]).iloc[0]
    # response=requests.get(item._links['assets'], auth=planet_api_key)
    # assets=response.json()
    # assets=pd.DataFrame(assets).T

    # ##### Submit clip order
    # collection=gpd.read_parquet(filenames[0])
    # date=collection.date_id.iloc[0]
    # collection_ids=collection.id.to_list()

    # orderPlanetItem(date=date, ids=collection_ids, clip_aoi=coordinates)


    ######### Request download for all scenes #########
    ### Request download for all files in collection ##
    def process_asset(url):
        retries = 0
        max_retries = 20

        while retries < max_retries:
            response = requests.get(url, auth=planet_api_key)

            if response.status_code == 429:
                retry_after = 5
                try:
                    if "retry-in" in response.text:
                        retry_after = float(response.text.split("retry-in")[1].strip().split()[0].replace("ms", "")) / 1000.0
                except Exception as e:
                    print(f"Failed to parse retry-in from 429 response: {e}")
                
                #print(f"Rate limited (429) - Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                retries += 1
                continue  # retry the request

            if not response.ok:
                print(f"Failed to fetch asset metadata from {url}. Status code: {response.status_code}, Response text: {response.text}")
                return None
            
            try:
                assets = response.json()
            except requests.JSONDecodeError as e:
                print(f"Failed to decode JSON from {url}. Response text: {response.text}")
                return None

            analytic_sr = assets.get("ortho_analytic_4b_sr")
            if not analytic_sr:
                print(f"No 'ortho_analytic_4b_sr' asset found in {url}")
                return None

            if "location" not in analytic_sr:
                # Activate the asset
                activate_url = analytic_sr["_links"]["activate"]
                activate_response = requests.get(activate_url, auth=planet_api_key)

                if not activate_response.ok:
                    print(f"Failed to activate asset for {url}")
                    return None

                # Poll until location appears
                self_url = analytic_sr["_links"]["_self"]
                max_activation_retries = 60
                retry_count = 0

                while retry_count < max_activation_retries:
                    checkstatus_response = requests.get(self_url, auth=planet_api_key)
                    checkstatus_assets = checkstatus_response.json()

                    if "location" in checkstatus_assets:
                        return checkstatus_assets["location"]

                    time.sleep(30)
                    retry_count += 1

                print(f"Asset {url} failed to become available after max retries.")
                return None

            else:
                return analytic_sr["location"]

        print(f"Asset {url} failed after {max_retries} retries.")
        return None

    def download_file(download_url, collection_id, folder_path):
        
        # Generate a short hash of the URL to make the filename unique
        url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
        
        filename = f"{folder_path}/psscene_{collection_id}_{url_hash}.tif"
        # print(f"Downloading {filename} into folder {folder_path}...")
        if not os.path.exists(filename):
            with requests.get(download_url, auth=planet_api_key, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filename, 'wb') as f, tqdm(
                    desc=f"Downloading {collection_id}_{url_hash}.tif",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        else:
            print(f"File {collection_id}_{url_hash}.tif already exists, skipping download.")


    def requestPlanetItemDownload(collection_gdf_file:str):
        """
        Request Planet item download for each item in the GeoDataFrame.
        
        args:
        collection_gdf_file: str, path to the GeoDataFrame file
        """

        collection=gpd.read_parquet(collection_gdf_file)
        collection_ids=collection.id.to_list()

        # get download urls
        download_urls=[]
        lock = threading.Lock()
        
        urls = pd.DataFrame(collection["_links"].to_list()).assets

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_asset, url): url for url in urls}

            for future in as_completed(futures):
                download_url = future.result()
                if download_url:
                    with lock:
                        download_urls.append(download_url)

        print(f"Collected {len(download_urls)} download URLs.")
        
        scene_date=collection.date_id.iloc[0]
        scene_date=scene_date.replace("-","")
        scene_folderpath=f"{folderpath}/psscene_{scene_date}"
        os.makedirs(scene_folderpath, exist_ok=True)
        
        # if already lebgth of files in scene_folderpath is equal to length of download_urls, skip download
        if len(os.listdir(scene_folderpath)) == len(download_urls):
            print(f"All files for {collection_gdf_file} already downloaded, skipping download.")
            return

        # download files
        for i, url in enumerate(download_urls):
            download_file(url, collection_ids[i], scene_folderpath)

        print(f"Downloaded {len(download_urls)} files for {collection_gdf_file}")
        return
    
    
    ######### request all date downloads #########
    # for filename in filenames:
    #     if not os.path.exists(filename):
    #         print(f"File {filename} does not exist, skipping download.")
    #         continue
        
        # Request download for each collection
        # requestPlanetItemDownload(filename)
    def process_filename_wrapper(filename):
        """Wrapper function for multiprocessing"""
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, skipping download.")
            return f"Skipped: {filename}"
        
        try:
            requestPlanetItemDownload(filename)
            return f"Completed: {filename}"
        except Exception as e:
            return f"Error processing {filename}: {e}"

        
    with ProcessPoolExecutor(max_workers=min(len(filenames), 4)) as executor:
        futures = {executor.submit(process_filename_wrapper, filename): filename for filename in filenames}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)
        
    print("Finished processing PlanetScope data at", time.strftime("%Y-%m-%d %H:%M:%S"))
    submit_job_with_dependency("./process_planetscope.sh", region=region, landsat_zarr_name=landsat_zarr_name, filenames=filenames, region_filenames_json=region_filenames_json)
    exit(0)

except Exception as e:
    print(f"An error occurred: {e}")
    exit_with_error(f"An error occurred: {e}")
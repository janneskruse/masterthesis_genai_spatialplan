"""
================================================================================
Module to request and download PlanetScope satellite imagery from the Planet API.
Provides functions for scene search, scene cover selection, asset activation,
and file download.
================================================================================
"""

##### Import libraries ######
# system
import os
import time
import calendar
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# downloading
import requests

# data manipulation
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import xarray as xr

# visualization
from tqdm import tqdm


###### get planet scenes for the bbox and time ranges ######
def request_planet_item_info(
    coordinates: list,
    planet_api_key: tuple,
    url: str,
    item_types: list = ["PSScene"],
    date: str | None = None,
    cloud_cover_limit: float | None = None,
    download: bool = False,
    further_filters: dict | None = None
) -> pd.DataFrame | None:
    """
    Request Planet item metadata for a given area and date.
    
    Args:
        coordinates: Polygon coordinates for the geometry filter.
        planet_api_key: Tuple of (api_key, "") for authentication.
        url: Planet API search URL.
        item_types: List of item types to search for.
        date: Date string in YYYY-MM-DD format.
        cloud_cover_limit: Maximum cloud cover fraction.
        download: Whether to add a download permission filter.
        further_filters: Additional filters to include.
        
    Returns:
        DataFrame of matching features, or None if no features found.
    """
    # Define filters
    filters = [
        {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": {
                "type": "Polygon",
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
        cloud_cover_filter = {
            "type": "RangeFilter",
            "config": {
                "gte": 0,
                "lte": cloud_cover_limit  #0.6
            },
            "field_name": "cloud_cover"
        }
        filters.append(cloud_cover_filter)
        
    if download:
        download_filter = {
            "type": "PermissionFilter",
            "config": [
                "assets:download"
            ]
        }
        filters.append(download_filter)
        
    if further_filters is not None:
        filters.append(further_filters)

    if date is not None:
        year = date.split("-")[0]
        month = int(date.split("-")[1])
        
                
        # define a date range of plus and minus 1 month
        start_month = str(month - 1 if month != 1 else 12).zfill(2)
        end_month = str(month + 1 if month != 12 else 1).zfill(2)
        end_day = calendar.monthrange(int(year), int(end_month))[1]
        
        local_start_year = year if month != 1 else str(int(year) - 1)
        local_end_year = year if month != 12 else str(int(year) + 1)

        start = f"{local_start_year}-{start_month}-01"
        end = f"{local_end_year}-{end_month}-{end_day}"
        
        date_range_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {
                "gte": f"{start}T00:00:00Z",
                "lte": f"{end}T00:00:00Z"
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
    response = requests.post(url, auth=planet_api_key, json=payload)
    response = response.json()
    if "features" not in response:
        #print(response)
        return None
    else:
        #print(response)
        features_df = pd.DataFrame(response["features"])
        
        return features_df


def search_planet_scenes_for_dates(
    time_ranges: list[str],
    coordinates: list,
    planet_api_key: tuple,
    url: str,
    cloud_cover_limit: float,
    max_workers: int = 8
) -> gpd.GeoDataFrame:
    """
    Search Planet API for scenes across multiple dates using multithreaded requests.
    
    Args:
        time_ranges: List of date strings in YYYY-MM-DD format.
        coordinates: Polygon coordinates for the geometry filter.
        planet_api_key: Tuple of (api_key, "") for authentication.
        url: Planet API search URL.
        cloud_cover_limit: Maximum cloud cover fraction.
        max_workers: Number of parallel workers.
        
    Returns:
        GeoDataFrame with scene metadata.
        
    Raises:
        ValueError: If no PlanetScope items are found.
    """
    # planet scope ("PSScene")
    item_types = ["PSScene"]

    # thread collect for all time ranges
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        planet_bydate_list = list(tqdm(
            executor.map(
                lambda date: request_planet_item_info(
                    coordinates=coordinates,
                    planet_api_key=planet_api_key,
                    url=url,
                    item_types=item_types,
                    date=date,
                    cloud_cover_limit=cloud_cover_limit
                ),
                time_ranges
            ),
            total=len(time_ranges)
        ))
        
    # merge all dataframes
    planet_bydate_list = [df for df in planet_bydate_list if df is not None]
    if not planet_bydate_list:
        raise ValueError("No PlanetScope items found")
    else:
        planet_bydate_df = pd.concat(planet_bydate_list, ignore_index=True)
        planet_bydate_df.head(2)

    planet_bydate_gdf = gpd.GeoDataFrame(
        planet_bydate_df,
        geometry=[shape(geom) for geom in planet_bydate_df["geometry"]],
        crs="EPSG:4326"
    )
    
    return planet_bydate_gdf


def merge_nearest_rows(
    df: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    max_distance: float = 0.01
) -> gpd.GeoDataFrame:
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
        df = df.iloc[1:]  # exclude the first row
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


def get_planetscope_scenes_cover_for_date(
    time_id: int,
    landsat_xr_ds: xr.Dataset,
    planet_bydate_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Get the PlanetScope scenes that cover the bbox for a specific Landsat time index.
    
    Args:
        time_id: Index into the landsat xarray time dimension.
        landsat_xr_ds: Landsat xarray dataset with time coordinates.
        planet_bydate_gdf: GeoDataFrame with all Planet scene metadata.
        bbox_gdf: GeoDataFrame with the bounding box geometry (in UTM CRS).
        
    Returns:
        GeoDataFrame with the selected scenes that cover the bbox.
    """
    # Get time values from xarray dataset
    time_stamp = landsat_xr_ds.isel(time=time_id).time.values
    time_stamp = pd.to_datetime(time_stamp).strftime("%Y-%m-%d")
    #time_stamp_flat=time_stamp.replace("-", "")
    time_stamp_flat_month = time_stamp.replace("-", "")[:-2]
    month = int(time_stamp_flat_month[-2:])
    previous_month = str(month - 1 if month != 1 else 12).zfill(2)
    previous_month_year = time_stamp_flat_month[:-2] if month != 1 else str(int(time_stamp_flat_month[:-2]) - 1)
    previous_month_time_stamp_flat = f"{previous_month_year}{previous_month}"
    next_month = str(month + 1 if month != 12 else 1).zfill(2)
    next_month_year = time_stamp_flat_month[:-2] if month != 12 else str(int(time_stamp_flat_month[:-2]) + 1)
    next_month_time_stamp_flat = f"{next_month_year}{next_month}"

    # filter planet_bydate_gdf by id of time_stamp_flat_month, previous_month_time_stamp_flat, next_month_time_stamp_flat
    planet_bydate_gdf_filtered = planet_bydate_gdf[
        planet_bydate_gdf['id'].str.contains(time_stamp_flat_month) | 
        planet_bydate_gdf['id'].str.contains(previous_month_time_stamp_flat) | 
        planet_bydate_gdf['id'].str.contains(next_month_time_stamp_flat)
    ].copy()

    # create date id
    planet_bydate_gdf_filtered['date_id'] = planet_bydate_gdf_filtered['id'].str[0:8]
    
    planet_bydate_gdf_filtered_clipped = planet_bydate_gdf_filtered.clip(bbox_gdf)
    
    # get the nearest ids for time_stamp_flat
    planet_bydate_gdf_filtered_clipped.loc[:, 'date_id_dt'] = pd.to_datetime(planet_bydate_gdf_filtered_clipped['date_id'])
    planet_bydate_gdf_filtered_clipped.loc[:, 'time_stamp_dt'] = pd.to_datetime(time_stamp)
    planet_bydate_gdf_filtered_clipped.loc[:, 'time_diff'] = (planet_bydate_gdf_filtered_clipped['date_id_dt'] - planet_bydate_gdf_filtered_clipped['time_stamp_dt']).dt.days
    planet_bydate_gdf_filtered_clipped.loc[:, 'time_diff'] = planet_bydate_gdf_filtered_clipped['time_diff'].abs()

    # sort the dataframe by diff
    planet_bydate_gdf_filtered_clipped.sort_values('time_diff', inplace=True)
    
    planet_scenes_cover_df = merge_nearest_rows(planet_bydate_gdf_filtered_clipped, bbox_gdf)
    
    return planet_scenes_cover_df


######### Request download for all scenes #########
### Request download for all files in collection ##
def _process_asset(
    url: str,
    planet_api_key: tuple,
    asset: str = "ortho_analytic_4b_sr"
) -> str | None:
    """
    Activate and get the download URL for a Planet asset.
    
    Args:
        url: URL to the asset metadata endpoint.
        planet_api_key: Tuple of (api_key, "") for authentication.
        asset: Asset name to request.
        
    Returns:
        Download URL string, or None if activation failed.
    """
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

        asset_data = assets.get(asset)
        if not asset_data:
            print(f"No '{asset}' asset found in {url}")
            return None

        if "location" not in asset_data:
            # Activate the asset
            activate_url = asset_data["_links"]["activate"]
            activate_response = requests.get(activate_url, auth=planet_api_key)

            if not activate_response.ok:
                print(f"Failed to activate asset for {url}")
                return None

            # Poll until location appears
            self_url = asset_data["_links"]["_self"]
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
            return asset_data["location"]

    print(f"Asset {url} failed after {max_retries} retries.")
    return None


def _download_file(
    download_url: str,
    collection_id: str,
    folder_path: str,
    id: str,
    planet_api_key: tuple
) -> bool:
    """
    Download a single PlanetScope file from a download URL.
    
    Args:
        download_url: URL to download the file from.
        collection_id: Collection identifier for naming.
        folder_path: Path to save the downloaded file.
        id: Unique identifier suffix for the filename.
        planet_api_key: Tuple of (api_key, "") for authentication.
        
    Returns:
        True if file was downloaded, False if it already existed.
    """
    # # Generate a short hash of the URL to make the filename unique
    url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
    
    # filename = f"{folder_path}/psscene_{collection_id}_{url_hash}.tif"
    filename = f"{folder_path}/psscene_{collection_id}_{id}.tif"
    
    # print(f"Downloading {filename} into folder {folder_path}...")
    if not os.path.exists(filename):
        print(f"Downloading {filename} into folder {folder_path}...")
        with requests.get(download_url, auth=planet_api_key, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=f"Downloading {collection_id}_{url_hash} to {filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            
        # print(f"Downloaded {filename} successfully.")
        return True
    else:
        print(f"File {filename} already exists, skipping download.")
        return False


def request_planet_item_download(
    collection_gdf_file: str,
    folderpath: str,
    planet_api_key: tuple,
    asset_names: list[str] = ["ortho_analytic_4b_sr", "ortho_udm2"]
):
    """
    Request Planet item download for each item in the GeoDataFrame.
    
    Args:
        collection_gdf_file: str, path to the GeoDataFrame parquet file.
        folderpath: str, base folder path for downloads.
        planet_api_key: Tuple of (api_key, "") for authentication.
        asset_names: List of asset names to download.
    """
    collection = gpd.read_parquet(collection_gdf_file)
    collection_ids = collection.id.to_list()

    scene_date = collection.date_id.iloc[0]
    scene_date = scene_date.replace("-", "")
    scene_folderpath = f"{folderpath}/psscene_{scene_date}"
    os.makedirs(scene_folderpath, exist_ok=True)
    
    # rename download_url to download_url_ortho_analytic_4b_sr for backwards compatibility
    collection = collection.rename(columns={"download_url": "download_url_ortho_analytic_4b_sr"})

    # check if download_url column exists
    for asset_name in asset_names:
        if not f'download_url_{asset_name}' in collection.columns or collection[f'download_url_{asset_name}'].isnull().any():

            # get download urls
            download_urls = []
            
            # check the rows with missing download urls
            if f'download_url_{asset_name}' in collection.columns:
                missing_rows = collection[collection[f'download_url_{asset_name}'].isnull()]
                # update download urls with existing ones
                download_urls = collection.loc[collection[f'download_url_{asset_name}'].notnull(), f'download_url_{asset_name}'].to_list()
            else:
                missing_rows = collection

            lock = threading.Lock()
            
            urls = pd.DataFrame(missing_rows["_links"].to_list()).assets

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(_process_asset, url, planet_api_key, asset_name): url 
                    for url in urls
                }

                for future in as_completed(futures):
                    try:
                        download_url = future.result()
                    except Exception as e:
                        print(f"Asset worker raised an exception for {futures[future]}: {e}")
                        continue

                    if download_url:
                        with lock:
                            download_urls.append(download_url)
                            
            # check if doubled download urls
            if len(set(download_urls)) < len(download_urls):
                print(f"Warning: Duplicate download URLs found for {collection_gdf_file}.")
                # drop duplicates
                download_urls = list(set(download_urls))
            
            # save to parquet
            collection[f'download_url_{asset_name}'] = pd.Series(download_urls)
            collection.to_parquet(collection_gdf_file)
        else:
            print(f"Download URLs for {collection_gdf_file} already exist, skipping activation request.")
            download_urls = collection[f'download_url_{asset_name}'].to_list()

        print(f"Collected {len(download_urls)} download URLs.")

        # download files
        downloaded_files = 0
        for i, url in enumerate(download_urls):
            if _download_file(url, collection_ids[i], scene_folderpath, f"{asset_name}_{i}", planet_api_key):
                downloaded_files += 1

        print(f"Downloaded {downloaded_files} files for {collection_gdf_file}")

    print(f"Completed downloads for {collection_gdf_file}")
    return


def download_all_collections(
    filenames: list[str],
    folderpath: str,
    planet_api_key: tuple,
    asset_names: list[str],
    max_scenes_per_region: int = 1,
    max_workers: int = 4
):
    """
    Download all PlanetScope collections using multiprocessing.
    
    Args:
        filenames: List of parquet file paths for each collection.
        folderpath: Base folder path for downloads.
        planet_api_key: Tuple of (api_key, "") for authentication.
        asset_names: List of asset names to download.
        max_scenes_per_region: Maximum number of scenes to process.
        max_workers: Maximum number of parallel workers.
    """
    ######### request all date downloads #########
    def process_filename_wrapper(filename):
        """Wrapper function for multiprocessing"""
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, skipping download.")
            return f"Skipped: {filename}"
        
        try:
            request_planet_item_download(filename, folderpath, planet_api_key, asset_names=asset_names)
            return f"Completed: {filename}"
        except Exception as e:
            return f"Error processing {filename}: {e}"

    with ProcessPoolExecutor(max_workers=min(len(filenames), max_workers)) as executor:
        futures = {
            executor.submit(process_filename_wrapper, filename): filename 
            for filename in filenames[0:max_scenes_per_region]
        }
        
        for future in as_completed(futures):
            result = future.result()
            print(result)

# # Script to aquire and pre-process Landsat LST data for high temperaturre days

####### Import libraries #######
# system
import os
import zipfile
import time
import calendar
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from dotenv import load_dotenv

# downloading and website scraping
import requests
from bs4 import BeautifulSoup

# aws bucket access
import boto3

# data manipulation
import yaml
import json
from thefuzz import fuzz
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio # needed to load for xarray conversions
import xarray as xr
import rioxarray as rxr
from shapely.geometry import box

# visualization
from tqdm import tqdm

p=os.popen('git rev-parse --show-toplevel')
repo_dir = p.read().strip()
p.close()

# Load .env file
load_dotenv(dotenv_path=f"{repo_dir}/.env")

with open(f"{repo_dir}/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)
    
big_data_storage_path = config.get("big_data_storage_path", "work/zt75vipu-master/data")

# get aws credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")


def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)


####### Get the region to process #######
try:
    if "REGION_NAME" in os.environ:
        region = os.environ["REGION_NAME"] 
    else:
        exit_with_error("Region not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print("Error getting region from environment:", e)
    exit_with_error("Region not set in environment, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))
    

######## Try except Landsat data processing ########
try:
    ############ Define the bbox ############ 
    # load from GSHL regions file
    # region = config.get("regions", ["Leipzig"])[0]
    ghsl_df_new = gpd.read_parquet(f"{repo_dir}/data/processed/ghsl_regions.parquet")
    bbox_gdf = gpd.GeoDataFrame(geometry=ghsl_df_new[ghsl_df_new["region_name"]==region].bbox, crs="EPSG:4326")
    bbox_polygon=json.loads(bbox_gdf.to_json())['features'][0]['geometry']

    ####### Get/Define the config parameters ########
    landsat_region_folder = f"{big_data_storage_path}/landsat/{region.lower()}"
    os.makedirs(landsat_region_folder, exist_ok=True)
    
    end_year = config["temperature_day_filter"]["years"].get("end", 2023)
    start_year = config["temperature_day_filter"]["years"].get("start", 1950)
    
    min_temperature = config["temperature_day_filter"]["min"]
    consecutive_days = config["temperature_day_filter"]["consecutive_days"]
    
    max_cloud_cover = config["landsat_query"].get("max_cloud_coverage", 10)
    collections = config["landsat_query"].get("collections", ["landsat-c2l2-st"])

    landsat_zarr_name = f"{landsat_region_folder}/landsat_temperature_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"
    stac_filename = f"{landsat_region_folder}/stac_query_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.parquet"
    
    if not os.path.exists(landsat_zarr_name):
        print(f"Creating Landsat zarr dataset at {landsat_zarr_name} at", time.strftime("%Y-%m-%d %H:%M:%S"))
    
        if not os.path.exists(stac_filename):
            ########### Get consecutive high temperatures from DWD #########
            # DWD data: 
            # - an overview on all german stations is available here: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/KL_Tageswerte_Beschreibung_Stationen.txt
            # - the historic daily data for each station is available for download here: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/
            # - for a quick search, this page displays the table interactively: https://www.dwd.de/DE/leistungen/klimadatendeutschland/klimadatendeutschland.html 

            colnames = [
                "Stations_id", "von_datum", "bis_datum", "Stationshoehe",
                "geoBreite", "geoLaenge", "Stationsname", "Bundesland", "Abgabe"
            ]

            stations_df = pd.read_fwf(
                f"{repo_dir}/data/dwd/KL_Tageswerte_Beschreibung_Stationen.txt",
                skiprows=2,
                encoding="latin1",
                names=colnames,
                dtype={"Stations_id": str} 
            )


            # Get all stations for the region
            stations = []
            for index, row in stations_df.iterrows():
                # print("Fuzzy matching station:", row["Stationsname"], fuzz.ratio(row["Stationsname"], region))
                if fuzz.ratio(row["Stationsname"], region) > 70 or region in row["Stationsname"]:
                    stations.append({
                        'name': row["Stationsname"],
                        'lat': row["geoBreite"],
                        'lon': row["geoLaenge"],
                        'date_start': pd.to_datetime(row['von_datum'], format='%Y%m%d'),
                        'date_end': pd.to_datetime(row['bis_datum'], format='%Y%m%d'),
                        'station_id': row["Stations_id"]
                    })


            stations_gpd = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy([station['lon'] for station in stations], [station['lat'] for station in stations]), crs="EPSG:4326")

            ######## Download the temperature data for the most urban and recent station for the region #########
            # Leipzig-Mockau has data only until the 70s. Leipzig-Halle has data for the airport, which is a different environment. We, therefore, choose Leipzig-Holzhausen here to get data for a more urban environment.
            # The following does this selection and download programmatically to automize for other cities
            # Get station with data in the configured year range
            try:
                stations_gpd = stations_gpd[(stations_gpd['date_end'].dt.year >= end_year) & (stations_gpd['date_start'].dt.year <= start_year)]
            except KeyError:
                print("No stations found for the given year range. Please check your configuration.")
                exit_with_error("No stations found for the given year range, finishing at", time.strftime("%Y-%m-%d %H:%M:%S"))

            # Get the center of the bbox and find the closest (most urban) station
            bbox_center = bbox_gdf.geometry.centroid.iloc[0]
            stations_gpd['distance'] = stations_gpd.geometry.distance(bbox_center)
            stations_gpd = stations_gpd.sort_values(by='distance').reset_index(drop=True)
            stations_gpd = stations_gpd.head(1)  # keep only the closest station

            # download the DWD data for the station (Leipzig-Holzhausen)
            #Construct the URL for the daily DWD data
            base_url= "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical"
            station = stations_gpd.iloc[0]

            # Download the zip and extract
            foldername = f"{repo_dir}/data/dwd/{station['station_id']}_data"
            if not os.path.exists(foldername):
                print(f"Requesting zip urls for {station['name']}...")
                
                #get all zip paths
                response = requests.get(base_url)
                soup = BeautifulSoup(response.text, "html.parser")

                # extract links to .zip files
                zip_files = [
                    f"{base_url}/{a['href']}"
                    for a in soup.find_all("a")
                    if a["href"].endswith(".zip")
                ]

                # get the station specific url
                station_url = next(
                    (url for url in zip_files if station['station_id'] in url), 
                    None
                )
                print(f"Found station URL: {station_url}")
                
                #download the zip file
                print(f"Downloading data for {station['name']} from {station_url}")
                response = requests.get(station_url)
                
                if response.status_code == 200:
                    # save zip
                    with open(f"{foldername}.zip", 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded data for {station['name']} from {station_url}")
                    
                    # extract zip
                    with zipfile.ZipFile(f"{foldername}.zip", 'r') as zip_ref:
                        zip_ref.extractall(foldername)
                    print(f"Extracted data for {station['name']} to {repo_dir}/data/dwd/")
                    
                    # remove zip
                    os.remove(f"{foldername}.zip")
                    print(f"Removed zip file {foldername}.zip")
                else:
                    print(f"Failed to download data for {station['name']} from {station_url}. Status code: {response.status_code}")
            else:
                print(f"Data for {station['name']} is already downloaded at {foldername}")

            ######## Read the temperature data for the station #######
            # Metadata for the column names:
            # https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/standard_format/formate_kx.html

            # Read the data
            files = os.listdir(foldername)
            kl_file = [f for f in files if f.startswith("produkt_klima_tag") and f.endswith(".txt")]

            if not kl_file:
                print(f"No climate data file found in {foldername}.")

            station_kl=pd.read_csv(f"{foldername}/{kl_file[0]}", sep=";")

            #trim column names
            station_kl.columns = [col.strip() for col in station_kl.columns]

            # date to datetime64
            station_kl['MESS_DATUM'] = pd.to_datetime(station_kl['MESS_DATUM'], format="%Y%m%d")

            #extract TXK temperature column
            station_temp_max = station_kl[['MESS_DATUM','TXK']]

            #replace -999.0 with NaN
            station_temp_max.loc[:, 'TXK']  = station_temp_max['TXK'].replace(-999.0, np.nan)

            print(f"Number of missing values in maximum temperature: {station_temp_max['TXK'].isna().sum()}")
            print("Head of maximum temperature data:")
            station_temp_max.head(3)
            
            
            ######## Get consecutive days with high temperatures #######
            # summed days with max temperature >= min_temperature for rolling window of 3 days
            station_temp_max.loc[:, 'gt_roll']=station_temp_max['TXK'].ge(min_temperature).rolling(window=consecutive_days).sum()

            # get only the days where the rolling window is equal to the number of consecutive days
            station_temp_max_gt=station_temp_max[station_temp_max['gt_roll'] == consecutive_days].copy()

        
            ################## Get LST Data from Landsat #################
            # How to query the stac api: https://code.usgs.gov/eros-user-services/quick-guides/querying-the-stac-api-with-geojson-objects/-/blob/main/querying_with_geojson_objects_v3.ipynb?ref_type=heads
            # Further information on the post requests can be found on the node api documentation:
            # https://github.com/stac-utils/stac-server
            
            ########## Define the query functions ##########
            # Function to query the stac server for features with boundary geolocation
            def fetch_stac_server(query):
                '''
                Queries the stac-server (STAC) backend.
                query is a python dictionary to pass as json to the request.
                '''
                
                search_url = f"https://landsatlook.usgs.gov/stac-server/search"
                query_return = requests.post(search_url, json=query).json()
                error = query_return.get("message", "")
                if error:
                    raise Exception(f"STAC-Server failed and returned: {error}")
                    
                if 'code' in query_return: # if query fails, return failure code
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
                        query_gdf['stac_id']= [feature["id"] for feature in features]

                        return query_gdf
                    else:
                        #print("No features found")
                        return None

            # Function to send a a filtered request to the stac server using the function above:
            def send_STAC_query(limit=200, collections='landsat-c2l2-sr', intersects=None, year:str=None, month:str=None, date_list:list[str]=None, max_cloud_cover=None):
                '''
                This function helps to create a simple parameter dictionary for querying 
                the Landsat Collection 2 Level 2 Surface Reflectance feature in the STAC Server.
                It prints the parameter dictionary and returns the query results.
                
                args:
                limit: int, default 200, number of items to return
                collections: str, default 'landsat-c2l2-sr', collection to query
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
                    
                #filter by date
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
                        return gpd.pd.concat(all_results, ignore_index=True)
                    else:
                        return None
                    
                else:
                    max_day = 31
                    
                    if year is not None:
                        params['datetime'] = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
                    if month is not None:
                        #set last day for month
                        max_day=calendar.monthrange(int(year), int(month))[1]
                        
                        params['datetime'] = f"1970-{month}-01T00:00:00Z/2024-{month}-{max_day}T23:59:59Z"
                    if year is not None and month is not None:        
                        params['datetime'] = f"{year}-{month}-01T00:00:00Z/{year}-{month}-{max_day}T23:59:59Z"
                    
                    print(params) 
                    
                    return fetch_stac_server(params)

            
            ####### Query for the high temperature days #########
            # Get the defined years of consecutive high temperatures for day 2 and 3 in a compatible date format
            station_temp_max_gt=station_temp_max_gt[station_temp_max_gt.MESS_DATUM.dt.year>=start_year].copy()
            station_temp_max_gt=station_temp_max_gt[station_temp_max_gt.MESS_DATUM.dt.year<=end_year].copy()
            station_temp_max_gt_dates=station_temp_max_gt.MESS_DATUM.to_list()

            # get the day before each third day as well
            station_temp_max_gt_dates_before=[date - pd.DateOffset(days=1) for date in station_temp_max_gt_dates]

            # merge the two lists
            station_temp_max_gt_dates.extend(station_temp_max_gt_dates_before)

            # sort the list
            station_temp_max_gt_dates.sort()

            # remove duplicates
            station_temp_max_gt_dates=list(dict.fromkeys(station_temp_max_gt_dates))

            # get dates in format YYYY-MM-DD
            station_temp_max_gt_dates=[date.strftime("%Y-%m-%d") for date in station_temp_max_gt_dates]

            # Query for the dates using multithreaded requests
            chunk_size=20
            chunks=[station_temp_max_gt_dates[i:i + chunk_size] for i in range(0, len(station_temp_max_gt_dates), chunk_size)]

            query_gdf = gpd.pd.DataFrame()

            # multithread requests using tqdm progress bar
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(send_STAC_query, intersects=bbox_polygon, limit=1, date_list=chunk, collections=collections, max_cloud_cover=max_cloud_cover) for chunk in chunks]
                
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        if len(result) > 0:
                            query_gdf = gpd.pd.concat([query_gdf, result], ignore_index=True)

            # save as geoparquet
            print(f"Saving STAC query results to {stac_filename}")
            query_gdf.to_parquet(stac_filename)
            print(f"Saved STAC query results to {stac_filename}")
        else:
            print(f"STAC query results already exist at {stac_filename}")

        query_gdf=gpd.read_parquet(stac_filename)

        ####### Get the images for the requested collection information #######
        # For information on the asset links:
        # https://landsat.usgs.gov/stac/LC09_L2SP_095022_20220625_20220627_02_T1_ST_stac.json
        def getLandsatTemperatureProducts(query_gdf):
            '''
            This function retrieves the Landsat 8 Surface Temperature products from the query results.
            It returns a list of the product urls.
            '''
            products = []
            for index, row in query_gdf.iterrows():
                assets = row.assets
                if 'lwir11' in assets and assets['lwir11'] is not None:
                    products.append({"stac_id": row.stac_id, "datetime": row.datetime,
                                    "thermal": {
                                    "url": assets['lwir11']['href'],
                                    "alternate": assets['lwir11']['alternate']},
                                    "qa_pixel":{"url": assets['qa_pixel']['href'], "alternate": assets['qa_pixel']['alternate']}
                                    })
                                    
                elif 'lwir' in assets and assets['lwir'] is not None:
                    print(f"found only B6 from Landsat 7 for {row.stac_id}")
                    # products.append({"stac_id": row.stac_id, "datetime": row.datetime, 
                                    #  "url": assets['lwir']['href'], "alternate": assets['lwir']['alternate']})
                    
                else:
                    print(f"No lwir11 asset for {row.stac_id}")

            return products

        products=getLandsatTemperatureProducts(query_gdf)
        pd.DataFrame(products[0])

        # %% [markdown]
        # ### Download the images using AWS CLI

        # %%
        # Setup tqdm lock to prevent corruption of output
        tqdm.set_lock(RLock())

        # Setup the boto3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name='us-west-2'
        )

        def parse_s3_url(s3_url):
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

        def download_tif(s3_url, local_path, position=0):
            bucket, key = parse_s3_url(s3_url)

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


        def requestProduct(product, region, position=0):
            '''
            This function retrieves the products from the USGS server and saves it to the output path.
            It stores both the thermal and the qa_pixel tif files.
            '''
            
            landsat_region_folder = f"{big_data_storage_path}/landsat/{region.lower()}"
            os.makedirs(f"{landsat_region_folder}/landsat_temperature", exist_ok=True)
            output_path_base = f"{landsat_region_folder}/landsat_temperature/{product['stac_id']}"
            output_path_thermal = f"{output_path_base}_thermal.tif"
            output_path_qa_pixel = f"{output_path_base}_qa_pixel.tif"
            
            if not os.path.exists(output_path_thermal):
                os.makedirs(os.path.dirname(output_path_thermal), exist_ok=True)
                s3_url = product["thermal"]['alternate']['s3']['href']
                download_tif(s3_url, output_path_thermal, position)

            if not os.path.exists(output_path_qa_pixel):
                os.makedirs(os.path.dirname(output_path_qa_pixel), exist_ok=True)
                s3_url = product["qa_pixel"]['alternate']['s3']['href']
                download_tif(s3_url, output_path_qa_pixel, position + 1)

            return

        requestProduct(products[0], region)

        # Thread download all the datasets
        def process_all_products(products, max_workers=8):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(requestProduct, product, region, i*2): product
                    for i, product in enumerate(products, start=1)
                }
        process_all_products(products)

        ######### Create a pre-processed/cleaned zarr for the Landsat data #########
        #read files
        files=os.listdir(f"{landsat_region_folder}/landsat_temperature")
        files=[file for file in files if file.startswith("LC08")]
        file_paths=[f"{landsat_region_folder}/landsat_temperature/{file}" for file in files]
        thermal_file_paths=[file for file in file_paths if file.endswith("thermal.tif")]
        qa_pixel_file_paths=[file for file in file_paths if file.endswith("qa_pixel.tif")]

        # find qa pixel files for thermal files and create set of file paths
        file_sets=[]
        for thermal_file in thermal_file_paths:
            qa_pixel_file=thermal_file.replace("thermal", "qa_pixel")
            if qa_pixel_file in qa_pixel_file_paths:
                file_sets.append([qa_pixel_file, thermal_file])

        # find file in query_gdf
        def find_file(stac_id):
            file=query_gdf[query_gdf['stac_id']==stac_id]
            if len(file) > 0:
                return file.iloc[0]
            else:
                return None

        # read file function with metadata attached
        def readLandsatTifsToXarrayDS(ds_file_path_set):
            """
            Read Landsat tif files to xarray dataset and attach metadata.
            
            args:
            file_path: str, path to the tif file    
            """
            
            thermal_file_path=ds_file_path_set[1]
            qa_pixel_file_path=ds_file_path_set[0]
            
            stac_id=thermal_file_path.split("/")[-1].split(".")[0].split("_thermal")[0]
            file=find_file(stac_id)
            
            if file is None:
                print(f"File not found in query_gdf: {stac_id}")
                return None
            
            if not os.path.exists(thermal_file_path):
                print(f"File not found: {thermal_file_path}")
                return None
            if not os.path.exists(qa_pixel_file_path):
                print(f"File not found: {qa_pixel_file_path}")
                return None
            
            #####create thermal data array######
            #read file
            xda=rxr.open_rasterio(thermal_file_path, masked=True)
            xda = xda.rio.reproject("EPSG:4326")
            
            #clip to bbox
            xda = xda.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)
            
            # apply scaling factor to degrees celsius
            scale_factor=0.00341802
            add_offset=149.0-273.15
            #xda=xda*scale_factor+add_offset
            xda.attrs['scale_factor']=scale_factor
            xda.attrs['add_offset']=add_offset
            
            #rename data array
            xda=xda.rename('surface_temp_b10')
            
            
            #####create qa pixel data array#######
            #read file
            xda_qa=rxr.open_rasterio(qa_pixel_file_path)
            
            # fill nodata values with 0
            xda_qa = xda_qa.rio.write_nodata(0, inplace=True)
            xda_qa = xda_qa.rio.reproject("EPSG:4326")
            
            #clip to bbox
            xda_qa = xda_qa.rio.clip([bbox_gdf.geometry.iloc[0]], bbox_gdf.crs)
            
            #mask cloud and cloud shadow
            # Define bit positions for cloud and cloud shadow
            CLOUD_SHADOW_BIT = 3  # Bit 3 = cloud shadow
            CLOUD_BIT = 5         # Bit 5 = cloud

            # Create masks
            cloud_shadow_mask = (xda_qa & (1 << CLOUD_SHADOW_BIT)) == 0  # True = no shadow
            cloud_mask = (xda_qa & (1 << CLOUD_BIT)) == 0                 # True = no cloud

            clear_mask = cloud_shadow_mask & cloud_mask
            xda_qa = clear_mask.astype(np.uint8)
            
            #rename data array
            xda_qa=xda_qa.rename('qa_pixel')
            
            
            #####create masked array #####
            #create masked array
            xda_mask=xda.where(xda_qa)
            
            #rename data array
            xda_mask=xda_mask.rename('surface_temp_b10_masked')
            
            
            #####combine data arrays####
            xds=xr.merge([xda, xda_qa, xda_mask])
            
            
            #####add metadata#####
            # get stac data
            date=file.datetime
            
            #add general metadata
            xds.attrs['title']="Landsat 8 Surface Temperature"
            xds.attrs['description']="Landsat 8 Surface Temperature data from USGS for specific hot days (3 continous >30C° days) in Leipzig"
            xds.attrs['source']="USGS"
            xds.attrs['crs']="EPSG:4326"
            xds.attrs['bbox']=bbox_gdf.to_json()
            xds.attrs['variables']={"surface_temp_b10": "Surface Temperature Band (B10)",
                                    "qa_pixel": "Quality Assessment Pixel",
                                    "surface_temp_b10_masked": "Surface Temperature Band (B10) Masked"}
            xds.attrs['units']={"surface_temp_b10": "°C", "qa_pixel": "1", "surface_temp_b10_masked": "°C"}
            
            #remove scale_factor and add_offset from attrs
            xds.attrs.pop('scale_factor', None)
            xds.attrs.pop('add_offset', None)
            
            #remove spatial_ref and band coords
            xds=xds.drop_vars(["spatial_ref", "band"])
            
            #squeeze band from variables
            xds=xds.squeeze("band", drop=True)
            
            # add time coordinate
            xds=xds.expand_dims(time=[date])
            # xds=xds.expand_dims(stac_id=[stac_id])
            # xds=xds.expand_dims(view_sun_elevation=[file['view:sun_elevation']])
            # xds=xds.expand_dims(view_sun_azimuth=[file['view:sun_azimuth']])
            # xds=xds.expand_dims(view_off_nadir=[file['view:off_nadir']])

            #add metadata as variables over time
            xds['stac_id']=xr.DataArray([stac_id], dims=['time'])
            xds['view_sun_elevation']=xr.DataArray([file['view:sun_elevation']], dims=['time'])
            xds['view_sun_azimuth']=xr.DataArray([file['view:sun_azimuth']], dims=['time'])
            xds['view_off_nadir']=xr.DataArray([file['view:off_nadir']], dims=['time'])

            return xds

        # read all files
        xds_list = [
            ds for file_set in file_sets
            if (ds := readLandsatTifsToXarrayDS(file_set)) is not None
        ]

        ###### reindex to common grid ######
        # Therefore, the coordinates should be be reindexed to a template grid before concatenating the data, so the data aligns.

        common_x = xds_list[0].x
        common_y = xds_list[0].y

        xds_list = [
            ds.reindex(x=common_x, y=common_y, method="nearest")  # or method="pad"
            for ds in xds_list
        ]

        #combine datasets
        landsat_xr_ds=xr.concat(xds_list, dim='time')
        
        # ensure time coordinate is in timezone-naive datetime64[ns] format 
        landsat_xr_ds = landsat_xr_ds.assign_coords(
            time=pd.to_datetime(landsat_xr_ds.time.values).tz_localize(None)
        )

        ###### filter data a second time ######
        # Somehow the stac filtering did not work on all files, so a second filter is applied here to remove all files where the qa_pixel is not null more than the allowed configuration percentage.
        # filter out all timesteps where no data values are present
        mask = landsat_xr_ds.surface_temp_b10_masked.notnull().compute()
        landsat_xr_ds=landsat_xr_ds.where(mask, drop=True)

        #filter where qa_pixel not more than max_cloud_cover percentage
        valid_pixel_percentage = landsat_xr_ds.qa_pixel.notnull().mean(dim=['x', 'y']).compute()
        landsat_xr_ds = landsat_xr_ds.where(valid_pixel_percentage >= (100-max_cloud_cover)/100, drop=True)

        try:
            max_dates_per_year = config["temperature_day_filter"]["max_dates_per_year"]

            if max_dates_per_year:
                # group by year and take the first max_dates_per_year dates
                landsat_xr_ds = landsat_xr_ds.groupby('time.year').apply(
                    lambda x: x.isel(time=np.arange(min(max_dates_per_year, len(x.time))))
                )
        except:
            print("No max_dates_per_year configured, skipping this step.")

        #save as zarr dataset
        landsat_xr_ds.to_zarr(landsat_zarr_name, mode='w')

        print(f"Saved Landsat dataset to {landsat_zarr_name} at", time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print(f"Landsat zarr dataset already exists at {landsat_zarr_name}, skipping creation at", time.strftime("%Y-%m-%d %H:%M:%S"))

except Exception as e:
    print(f"An error occurred: {e}")
    exit_with_error(f"An error occurred: {e}")


"""
================================================================================
Module to acquire and pre-process DWD temperature station data.
Provides functions for station lookup, data download, temperature reading,
and filtering for consecutive high-temperature days.
================================================================================
"""

##### Import libraries ######
# system
import os
import zipfile

# downloading and website scraping
import requests
from bs4 import BeautifulSoup

# data manipulation
from thefuzz import fuzz
import numpy as np
import pandas as pd
import geopandas as gpd


########### Get consecutive high temperatures from DWD #########
# DWD data: 
# - an overview on all german stations is available here: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/KL_Tageswerte_Beschreibung_Stationen.txt
# - the historic daily data for each station is available for download here: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/
# - for a quick search, this page displays the table interactively: https://www.dwd.de/DE/leistungen/klimadatendeutschland/klimadatendeutschland.html 

def load_dwd_stations(repo_dir: str) -> pd.DataFrame:
    """
    Load the DWD station overview file.
    
    Args:
        repo_dir: Path to the repository root directory.
        
    Returns:
        DataFrame with station metadata.
    """
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
    
    return stations_df


def find_stations_for_region(stations_df: pd.DataFrame, region: str) -> gpd.GeoDataFrame:
    """
    Get all DWD stations matching a region name via fuzzy matching.
    
    Args:
        stations_df: DataFrame of station metadata from load_dwd_stations.
        region: Region name to match against station names.
        
    Returns:
        GeoDataFrame of matching stations.
    """
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

    stations_gpd = gpd.GeoDataFrame(
        stations, 
        geometry=gpd.points_from_xy(
            [station['lon'] for station in stations], 
            [station['lat'] for station in stations]
        ), 
        crs="EPSG:4326"
    )
    
    return stations_gpd


def select_closest_station(
    stations_gpd: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    start_year: int,
    end_year: int
) -> gpd.GeoDataFrame:
    """
    Download the temperature data for the most urban and recent station for the region.
    
    Leipzig-Mockau has data only until the 70s. Leipzig-Halle has data for the airport,
    which is a different environment. We, therefore, choose Leipzig-Holzhausen here to
    get data for a more urban environment.
    The following does this selection programmatically to automate for other cities.
    
    Args:
        stations_gpd: GeoDataFrame of candidate stations.
        bbox_gdf: GeoDataFrame with the bounding box geometry.
        start_year: Start year for station data availability.
        end_year: End year for station data availability.
        
    Returns:
        GeoDataFrame with the single closest station.
        
    Raises:
        ValueError: If no stations are found for the given year range.
    """
    # Get station with data in the configured year range
    try:
        stations_gpd = stations_gpd[
            (stations_gpd['date_end'].dt.year >= end_year) & 
            (stations_gpd['date_start'].dt.year <= start_year)
        ]
    except KeyError:
        raise ValueError("No stations found for the given year range. Please check your configuration.")

    if len(stations_gpd) == 0:
        raise ValueError("No stations found for the given year range. Please check your configuration.")

    # Get the center of the bbox and find the closest (most urban) station
    bbox_center = bbox_gdf.geometry.centroid.iloc[0]
    stations_gpd['distance'] = stations_gpd.geometry.distance(bbox_center)
    stations_gpd = stations_gpd.sort_values(by='distance').reset_index(drop=True)
    stations_gpd = stations_gpd.head(1)  # keep only the closest station
    
    return stations_gpd


def download_station_data(station: pd.Series, repo_dir: str) -> str:
    """
    Download the DWD data for a station.
    
    Args:
        station: Series with station metadata (must contain 'station_id' and 'name').
        repo_dir: Path to the repository root directory.
        
    Returns:
        Path to the extracted data folder.
    """
    # Construct the URL for the daily DWD data
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical"

    # Download the zip and extract
    foldername = f"{repo_dir}/data/dwd/{station['station_id']}_data"
    if not os.path.exists(foldername):
        print(f"Requesting zip urls for {station['name']}...")
        
        # get all zip paths
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
        
        # download the zip file
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
    
    return foldername


def read_station_temperature(foldername: str) -> pd.DataFrame:
    """
    Read the temperature data for the station.
    
    Metadata for the column names:
    https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/standard_format/formate_kx.html

    Args:
        foldername: Path to the extracted station data folder.
        
    Returns:
        DataFrame with columns MESS_DATUM and TXK (maximum temperature).
        
    Raises:
        FileNotFoundError: If no climate data file is found in the folder.
    """
    # Read the data
    files = os.listdir(foldername)
    kl_file = [f for f in files if f.startswith("produkt_klima_tag") and f.endswith(".txt")]

    if not kl_file:
        raise FileNotFoundError(f"No climate data file found in {foldername}.")

    station_kl = pd.read_csv(f"{foldername}/{kl_file[0]}", sep=";")

    # trim column names
    station_kl.columns = [col.strip() for col in station_kl.columns]

    # date to datetime64
    station_kl['MESS_DATUM'] = pd.to_datetime(station_kl['MESS_DATUM'], format="%Y%m%d")

    # extract TXK temperature column
    station_temp_max = station_kl[['MESS_DATUM', 'TXK']]

    # replace -999.0 with NaN
    station_temp_max.loc[:, 'TXK'] = station_temp_max['TXK'].replace(-999.0, np.nan)

    print(f"Number of missing values in maximum temperature: {station_temp_max['TXK'].isna().sum()}")
    print("Head of maximum temperature data:")
    print(station_temp_max.head(3))
    
    return station_temp_max


def get_consecutive_hot_days(
    station_temp_max: pd.DataFrame,
    min_temperature: float,
    consecutive_days: int,
    start_year: int,
    end_year: int
) -> list[str]:
    """
    Get consecutive days with high temperatures and return date strings for STAC queries.
    
    Args:
        station_temp_max: DataFrame with MESS_DATUM and TXK columns.
        min_temperature: Minimum temperature threshold.
        consecutive_days: Number of consecutive days above threshold.
        start_year: Start year for filtering.
        end_year: End year for filtering.
        
    Returns:
        Sorted list of date strings in YYYY-MM-DD format.
    """
    ######## Get consecutive days with high temperatures #######
    # summed days with max temperature >= min_temperature for rolling window of 3 days
    station_temp_max.loc[:, 'gt_roll'] = station_temp_max['TXK'].ge(min_temperature).rolling(window=consecutive_days).sum()

    # get only the days where the rolling window is equal to the number of consecutive days
    station_temp_max_gt = station_temp_max[station_temp_max['gt_roll'] == consecutive_days].copy()

    ####### Query for the high temperature days #########
    # Get the defined years of consecutive high temperatures for day 2 and 3 in a compatible date format
    station_temp_max_gt = station_temp_max_gt[station_temp_max_gt.MESS_DATUM.dt.year >= start_year].copy()
    station_temp_max_gt = station_temp_max_gt[station_temp_max_gt.MESS_DATUM.dt.year <= end_year].copy()
    station_temp_max_gt_dates = station_temp_max_gt.MESS_DATUM.to_list()

    # get the day before each third day as well
    station_temp_max_gt_dates_before = [date - pd.DateOffset(days=1) for date in station_temp_max_gt_dates]

    # merge the two lists
    station_temp_max_gt_dates.extend(station_temp_max_gt_dates_before)

    # sort the list
    station_temp_max_gt_dates.sort()

    # remove duplicates
    station_temp_max_gt_dates = list(dict.fromkeys(station_temp_max_gt_dates))

    # get dates in format YYYY-MM-DD
    station_temp_max_gt_dates = [date.strftime("%Y-%m-%d") for date in station_temp_max_gt_dates]
    
    return station_temp_max_gt_dates


def get_high_temperature_dates(
    region: str,
    repo_dir: str,
    bbox_gdf: gpd.GeoDataFrame,
    start_year: int,
    end_year: int,
    min_temperature: float,
    consecutive_days: int
) -> list[str]:
    """
    Full pipeline: find a DWD station for the region, download data, and return
    date strings for consecutive high-temperature days.
    
    Args:
        region: Region name to match against DWD stations.
        repo_dir: Path to the repository root directory.
        bbox_gdf: GeoDataFrame with the bounding box geometry.
        start_year: Start year for station data availability.
        end_year: End year for station data availability.
        min_temperature: Minimum temperature threshold.
        consecutive_days: Number of consecutive days above threshold.
        
    Returns:
        Sorted list of date strings in YYYY-MM-DD format.
    """
    stations_df = load_dwd_stations(repo_dir)
    stations_gpd = find_stations_for_region(stations_df, region)
    stations_gpd = select_closest_station(stations_gpd, bbox_gdf, start_year, end_year)
    
    station = stations_gpd.iloc[0]
    foldername = download_station_data(station, repo_dir)
    station_temp_max = read_station_temperature(foldername)
    
    dates = get_consecutive_hot_days(
        station_temp_max, min_temperature, consecutive_days, start_year, end_year
    )
    
    return dates

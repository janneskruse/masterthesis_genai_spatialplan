######## Corine Landcover data loading and processing functions #######

##### Import libraries ######
# system
import os
from pathlib import Path
from typing import Optional, Tuple

# data manipulation
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr


def get_latest_corine_folder(corine_base_path: str) -> str:
    """
    Get the latest Corine Landcover folder based on year in folder name.
    
    Parameters:
    -----------
    corine_base_path (str):
        Base path to the Corine data directory
        
    Returns:
    --------
    str:
        Name of the latest Corine folder
        
    Raises:
    -------
    FileNotFoundError:
        If no Corine folders are found
    """
    corine_folders = [
        folder for folder in os.listdir(corine_base_path) 
        if os.path.isdir(os.path.join(corine_base_path, folder))
    ]
    
    if not corine_folders:
        raise FileNotFoundError(f"No Corine folders found in {corine_base_path}")
    
    # Reverse sort by year (extract last number in folder name)
    corine_folders.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
    
    return corine_folders[0]


def read_corine_crs_from_xml(xml_folder_path: str) -> str:
    """
    Parse Corine metadata XML to extract CRS information.
    
    Parameters:
    -----------
    xml_folder_path (str):
        Path to the folder containing Corine metadata XML files
        
    Returns:
    --------
    str:
        CRS string in EPSG format (e.g., "EPSG:3035")
        
    Raises:
    -------
    FileNotFoundError:
        If no XML files are found
    ValueError:
        If CRS information cannot be extracted from XML
    """
    xml_files = [f for f in os.listdir(xml_folder_path) if f.endswith(".xml")]
    
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {xml_folder_path}")
    
    # Get shortest XML path (usually the main metadata file)
    xml_path = min(xml_files, key=len)
    
    with open(os.path.join(xml_folder_path, xml_path), 'r') as xml_file:
        xml_content = xml_file.read()
        
        # Find the CRS string
        start_index = xml_content.find('EPSG:')
        if start_index == -1:
            raise ValueError(f"Could not find EPSG code in {xml_path}")
        
        end_index = xml_content.find('</gmd:code>', start_index)
        if end_index == -1:
            raise ValueError(f"Could not parse EPSG code from {xml_path}")
        
        crs_string = xml_content[start_index:end_index].split('<')[0]
    
    return crs_string


def load_corine_legend(corine_folder_path: str) -> pd.DataFrame:
    """
    Load and parse the Corine Landcover legend file.
    
    Parameters:
    -----------
    corine_folder_path (str):
        Path to the Corine folder containing the Legend subfolder
        
    Returns:
    --------
    pd.DataFrame:
        DataFrame with Corine class information, including 'Class_Name' column
        
    Raises:
    -------
    FileNotFoundError:
        If no legend file is found
    """
    legend_folder = os.path.join(corine_folder_path, "Legend")
    legend_files = [f for f in os.listdir(legend_folder) if f.endswith(".txt")]
    
    if not legend_files:
        raise FileNotFoundError(f"No legend file found in {legend_folder}")
    
    legend_path = legend_files[0]
    legend = pd.read_csv(os.path.join(legend_folder, legend_path), sep=",", header=None)
    
    # Rename last column to Class_Name
    column_count = legend.shape[1]
    legend = legend.rename(columns={column_count - 1: "Class_Name"})
    
    return legend


def load_corine_raster(
    repo_dir: str,
    bbox_gdf: gpd.GeoDataFrame,
    target_crs: str,
    corine_base_path: Optional[str] = None
) -> Tuple[xr.DataArray, str]:
    """
    Load Corine Landcover raster, clip to bounding box, and reproject to target CRS.
    
    Parameters:
    -----------
    repo_dir (str):
        Path to the repository directory
    bbox_gdf (gpd.GeoDataFrame):
        Bounding box GeoDataFrame for clipping (in any CRS, will be reprojected)
    target_crs (str):
        Target CRS to reproject the raster to (e.g., "EPSG:32633")
    corine_base_path (str, optional):
        Custom path to Corine data directory. If None, uses repo_dir/data/corine
        
    Returns:
    --------
    Tuple[xr.DataArray, str]:
        - Corine raster as xarray DataArray, clipped and reprojected
        - Path to the Corine folder used
        
    Raises:
    -------
    FileNotFoundError:
        If Corine data files are not found
    """
    # Set default Corine path
    if corine_base_path is None:
        corine_base_path = os.path.join(repo_dir, "data", "corine")
    
    # Get latest Corine folder
    corine_folder = get_latest_corine_folder(corine_base_path)
    corine_folder_path = os.path.join(corine_base_path, corine_folder)
    
    # Load GeoTIFF
    data_folder = os.path.join(corine_folder_path, "DATA")
    geotiff_files = [f for f in os.listdir(data_folder) if f.endswith(".tif")]
    
    if not geotiff_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {data_folder}")
    
    geotiff_path = geotiff_files[0]
    corine_raster = rxr.open_rasterio(
        os.path.join(data_folder, geotiff_path),
        masked=True
    ).squeeze("band", drop=True)
    
    # Read and set CRS from metadata
    metadata_folder = os.path.join(corine_folder_path, "Metadata")
    crs_string = read_corine_crs_from_xml(metadata_folder)
    corine_raster.rio.write_crs(crs_string, inplace=True)
    
    # Reproject bbox to Corine CRS
    bbox_gdf_corine = bbox_gdf.to_crs(corine_raster.rio.crs)
    
    # Clip to bounding box
    bounds = bbox_gdf_corine.geometry.bounds
    corine_raster = corine_raster.rio.clip_box(
        minx=bounds.minx.values[0],
        miny=bounds.miny.values[0],
        maxx=bounds.maxx.values[0],
        maxy=bounds.maxy.values[0]
    )
    
    # Reproject to target CRS
    corine_raster = corine_raster.rio.reproject(target_crs)
    
    return corine_raster, corine_folder_path


def filter_corine_by_classes(
    corine_raster: xr.DataArray,
    corine_folder_path: str,
    drop_classes: list
) -> xr.DataArray:
    """
    Filter Corine raster by removing specified land cover classes.
    
    Parameters:
    -----------
    corine_raster (xr.DataArray):
        Corine raster to filter
    corine_folder_path (str):
        Path to the Corine folder containing the legend
    drop_classes (list):
        List of class names to filter out (set to NaN)
        
    Returns:
    --------
    xr.DataArray:
        Filtered Corine raster with specified classes set to NaN
    """
    import numpy as np
    
    # Load legend
    legend = load_corine_legend(corine_folder_path)
    
    # Filter by class names
    filtered_raster = corine_raster.copy()
    for class_name in drop_classes:
        class_df = legend[legend["Class_Name"].str.strip() == class_name.strip()]
        
        if class_df.empty:
            print(f"   Warning: Class '{class_name}' not found in legend, skipping...")
            continue
        
        class_id = class_df.index[0]
        filtered_raster = filtered_raster.where(filtered_raster != class_id, other=np.nan)
    
    return filtered_raster

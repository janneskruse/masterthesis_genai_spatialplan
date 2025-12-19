######## Metropolitan regions utilities for loading GHSL bounding boxes #######

##### Import libraries ######
# system
import os
from pathlib import Path
from typing import Optional

# data manipulation
import geopandas as gpd


def get_region_bbox(
    region: str,
    repo_dir: str,
    ghsl_parquet_path: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load the bounding box for a specified region from GHSL regions data.
    
    Parameters:
    -----------
    region (str):
        Name of the region to retrieve (e.g., "Leipzig")
    repo_dir (str):
        Path to the repository directory
    ghsl_parquet_path (str, optional):
        Custom path to GHSL regions parquet file. If None, uses default path
        
    Returns:
    --------
    gpd.GeoDataFrame:
        GeoDataFrame with the bounding box geometry for the specified region in EPSG:4326
        
    Raises:
    -------
    FileNotFoundError:
        If the GHSL regions parquet file does not exist
    ValueError:
        If the specified region is not found in the GHSL data
    """
    # Set default path
    if ghsl_parquet_path is None:
        ghsl_parquet_path = os.path.join(repo_dir, "data", "processed", "ghsl_regions.parquet")
    
    # Check if file exists
    if not os.path.exists(ghsl_parquet_path):
        raise FileNotFoundError(
            f"GHSL regions file not found at {ghsl_parquet_path}. "
            "Please ensure the GHSL regions data has been processed."
        )
    
    # Load GHSL regions data
    try:
        ghsl_df = gpd.read_parquet(ghsl_parquet_path)
    except Exception as e:
        raise IOError(f"Failed to read GHSL regions parquet file: {e}")
    
    # Check if required columns exist
    if "region_name" not in ghsl_df.columns:
        raise ValueError("GHSL data is missing 'region_name' column")
    
    if "bbox" not in ghsl_df.columns:
        raise ValueError("GHSL data is missing 'bbox' column")
    
    # Filter for specified region
    region_data = ghsl_df[ghsl_df["region_name"] == region]
    
    if region_data.empty:
        available_regions = sorted(ghsl_df["region_name"].unique().tolist())
        raise ValueError(
            f"Region '{region}' not found in GHSL data. "
            f"Available regions: {', '.join(available_regions)}"
        )
    
    # Create GeoDataFrame with bbox geometry
    bbox_gdf = gpd.GeoDataFrame(
        geometry=region_data.bbox,
        crs="EPSG:4326"
    )
    
    return bbox_gdf

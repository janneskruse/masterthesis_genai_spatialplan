##### Import libraries ######
# data manipulation
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon


def create_grid(bbox_gdf: gpd.GeoDataFrame, length: float = 0.03, width: float = 0.03) -> gpd.GeoDataFrame:
    """
    Create a grid of polygons for multithreaded OSM data extraction.
    
    Parameters:
    -----------
    bbox_gdf (gpd.GeoDataFrame):
        Bounding box GeoDataFrame
    length (float):
        Grid cell height in degrees (default: 0.03)
    width (float):
        Grid cell width in degrees (default: 0.03)
        
    Returns:
    --------
    gpd.GeoDataFrame:
        Grid of polygons covering the bounding box
    """
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds
    
    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + length, length))
    
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([
                (x, y), (x+width, y), (x+width, y+length), (x, y+length)
            ]))
    
    return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
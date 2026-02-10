"""
===============================================================================
OSM data processing functions for rasterization
================================================================================
"""
##### Import libraries ######
# System
import os
import time
from typing import Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data handling
from array import array
import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
from shapely import wkb
import geoarrow.pyarrow as ga

# Data Science/ML
from scipy.ndimage import label

# Visualization
from tqdm.auto import tqdm

# Local imports
from data_acquisition.osm.request import fetch_overpass_data
from data_acquisition.osm.classify import classify_building_shapes
from data_acquisition.cube.rasterize import register_xarray_accessor
from data_acquisition.cube.vectorize import xr_vectorize, vectorize_array

######## Constants ########

# Street width classifications based on German road standards
# Richtlinien für die Anlage von Autobahnen and Landstraßen
STREET_WIDTHS = {
    'motorway': 24,
    'motorway_link': 16,
    'trunk': 24,
    'trunk_link': 16,
    'primary': 15,
    'primary_link': 12,
    'secondary': 12,
    'secondary_link': 11,
    'tertiary': 11,
    'tertiary_link': 11,
    'residential': 5.5,
    'living_street': 5.5,
    'pedestrian': 2,
    'road': 11,
    'service': 5.5,
    'minor_service': 5.5,
    'footway': 2,
    'cycleway': 2,
    'path': 2,
    'steps': 2,
}

# Waterway buffer widths (in meters)
WATERWAY_BUFFER = {
    'river': 14,
    'stream': 1,
    'canal': 3,
}

# Register the XarrayAccessor for GeoDataFrames
register_xarray_accessor()


######## Data Extraction Functions ########
def extract_features_grid(
    grid: gpd.GeoDataFrame,
    tags: Dict,
    max_workers: int = 1
) -> pd.DataFrame:
    """
    Extract OSM features for a grid using multithreading.
    
    Parameters:
    -----------
    grid (gpd.GeoDataFrame):
        Grid of polygons to query
    tags (Dict):
        OSM tags to query
    max_workers (int):
        Number of parallel workers (default: 1)
        
    Returns:
    --------
    pd.DataFrame:
        DataFrame with extracted OSM features
    """
    features = []
    delay_between_requests = 5  # seconds between requests to avoid Overpass rate limiting
    
    if max_workers <= 1:
        # Sequential with rate limiting
        for _, row in tqdm(grid.iterrows(), total=len(grid), desc="Extracting OSM features"):
            try:
                result = fetch_overpass_data(row.geometry.bounds, tags)
                if result is not None:
                    features.append(result)
                time.sleep(delay_between_requests)
            except Exception as e:
                print(f"Error extracting features: {e}")
    else:
        # Parallel (though Overpass may throttle aggressively)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_overpass_data, row.geometry.bounds, tags): row 
                for _, row in grid.iterrows()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting OSM features"):
                try:
                    result = future.result()
                    if result is not None:
                        features.append(result)
                except Exception as e:
                    print(f"Error extracting features: {e}")
    
    return pd.concat(features, ignore_index=True)


######## Feature Processing Functions ########

def process_streets(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_crs: str,
    output_path: Optional[str] = None
) -> xr.Dataset:
    """
    Process street features from OSM data and create rasterized dataset.
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    utm_crs (str):
        UTM CRS for buffering operations
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.Dataset:
        Dataset with rasterized street data
    """
    # Filter streets
    streets_gdf = osm_gdf[osm_gdf["highway"].notnull()].copy()
    streets_gdf = streets_gdf[streets_gdf.geometry.type == "LineString"].copy()
    
    # Remove columns with more than 50% NaNs
    streets_gdf = streets_gdf.dropna(axis=1, thresh=len(streets_gdf) * 0.5)
    
    # Rename lit to lighting
    if "lit" in streets_gdf.columns:
        streets_gdf = streets_gdf.rename(columns={"lit": "lighting"})
    
    # Apply width classification
    streets_gdf['buffer_width'] = streets_gdf['highway'].apply(
        lambda x: STREET_WIDTHS.get(x, 5.5)
    )
    
    # Convert to projected coordinates and buffer
    streets_gdf = streets_gdf.to_crs(utm_crs)
    streets_gdf["geometry"] = streets_gdf.apply(
        lambda row: row['geometry'].buffer(row['buffer_width']), axis=1
    )
    streets_gdf = streets_gdf.to_crs(epsg=4326)
    
    # Create rasterized dataarrays
    streets_xr = streets_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="streets",
        long_name="Streets OSM",
        description="Rasterized streets from OSM data",
        mapping_col=None,
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    streets_xr_surface = streets_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="streets_surface",
        long_name="Streets OSM surface",
        description="Rasterized streets with surface types from OSM data",
        mapping_col='surface',
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    streets_xr_service = streets_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="streets_service",
        long_name="Streets OSM service",
        description="Rasterized streets with service types from OSM data",
        mapping_col='highway',
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    # Merge datasets
    streets_ds = xr.merge([streets_xr, streets_xr_surface, streets_xr_service], compat="override")
    streets_ds.attrs.update(streets_xr.attrs)
    
    if output_path:
        streets_ds.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return streets_ds


def process_street_blocks(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_crs: str,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process street blocks by inverting main streets.
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    utm_crs (str):
        UTM CRS for buffering operations
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.DataArray:
        DataArray with rasterized street blocks
    """
    # Filter main streets (exclude pedestrian paths, cycleways, etc.)
    exclude_types = ['cycleway', 'path', 'pedestrian', 'service', 'footway', 
                     'construction', 'track', 'steps', 'bridleway', 'corridor', 
                     'elevator', 'platform']
    
    streets_gdf = osm_gdf[osm_gdf["highway"].notnull()].copy()
    streets_gdf = streets_gdf[~streets_gdf["highway"].isin(exclude_types)]
    
    # Apply width classification
    streets_gdf['buffer_width'] = streets_gdf['highway'].apply(
        lambda x: STREET_WIDTHS.get(x, 5.5)
    )
    
    # Convert to projected coordinates and buffer
    streets_gdf = streets_gdf.to_crs(utm_crs)
    streets_gdf["geometry"] = streets_gdf.apply(
        lambda row: row['geometry'].buffer(row['buffer_width']), axis=1
    )
    streets_gdf = streets_gdf.to_crs(epsg=4326)
    
    # Rasterize main streets
    streets_main_xr = streets_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="streets_main",
        long_name="Main Streets OSM",
        description="Rasterized main streets from OSM data",
        mapping_col=None,
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    # Invert to get street blocks
    street_blocks_xr = xr.where(streets_main_xr == 0, 1, 0)
    street_blocks_xr.name = "street_blocks"
    street_blocks_xr.attrs.update({
        "long_name": "Street Blocks OSM",
        "description": "Rasterized street blocks from OSM data",
    })
    
    if output_path:
        street_blocks_xr.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return street_blocks_xr


def process_water_bodies(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_crs: str,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process water bodies and waterways from OSM data.
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    utm_crs (str):
        UTM CRS for buffering operations
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.DataArray:
        DataArray with rasterized water bodies
    """
    # Filter water features
    water_gdf = osm_gdf[
        (osm_gdf["water"].isin(["lake", "river", "canal"])) | 
        (osm_gdf["waterway"].isin(["river", "stream", "canal"]))
    ].copy()
    
    if water_gdf.empty:
        raise ValueError("No water features found in OSM data")
    
    water_gdf = water_gdf[["id", "geometry", "name", "water", "waterway"]]
    
    # Apply buffer widths
    water_gdf['buffer_width'] = water_gdf['waterway'].apply(
        lambda x: WATERWAY_BUFFER.get(x, 5)
    )
    
    # Convert to projected coordinates and buffer
    water_gdf = water_gdf.to_crs(utm_crs)
    water_gdf["geometry"] = water_gdf.apply(
        lambda row: row['geometry'].buffer(row['buffer_width']), axis=1
    )
    water_gdf = water_gdf.to_crs(epsg=4326)
    
    # Create combined water column
    water_gdf["combined_water"] = water_gdf["water"].combine_first(water_gdf["waterway"])
    
    # Remove duplicates
    water_gdf = water_gdf.drop_duplicates(subset=['id'])
    water_gdf = water_gdf.drop(columns=['water', 'waterway'])
    
    # Rasterize
    water_xr = water_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="water",
        long_name="Water OSM",
        description="Rasterized water from OSM data",
        mapping_col="combined_water",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    if output_path:
        water_xr.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return water_xr


def process_buildings(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    output_path: Optional[str] = None
) -> xr.Dataset:
    """
    Process building footprints from OSM data.
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.Dataset:
        Dataset with rasterized building data
    """
    buildings_gdf = osm_gdf[osm_gdf["building"].notnull()].copy()
    
    # Keep columns with at least 50% data
    buildings_gdf = buildings_gdf.dropna(axis=1, thresh=len(buildings_gdf) * 0.5)
    
    # Rasterize buildings
    buildings_xr = buildings_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="buildings",
        long_name="Buildings OSM",
        description="Rasterized buildings from OSM data",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    buildings_xr_service = buildings_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="buildings_service",
        long_name="Buildings OSM service",
        description="Rasterized buildings with service types from OSM data",
        mapping_col="building",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    buildings_ds = xr.merge([buildings_xr, buildings_xr_service], compat="override")
    buildings_ds.attrs.update(buildings_xr.attrs)
    
    if output_path:
        buildings_ds.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return buildings_ds


def process_building_heights(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    region: str,
    repo_dir: str,
    gdf_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process 3D building heights from Yangzi Che et al. (2024) dataset.
    
    Uses DuckDB for efficient spatial filtering of large parquet files.
    
    Parameters:
    -----------
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    region (str):
        Region name for temporary table naming
    repo_dir (str):
        Repository directory path
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.DataArray:
        DataArray with rasterized building heights
    """
    
    if not gdf_path:
        gdf_path = f"{repo_dir}/data/che_etal/building_heights_{region}.parquet"
    
    if not os.path.exists(gdf_path):
    
        xmin, ymin, xmax, ymax = bbox
        
        # Initialize DuckDB spatial extension
        duckdb.sql("""
            INSTALL spatial;
            LOAD spatial;
            SET enable_progress_bar = true;
            SET enable_geoparquet_conversion = false;
            
            CALL register_geoarrow_extensions();
        """)
        
        # Query within bbox, convert WKB geometry explicitly, and sort by Hilbert curve
        duckdb.sql(f"""
            CREATE TEMP TABLE tmp_buildings_{region} AS
            SELECT
                Height AS height,
                ST_GeomFromWKB(GEOMETRY) AS geometry
            FROM read_parquet('{repo_dir}/data/che_etal/Germany_Hungary_Iceland/building_heights_germany.parquet', 
                            filename=true, hive_partitioning=1)
            WHERE ST_Within(
                ST_GeomFromWKB(GEOMETRY),
                ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})
            )
            ORDER BY ST_Hilbert(ST_GeomFromWKB(GEOMETRY), 
                            ST_Extent(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})))
        """)
        
        result = duckdb.sql(f"SELECT * FROM tmp_buildings_{region}")
        
        # Fetch results
        gdf = gpd.GeoDataFrame.from_arrow(result)
        
        # Drop temp table
        duckdb.sql(f"DROP TABLE tmp_buildings_{region}")
        
        # Close duckdb connection
        duckdb.close()
        
        # set CRS
        if not gdf.crs:
            gdf.set_crs(epsg=4326, inplace=True)
        
        # save as geoparquet
        os.makedirs(os.path.dirname(gdf_path), exist_ok=True)
        gdf.to_parquet(gdf_path, index=False)
    else:
        gdf = gpd.read_parquet(gdf_path)
    
    # Rasterize
    building_heights_xr = gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="buildings_heights",
        long_name="Buildings Heights OSM",
        description="Rasterized building heights from Che et al. (2024)",
        mapping_col="height",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    if output_path:
        building_heights_xr.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return building_heights_xr


def process_building_heights_deprecated(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    region: str,
    repo_dir: str,
    gdf_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process 3D building heights from Yangzi Che et al. (2024) dataset.
    
    Uses DuckDB for efficient spatial filtering of large parquet files.
    
    Parameters:
    -----------
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    region (str):
        Region name for temporary table naming
    repo_dir (str):
        Repository directory path
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.DataArray:
        DataArray with rasterized building heights
    """
    
    if not gdf_path:
        gdf_path = f"{repo_dir}/data/che_etal/building_heights_{region}.parquet"
    
    if not os.path.exists(gdf_path):
    
        xmin, ymin, xmax, ymax = bbox
        
        # Initialize DuckDB spatial extension
        duckdb.sql("""
            INSTALL spatial;
            LOAD spatial;
            SET enable_geoparquet_conversion = false;
            SET enable_progress_bar = true;
        """)
        
        # Query within bbox and sort by Hilbert curve
        duckdb.sql(f"""
            CREATE TEMP TABLE tmp_buildings_{region} AS
            SELECT
                Height AS height,
                ST_AsWKB(ST_GeomFromWKB("GEOMETRY")) AS geom
            FROM read_parquet('{repo_dir}/data/che_etal/Germany_Hungary_Iceland/building_heights_germany.parquet', 
                            filename=true, hive_partitioning=1)
            WHERE ST_Within(
                ST_GeomFromWKB("GEOMETRY"),
                ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})
            )
            ORDER BY ST_Hilbert(ST_GeomFromWKB("GEOMETRY"), 
                            ST_Extent(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax})))
        """)
        
        # Fetch results
        building_heights_table = duckdb.sql(f"SELECT * FROM tmp_buildings_{region}").fetch_arrow_table()
        building_heights_df = duckdb.sql(f"SELECT * FROM tmp_buildings_{region}").df()
        
        # Drop temp table
        duckdb.sql(f"DROP TABLE tmp_buildings_{region}")
        
        # close duckdb connection
        duckdb.close()
        
        # Convert WKB to GeoDataFrame using geoarrow
        wkb_list = building_heights_table['geom'].to_pylist()
        
        # Collect coordinates for geoarrow
        poly_ring_offsets = [0]
        ring_coord_offsets = [0]
        xs_list = []
        ys_list = []
        n_rings = 0
        n_coords = 0
        
        for wkb_blob in wkb_list:
            geom = wkb.loads(wkb_blob)
            
            if geom.is_empty:
                poly_ring_offsets.append(n_rings)
                continue
            
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                poly_ring_offsets.append(n_rings)
                continue
            
            for poly in polys:
                rings = [poly.exterior, *poly.interiors]
                for ring in rings:
                    coords = np.asarray(ring.coords, dtype=np.float64)
                    xs_list.extend(coords[:, 0].tolist())
                    ys_list.extend(coords[:, 1].tolist())
                    n_coords += len(coords)
                    ring_coord_offsets.append(n_coords)
                    n_rings += 1
            
            poly_ring_offsets.append(n_rings)
        
        # Create geoarrow polygon array
        ring_offsets_buf = array('i', poly_ring_offsets)
        coord_offsets_buf = array('i', ring_coord_offsets)
        xs_buf = array('d', xs_list)
        ys_buf = array('d', ys_list)
        
        polygon_array = ga.polygon().from_geobuffers(
            None, ring_offsets_buf, coord_offsets_buf, xs_buf, ys_buf
        )
        
        gdf = ga.to_geopandas(polygon_array)
        
        building_heights_gdf = gpd.GeoDataFrame(
            building_heights_df.reset_index(drop=True),
            geometry=gdf.geometry,
            crs="EPSG:4326"
        ).drop(columns=['geom'])
        
        # save as geoparquet
        os.makedirs(os.path.dirname(gdf_path), exist_ok=True)
        building_heights_gdf.to_parquet(gdf_path, index=False)
    else:
        building_heights_gdf = gpd.read_parquet(gdf_path)
    
    # Rasterize
    building_heights_xr = building_heights_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="buildings_heights",
        long_name="Buildings Heights OSM",
        description="Rasterized building heights from Che et al. (2024)",
        mapping_col="height",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    if output_path:
        building_heights_xr.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return building_heights_xr


def process_building_shapes(
    osm_gdf: gpd.GeoDataFrame,
    building_heights_gdf_path: str,
    street_blocks_xr_path: str,
    buildings_xr_path: str,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_crs: str,
    gdf_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Processes buildings into shapes and
    classifies them to structural types.
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    building_heights_gdf_path (str):
        Path to GeoDataFrame with building heights from Che et al. (2024)
    street_blocks_xr_path (str):
        Path to street blocks xarray zarr file
    buildings_xr_path (str):
        Path to buildings xarray zarr file
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    utm_crs (str):
        UTM CRS for buffering operations
    output_path (str, optional):
        Path to save output zarr file
    gdf_path (str, optional):
        Path to save intermediate GeoDataFrame with building shapes
    
    Returns:
    --------    xr.Dataset:
        Dataset with building shapes and structural types
    """
    

    # set bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    
    # =============================
    # load the datasets
    # =============================
    buildings_xr= xr.open_zarr(buildings_xr_path, consolidated=True)
    building_heights_gdf = gpd.read_parquet(building_heights_gdf_path)
    
    street_blocks_xr = xr.open_zarr(street_blocks_xr_path, consolidated=True)
    street_blocks_gdf = xr_vectorize(
        street_blocks_xr.street_blocks,
        attribute_col="block",
        crs="EPSG:4326",
        dtype="float32",
    )
    street_blocks_gdf = street_blocks_gdf[street_blocks_gdf["block"]==1]
    
    # =============================
    # Process building shapes
    # =============================

    # label the building shapes
    labeled_array, num_features = label(buildings_xr.buildings.values)

    # vectorize the labeled array to get building shapes as polygons
    shapes_gdf = vectorize_array(
        labeled_array,
        bounds=(xmin, ymin, xmax, ymax),
        crs="EPSG:4326",
    )

    shapes_gdf = classify_building_shapes(
        osm_gdf=osm_gdf,
        building_heights_gdf=building_heights_gdf,
        street_blocks_gdf=street_blocks_gdf,
        shapes_gdf=shapes_gdf,
        utm_crs=utm_crs
    )
    
    # save to geoparquet
    if gdf_path:
        os.makedirs(os.path.dirname(gdf_path), exist_ok=True)
        shapes_gdf.to_parquet(gdf_path, index=False)
    
    # rasterize building shapes
    building_shapes_xr = shapes_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="building_shapes",
        long_name="Building Shapes OSM",
        description="Rasterized building shapes from OSM data",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    # rasterize structural types
    structural_types_xr = shapes_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="building_shapes_structural_types",
        long_name="Building Shapes Structural Types OSM",
        description="Rasterized building shapes classified to structural types from OSM data",
        mapping_col="structural_type",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    # rasterize elongation attribute
    elongation_xr = shapes_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="building_shapes_elongation",
        long_name="Building shapes Elongation OSM",
        description="Rasterized building shapes elongation attribute from OSM data",
        mapping_col="elongation",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    # merge datasets
    building_shapes_ds = xr.merge([building_shapes_xr, structural_types_xr, elongation_xr], compat="override")
    building_shapes_ds.attrs.update(building_shapes_xr.attrs)
    if output_path:
        building_shapes_ds.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return building_shapes_ds
    

def process_landuse(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    lon: np.ndarray,
    lat: np.ndarray,
    utm_crs: str,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process landuse features from OSM data (excluding streets, buildings, water).
    
    Parameters:
    -----------
    osm_gdf (gpd.GeoDataFrame):
        OSM features GeoDataFrame
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_width (int):
        Output raster width
    image_height (int):
        Output raster height
    lon (np.ndarray):
        Longitude coordinates
    lat (np.ndarray):
        Latitude coordinates
    utm_crs (str):
        UTM CRS for buffering operations
    output_path (str, optional):
        Path to save output zarr file
        
    Returns:
    --------
    xr.DataArray:
        DataArray with rasterized landuse
    """
    # Filter out streets, buildings, and water
    landuse_gdf = osm_gdf[
        ~osm_gdf["building"].notnull() & 
        ~osm_gdf["highway"].notnull() & 
        ~osm_gdf["railway"].notnull() & 
        ~osm_gdf["water"].isin(["lake", "river", "canal"]) & 
        ~osm_gdf["waterway"].isin(["river", "stream", "canal"])
    ].copy()
    
    # Select available columns
    required_columns = ["id", "geometry"]
    optional_columns = ["landuse", "boundary", "natural", "water", "waterway", 
                       "leisure", "railway", "amenity"]
    available_optional_columns = [col for col in optional_columns if col in landuse_gdf.columns]
    
    available_columns = required_columns + available_optional_columns
    landuse_gdf = landuse_gdf[available_columns]
    
    # Create combined landuse column
    if available_optional_columns:
        combined_landuse = landuse_gdf[available_optional_columns[0]]
        for col in available_optional_columns[1:]:
            combined_landuse = combined_landuse.combine_first(landuse_gdf[col])
        landuse_gdf["combined_landuse"] = combined_landuse
    else:
        landuse_gdf["combined_landuse"] = pd.NA
    
    # Remove duplicates
    landuse_gdf = landuse_gdf.drop_duplicates(subset=['id'])
    
    # Buffer railways slightly
    if 'railway' in landuse_gdf.columns:
        landuse_gdf['buffer_width'] = landuse_gdf['railway'].apply(
            lambda x: 0.5 if x == "rail" else 0
        )
    else:
        landuse_gdf['buffer_width'] = 0
    
    # Convert to projected coordinates and buffer
    landuse_gdf = landuse_gdf.to_crs(utm_crs)
    landuse_gdf["geometry"] = landuse_gdf.apply(
        lambda row: row['geometry'].buffer(row['buffer_width']), axis=1
    )
    landuse_gdf = landuse_gdf.to_crs(epsg=4326)
    
    # Remove empty geometries
    landuse_gdf = landuse_gdf[~landuse_gdf.geometry.is_empty]
    
    # Drop original columns
    columns_to_drop = [col for col in optional_columns if col in landuse_gdf.columns]
    landuse_gdf = landuse_gdf.drop(columns=columns_to_drop, errors='ignore')
    
    # Rasterize
    landuse_xr = landuse_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        x_coords=lon,
        y_coords=lat,
        name="landuse",
        long_name="Landuse OSM",
        description="Rasterized landuse from OSM data",
        mapping_col="combined_landuse",
        crs="EPSG:4326",
        x_dim="lon",
        y_dim="lat",
        units="1",
    )
    
    if output_path:
        landuse_xr.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return landuse_xr

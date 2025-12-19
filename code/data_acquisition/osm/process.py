######## OSM data processing functions for rasterization #######

##### Import libraries ######
# system
from typing import Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# data manipulation
from array import array
import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
from shapely import wkb
import geoarrow.pyarrow as ga

# visualization
from tqdm.auto import tqdm

# local imports
from data_acquisition.osm.request import fetch_overpass_data


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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_overpass_data, row.geometry.bounds, tags): row 
            for _, row in grid.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting OSM features"):
            try:
                features.append(future.result())
            except Exception as e:
                print(f"Error extracting features: {e}")
    
    return pd.concat(features, ignore_index=True)


######## Feature Processing Functions ########

def process_streets(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_size: int,
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
    image_size (int):
        Output raster size
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
    streets_gdf = streets_gdf[streets_gdf.geometry.type == "LineString"]
    
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
        image_size=image_size,
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
        image_size=image_size,
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
        image_size=image_size,
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
    streets_ds = xr.merge([streets_xr, streets_xr_surface, streets_xr_service])
    streets_ds.attrs.update(streets_xr.attrs)
    
    if output_path:
        streets_ds.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return streets_ds


def process_street_blocks(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_size: int,
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
    image_size (int):
        Output raster size
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
        image_size=image_size,
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
    image_size: int,
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
    image_size (int):
        Output raster size
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
        image_size=image_size,
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
    image_size: int,
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
    image_size (int):
        Output raster size
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
        image_size=image_size,
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
        image_size=image_size,
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
    
    buildings_ds = xr.merge([buildings_xr, buildings_xr_service])
    buildings_ds.attrs.update(buildings_xr.attrs)
    
    if output_path:
        buildings_ds.to_zarr(output_path, mode="w", consolidated=True, compute=True)
    
    return buildings_ds


def process_building_heights(
    bbox: Tuple[float, float, float, float],
    image_size: int,
    lon: np.ndarray,
    lat: np.ndarray,
    region: str,
    repo_dir: str,
    output_path: Optional[str] = None
) -> xr.DataArray:
    """
    Process 3D building heights from Yangzi Che et al. (2024) dataset.
    
    Uses DuckDB for efficient spatial filtering of large parquet files.
    
    Parameters:
    -----------
    bbox (Tuple):
        Bounding box (xmin, ymin, xmax, ymax)
    image_size (int):
        Output raster size
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
    xmin, ymin, xmax, ymax = bbox
    
    # Initialize DuckDB spatial extension
    duckdb.sql("""
        INSTALL spatial;
        LOAD spatial;
        SET enable_geoparquet_conversion = false;
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
    building_heights_table = duckdb.sql(f"SELECT * FROM tmp_buildings_{region}").arrow()
    building_heights_df = duckdb.sql(f"SELECT * FROM tmp_buildings_{region}").df()
    
    # Drop temp table
    duckdb.sql(f"DROP TABLE tmp_buildings_{region}")
    
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
    
    # Rasterize
    building_heights_xr = building_heights_gdf.to_raster.to_xr_dataarray(
        bbox=bbox,
        image_size=image_size,
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


def process_landuse(
    osm_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    image_size: int,
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
    image_size (int):
        Output raster size
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
        image_size=image_size,
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


def merge_osm_datasets(types_folder_path: str) -> xr.Dataset:
    """
    Merge all OSM rasterized datasets into a single dataset.
    
    Parameters:
    -----------
    types_folder_path (str):
        Path to folder containing individual zarr files
        
    Returns:
    --------
    xr.Dataset:
        Merged dataset with all OSM features
    """
    # Load all datasets
    building_heights_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_building_heights.zarr", 
        consolidated=True, decode_times=False
    )
    streets_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_streets.zarr", 
        consolidated=True, decode_times=False
    )
    street_blocks_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_street_blocks.zarr", 
        consolidated=True, decode_times=False
    )
    buildings_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_buildings.zarr", 
        consolidated=True, decode_times=False
    )
    landuse_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_landuse.zarr", 
        consolidated=True, decode_times=False
    )
    water_xr = xr.open_zarr(
        f"{types_folder_path}/rasterized_water.zarr", 
        consolidated=True, decode_times=False
    )
    
    # Remove spatial_ref coordinate if present (keep as attribute only)
    datasets = [streets_xr, street_blocks_xr, buildings_xr, 
                building_heights_xr, landuse_xr, water_xr]
    cleaned_datasets = []
    
    for ds in datasets:
        if 'spatial_ref' in ds.coords:
            ds = ds.drop_vars('spatial_ref')
        cleaned_datasets.append(ds)
    
    # Merge all datasets
    merged_xr = xr.merge(cleaned_datasets)
    
    return merged_xr

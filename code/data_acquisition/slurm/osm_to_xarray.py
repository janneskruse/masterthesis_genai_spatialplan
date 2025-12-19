######## Script to aquire and pre-process OSM data for the region to an Xarray raster image cube #######

##### Import libraries ######
# system
import os
import time

# data manipulation
import yaml
import json
import numpy as np
import geopandas as gpd
import utm
from pyproj import CRS

# local imports
from helpers.bbox import create_grid
from data_acquisition.osm.process import (
    extract_features_grid,
    process_streets, process_street_blocks, process_water_bodies,
    process_buildings, process_building_heights, process_landuse,
    merge_osm_datasets
)
from data_acquisition.cube.rasterize import register_xarray_accessor
from data_acquisition.cube.metropolitan_regions import get_region_bbox


#### Function to exit on error ######
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

config = {}
with open(f"{repo_dir}/code/data_acquisition/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)
    
    
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
osm_region_folder = f"{big_data_storage_path}/osm/{region.lower()}"
os.makedirs(osm_region_folder, exist_ok=True)

osm_zarr_name = f"{osm_region_folder}/osm_rasterized.zarr"

print("Processing region:", region, "at", time.strftime("%Y-%m-%d %H:%M:%S"), "to produce zarr file:", osm_zarr_name)
# exit(0)  # Exit early for testing purposes

######## Try except OSM data processing ########
try:
    if os.path.exists(osm_zarr_name):
        print(f"OSM data already exists at {osm_zarr_name}, skipping processing.")
        exit(0)
            
    ############ Define the bbox ############ 
    bbox_gdf = get_region_bbox(region=region, repo_dir=repo_dir)
    bbox_polygon=json.loads(bbox_gdf.to_json())['features'][0]['geometry']
    bbox = bbox_gdf.total_bounds
    coordinates=json.loads(bbox_gdf.geometry.to_json())["features"][0]["geometry"]["coordinates"]
    
    # Define UTM CRS for the region (e.g. 33N)
    easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
    is_south = zone_letter < 'N'  # True for southern hemisphere
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
    print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")

    # Create a grid for multithreaded OSM requests
    grid = create_grid(bbox_gdf, length=0.03, width=0.03)
    
    xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

    ######### Request the OpenStreetMap Data ########
    filename_all_features=f"{osm_region_folder}/osm_gdf.parquet"
    
    # Use default OSM tags
    tags = {
        "building": True,
        "waterway": True,
        "natural": ["water", "wood", "grassland", "wetland", "scrub", "heath", "moor", "bay", "beach", "sand", "mud"],
        "highway": True,
        "boundary": ["protected_area"],
        "landuse": True,
        "leisure": ["park", "garden", "playground", "pitch", "sports_centre"],
        "place": ["square"],
        "amenity": ["fountain", "school", "university", "college", "hospital", "kindergarten", "place_of_worship", "parking"],
        "aeroway": True,
        "railway": True,
    }
    
    # Threading parameters
    max_concurrent = 1  # Parallel requests to Overpass API

    if not os.path.exists(filename_all_features):
        # Download OSM features for the grid
        osm_gdf = extract_features_grid(grid, tags, max_workers=max_concurrent)

        # Clean geometry
        osm_gdf = osm_gdf[osm_gdf.geometry.is_valid]

        #remove duplicates by id
        osm_gdf = osm_gdf.drop_duplicates(subset=['id'])

        #set crs
        osm_gdf.crs = "EPSG:4326"

        #display summary of features
        print(f"Number of features: {len(osm_gdf)}")
        print("\nFeature types:")
        print(osm_gdf['geometry'].type.value_counts())

        #### Write to parquet for easier access ######
        osm_gdf.to_parquet(filename_all_features, index=False)
    else:
        print(f"OSM data already exists at {filename_all_features}, skipping download.")
        
        #read from parquet
        osm_gdf = gpd.read_parquet(filename_all_features)

    # Filter out invalid geometries
    osm_gdf = osm_gdf[osm_gdf.geometry.is_valid]

    ######### Create the rasterized datasets #########
    print("Creating rasterized datasets from OSM data...")
    
    # Register the XarrayAccessor for GeoDataFrames
    register_xarray_accessor()
    
    # Create the lat/lon coordinates and transform for the raster
    image_size = config["osm_query"].get("image_size", 5500)
    lat = np.linspace(ymax, ymin, image_size)  # Inverted for rasterio affine transform
    lon = np.linspace(xmin, xmax, image_size)
    bbox = (xmin, ymin, xmax, ymax)
    
    # Setup output folder
    types_folder_path = f"{osm_region_folder}/types"
    os.makedirs(types_folder_path, exist_ok=True)
    
    #### Process all OSM feature types
    print("Processing streets...")
    streets_zarr_name = f"{types_folder_path}/rasterized_streets.zarr"
    
    if not os.path.exists(streets_zarr_name):
        process_streets(osm_gdf, bbox, image_size, lon, lat, utm_crs, streets_zarr_name)
    
    print("Processing street blocks...")
    street_blocks_zarr_name = f"{types_folder_path}/rasterized_street_blocks.zarr"
    
    if not os.path.exists(street_blocks_zarr_name):
        process_street_blocks(osm_gdf, bbox, image_size, lon, lat, utm_crs, street_blocks_zarr_name)
    
    print("Processing water bodies...")
    water_zarr_name = f"{types_folder_path}/rasterized_water.zarr"
    
    if not os.path.exists(water_zarr_name):
        process_water_bodies(osm_gdf, bbox, image_size, lon, lat, utm_crs, water_zarr_name)
    
    print("Processing buildings...")
    buildings_zarr_name = f"{types_folder_path}/rasterized_buildings.zarr"
    
    if not os.path.exists(buildings_zarr_name):
        process_buildings(osm_gdf, bbox, image_size, lon, lat, buildings_zarr_name)
    
    print("Processing 3D building heights from Yangzi Che et al. (2024)...")
    building_heights_zarr_name = f"{types_folder_path}/rasterized_building_heights.zarr"
    
    if not os.path.exists(building_heights_zarr_name):
        process_building_heights(bbox, image_size, lon, lat, region, repo_dir, building_heights_zarr_name)
    
    print("Processing landuse...")
    landuse_zarr_name = f"{types_folder_path}/rasterized_landuse.zarr"
    
    if not os.path.exists(landuse_zarr_name):
        process_landuse(osm_gdf, bbox, image_size, lon, lat, utm_crs, landuse_zarr_name)
    
    ##### Merge all datasets ######
    print("Merging all datasets into a single xarray dataset...")
    merged_xr = merge_osm_datasets(types_folder_path)
    
    # Add spatial ref and rename coordinates
    merged_xr = merged_xr.rio.write_crs(merged_xr.attrs["spatial_ref"], inplace=True)
    merged_xr = merged_xr.rename({"lat": "y", "lon": "x"})
    
    # Save merged dataset
    print(f"Saving merged xarray dataset to {osm_zarr_name}")
    merged_xr.to_zarr(osm_zarr_name, mode="w", consolidated=True, compute=True)
    
    print(f"OSM data processing completed successfully for region {region} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    print(f"An error occurred: {e}")
    exit_with_error(f"An error occurred: {e}")

"""
================================================================================
Script to aquire and pre-process OSM data for the region to an Xarray 
raster image cube, saved as a Zarr file. 
=================================================================================
"""

##### Import libraries ######
# system
import argparse
import os
import time
import traceback

# data manipulation
import numpy as np
import geopandas as gpd
import utm
from pyproj import CRS

# local imports
from helpers.bbox import create_grid
from helpers.load_configs import add_config_arguments, load_configs
from data_acquisition.osm.process import (
    extract_features_grid,
    process_building_shapes,
    process_streets, process_street_blocks, process_water_bodies,
    process_buildings, process_building_heights, process_landuse,
)
from data_acquisition.cube.combine import merge_datasets
from data_acquisition.cube.metropolitan_regions import get_region_bbox


#### Function to exit on error ######
def exit_with_error(message):
    print(message)
    print("Finishing due to error at", time.strftime("%Y-%m-%d %H:%M:%S"))
    exit(1)

def main(args):

    ###### setup config variables #######
    config = load_configs()
    repo_dir = config.get("repo_dir", ".")
    config = config.get("data_config", {})
        
    ####### Get the region to process #######
    region = args.REGION
    # total_cpus = int(args.TOTAL_CPUS)

    # setup folders
    print(config)
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
        bbox = bbox_gdf.total_bounds
        # Define UTM CRS for the region (e.g. 33N)
        easting, northing, zone_number, zone_letter = utm.from_latlon(bbox_gdf.geometry.centroid.y.values[0], bbox_gdf.geometry.centroid.x.values[0])
        is_south = zone_letter < 'N'  # True for southern hemisphere
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': int(zone_number), 'south': is_south})
        print(f"UTM CRS: {utm_crs.to_authority()} with zone {zone_number}{zone_letter}")
        
        bbox_utm = bbox_gdf.to_crs(utm_crs).total_bounds
        width_m = bbox_utm[2] - bbox_utm[0]
        height_m = bbox_utm[3] - bbox_utm[1]

        # Create a grid for multithreaded OSM requests
        grid = create_grid(bbox_gdf, length=0.07, width=0.07)
        
        xmin, ymin, xmax, ymax = bbox_gdf.total_bounds

        ######### Request the OpenStreetMap Data ########
        filename_all_features=f"{osm_region_folder}/osm_gdf.parquet"
        
        # Use default OSM tags
        tags = {
            "building": True,
            "building:part": True,
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
        
        # Create the lat/lon coordinates and transform for the raster
        resolution = config["osm_query"].get("resolution", 3)
        image_width = int(width_m / resolution)
        image_height = int(height_m / resolution)
        lat = np.linspace(ymax, ymin, image_height)  # Inverted for rasterio affine transform
        lon = np.linspace(xmin, xmax, image_width)
        bbox = (xmin, ymin, xmax, ymax)
        
        # Setup output folder
        types_folder_path = f"{osm_region_folder}/types"
        os.makedirs(types_folder_path, exist_ok=True)
        
        #### Process all OSM feature types
        print("Processing streets...")
        streets_zarr_name = f"{types_folder_path}/rasterized_streets.zarr"
        
        if not os.path.exists(streets_zarr_name):
            process_streets(
                osm_gdf=osm_gdf, 
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                utm_crs=utm_crs, 
                output_path=streets_zarr_name, 
                use_main_streets=True,
                save_additional_layers=True
            )
        
        print("Processing street blocks...")
        street_blocks_zarr_name = f"{types_folder_path}/rasterized_street_blocks.zarr"
        
        if not os.path.exists(street_blocks_zarr_name):
            process_street_blocks(
                osm_gdf=osm_gdf, 
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                utm_crs=utm_crs, 
                output_path=street_blocks_zarr_name
            )
        
        print("Processing water bodies...")
        water_zarr_name = f"{types_folder_path}/rasterized_water.zarr"
        
        if not os.path.exists(water_zarr_name):
            process_water_bodies(
                osm_gdf=osm_gdf, 
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                utm_crs=utm_crs, 
                output_path=water_zarr_name
            )
        
        print("Processing buildings...")
        buildings_zarr_name = f"{types_folder_path}/rasterized_buildings.zarr"
        
        if not os.path.exists(buildings_zarr_name):
            process_buildings(
                osm_gdf=osm_gdf, 
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                output_path=buildings_zarr_name
            )
        
        print("Processing 3D building heights from Yangzi Che et al. (2024)...")
        building_heights_zarr_name = f"{types_folder_path}/rasterized_building_heights.zarr"
        building_heights_gdf_path = f"{big_data_storage_path}/che_etal/{region.lower()}/building_heights.parquet"
        
        if not os.path.exists(building_heights_zarr_name):
            process_building_heights(
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                region=region, 
                repo_dir=repo_dir, 
                gdf_path=building_heights_gdf_path,
                output_path=building_heights_zarr_name
            )
        
        print("Processing building shapes and classifying them...")
        building_shapes_zarr_name = f"{types_folder_path}/rasterized_building_shapes.zarr"
        building_shapes_gdf_path = f"{big_data_storage_path}/osm/{region.lower()}/building_shapes.parquet"
        
        if not os.path.exists(building_shapes_zarr_name):
            process_building_shapes(
                osm_gdf=osm_gdf,
                building_heights_gdf_path=building_heights_gdf_path,
                buildings_xr_path=buildings_zarr_name,
                street_blocks_xr_path=street_blocks_zarr_name,
                bbox_gdf=bbox_gdf,
                utm_crs=utm_crs,
                target_resolution=resolution,
                gdf_path=building_shapes_gdf_path,
                output_path=building_shapes_zarr_name
            )
        
        print("Processing landuse...")
        landuse_zarr_name = f"{types_folder_path}/rasterized_landuse.zarr"
        
        if not os.path.exists(landuse_zarr_name):
            process_landuse(
                osm_gdf=osm_gdf, 
                bbox=bbox, 
                image_width=image_width, 
                image_height=image_height, 
                lon=lon, 
                lat=lat, 
                utm_crs=utm_crs, 
                output_path=landuse_zarr_name
            )

        ##### Merge all datasets ######
        print("Merging all datasets into a single xarray dataset...")
        
        merged_xr = merge_datasets(
            [
                streets_zarr_name,
                street_blocks_zarr_name,
                water_zarr_name,
                buildings_zarr_name,
                building_heights_zarr_name,
                building_shapes_zarr_name,
                landuse_zarr_name
            ]
        )
        
        # Add spatial ref and rename coordinates
        merged_xr = merged_xr.rio.write_crs(merged_xr.attrs["spatial_ref"], inplace=True)
        merged_xr = merged_xr.rename({"lat": "y", "lon": "x"})
        
        # Save merged dataset
        print(f"Saving merged xarray dataset to {osm_zarr_name}")
        merged_xr.to_zarr(osm_zarr_name, mode="w", consolidated=True, compute=True)
        
        print(f"OSM data processing completed successfully for region {region} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"An error occurred: {e}")
        
        # print full stack trace for debugging
        traceback.print_exc()
        
        exit_with_error(f"An error occurred: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE DDP for Urban Inpainting')
    
    # Add config file arguments
    add_config_arguments(parser)
    
    parser.add_argument('--REGION', type=str, required=True, help='Metropolitan region to process (e.g. Berlin, London, New York)')
    # parser.add_argument('--TOTAL_CPUS', type=int, default=1, help='Total number of CPUs available for processing (default: 1)')
    
    args = parser.parse_args()
    
    main(args)

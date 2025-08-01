########## get unique dates from landsat zarr file and convert to filenames ##########
import os
import sys
import yaml
import json

def get_region_filenames(config_path):
    
    print("Getting region filenames from config...")
    
    if not config_path:
        p=os.popen('git rev-parse --show-toplevel')
        repo_dir = p.read().strip()
        p.close()
        config_path = f"{repo_dir}/config.yml"

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    regions = config.get("regions", [])
    big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    
    print("Regions to process:", regions)
    
    region_filenames = {}
    for region in regions:
        landsat_region_folder = f"{big_data_storage_path}/landsat/{region.lower()}"
        osm_region_folder = f"{big_data_storage_path}/osm/{region.lower()}"
        planet_region_folder = f"{big_data_storage_path}/planet_scope/{region.lower()}"
        os.makedirs(landsat_region_folder, exist_ok=True)
        os.makedirs(osm_region_folder, exist_ok=True)
        os.makedirs(planet_region_folder, exist_ok=True)

        years = config["temperature_day_filter"]["years"]
        end_year = years.get("end", 2023)
        start_year = years.get("start", 1950)
        min_temperature = config["temperature_day_filter"]["min"]
        max_cloud_cover = config["landsat_query"].get("max_cloud_coverage", 10)

        landsat_zarr_name = f"{landsat_region_folder}/landsat_temperature_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"
        planet_zarr_name = f"{planet_region_folder}/planet_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"
        osm_zarr_name = f"{osm_region_folder}/osm_rasterized.zarr"
        processed_zarr_name = f"{big_data_storage_path}/processed/{region.lower()}/input_config_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"

        region_filenames[region] = {
            "landsat_zarr_name": landsat_zarr_name,
            "osm_zarr_name": osm_zarr_name,
            "planet_zarr_name": planet_zarr_name,
            "processed_zarr_name": processed_zarr_name,
        }

    return region_filenames


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: get_region_filenames.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    filenames = get_region_filenames(config_path)
    print(json.dumps(filenames))
        
import os
import sys
import yaml

def get_landsat_config_vars(config_path, region):
    
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    big_data_storage_path = config.get("big_data_storage_path", "/work/zt75vipu-master/data")
    landsat_region_folder = f"{big_data_storage_path}/landsat/{region.lower()}"
    os.makedirs(landsat_region_folder, exist_ok=True)

    years = config["temperature_day_filter"]["years"]
    end_year = years.get("end", 2023)
    start_year = years.get("start", 1950)
    min_temperature = config["temperature_day_filter"]["min"]
    consecutive_days = config["temperature_day_filter"]["consecutive_days"]
    max_cloud_cover = config["landsat_query"].get("max_cloud_coverage", 10)
    collections = config["landsat_query"].get("collections", ["landsat-c2l2-st"])
    
    try:
        max_dates_per_year = config["temperature_day_filter"]["max_dates_per_year"]
    except:
        print("No max_dates_per_year configured")
    

    landsat_zarr_name = f"{landsat_region_folder}/landsat_temperature_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.zarr"
    stac_filename = f"{landsat_region_folder}/stac_query_ge{min_temperature}_cc{max_cloud_cover}_{start_year}_{end_year}.parquet"

    return {
        "big_data_storage_path": big_data_storage_path,
        "landsat_region_folder": landsat_region_folder,
        "end_year": end_year,
        "start_year": start_year,
        "min_temperature": min_temperature,
        "max_dates_per_year": max_dates_per_year if 'max_dates_per_year' in locals() else None,
        "consecutive_days": consecutive_days,
        "max_cloud_cover": max_cloud_cover,
        "collections": collections,
        "landsat_zarr_name": landsat_zarr_name,
        "stac_filename": stac_filename,
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: landsat_config_helper.py <config_path> <region>")
        sys.exit(1)
    config_path = sys.argv[1]
    region = sys.argv[2]
    result = get_landsat_config_vars(config_path, region)
    # Print as key=value pairs for easy parsing in bash
    for k, v in result.items():
        if isinstance(v, list):
            v = ','.join(map(str, v))
        print(f"{k}={v}")
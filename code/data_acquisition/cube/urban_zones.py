## Import libraries
# system
import os

# data manipulation
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
import rioxarray as rxr # adds rioxarray capabilities to xarray
import xarray as xr

# local imports
from data_acquisition.cube.vectorize import xr_vectorize

# function to define a urban areas mask from OSM data and the Corine Landcover dataset
def define_urban_areas(region: str,
                        big_data_storage_path: str,
                        repo_dir: str,
                        utm_crs="EPSG:32633", 
                        max_distance=20, 
                        bbox_gdf=None):
    """ 
    Define urban areas from OSM data and the Corine Landcover dataset.
    Creates and saves a binary urban areas mask as a Zarr file.
    Args:
        region (str): Region name to process.
        big_data_storage_path (str): Path to the big data storage directory.
        repo_dir (str): Path to the repository directory.
        utm_crs (str): UTM Coordinate Reference System (CRS) to use. Default is "EPSG:32633".
        max_distance (int): Maximum distance in meters to buffer urban areas. Default is 20
        bbox_gdf (gpd.GeoDataFrame): Bounding box GeoDataFrame for clipping. Default is None.
    Returns:
        xr.DataArray: Binary urban areas mask.
    """
    filename = f"{big_data_storage_path}/corine/{region.lower()}/urban_areas.zarr"

    if not os.path.exists(filename):
        print(f"Urban areas file not found at {filename}. Creating urban areas mask...")
        
        print("   Reading OSM data...")
        # define the osm file paths
        osm_path = f"{big_data_storage_path}/osm/{region.lower()}"
        data_filename = f"{osm_path}/osm_gdf.parquet"
        
        # check if OSM data exists
        if not os.path.exists(data_filename):
            raise FileNotFoundError(f"OSM data file not found at {data_filename}. Please ensure OSM data is available before combining datasets.")

        # read the OSM data
        osm_gdf=gpd.read_parquet(data_filename)[["id","geometry","natural", "water", "boundary", "landuse", "building", "highway", "waterway", "leisure", "width"]]

        ##### read the corine data
        print("   Reading Corine Landcover data...")
        # get the corine folder
        corine_folder = [folder for folder in os.listdir(f"{repo_dir}/data/corine") if os.path.isdir(os.path.join(f"{repo_dir}/data/corine", folder))]
        # reverse sort by year
        corine_folder.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
        print("corine debug", corine_folder, os.listdir(f"{repo_dir}/data/corine"))
        corine_folder = corine_folder[0]

        # xarray read geotiff
        geotiff_path = [f for f in os.listdir(f"{repo_dir}/data/corine/{corine_folder}/DATA") if f.endswith(".tif")][0]
        corine_raster = rxr.open_rasterio(f"{repo_dir}/data/corine/{corine_folder}/DATA/{geotiff_path}", masked=True).squeeze("band", drop=True)

        # parse the xml to read the crs
        xml_paths = [f for f in os.listdir(f"{repo_dir}/data/corine/{corine_folder}/Metadata") if f.endswith(".xml")]
        xml_path = min(xml_paths, key=len) # get shortest xml path
        with open(f"{repo_dir}/data/corine/{corine_folder}/Metadata/{xml_path}", 'r') as xml_file:
            xml_content = xml_file.read()
            # find the crs string
            start_index = xml_content.find('EPSG:')
            # print("Start index of CRS in XML:", start_index)
            # get line
            end_index = xml_content.find('</gmd:code>', start_index)
            crs_string = xml_content[start_index:end_index].split('<')[0]

        # set crs
        corine_raster.rio.write_crs(crs_string, inplace=True)

        # reproject bbox to corine crs
        bbox_gdf_corine = bbox_gdf.to_crs(corine_raster.rio.crs)

        # clip to bbox
        corine_raster = corine_raster.rio.clip_box(minx=bbox_gdf_corine.geometry.bounds.minx.values[0],
                                                    miny=bbox_gdf_corine.geometry.bounds.miny.values[0],
                                                    maxx=bbox_gdf_corine.geometry.bounds.maxx.values[0],
                                                    maxy=bbox_gdf_corine.geometry.bounds.maxy.values[0])

        # reproject to utm crs
        corine_raster = corine_raster.rio.reproject(utm_crs)
        corine_raster_copy = corine_raster.copy()

        ##### mask water bodies
        print("   Masking water bodies...")
        water_gdf=osm_gdf[osm_gdf["natural"] == "water"]
        water_gdf = water_gdf[["geometry",  "water"]]

        # filter out waterways
        water_gdf = osm_gdf[(osm_gdf["water"].isin(["lake", "river", "canal"]) | osm_gdf["waterway"].isin(["river", "stream", "canal"]))]
        water_gdf = water_gdf[["id", "geometry", "water", "waterway", "width"]]

        # buffer dict for waterway types
        waterway_buffer = {
            'river': 400,
            'stream': 50,
            'canal': 50,
        }
        width_scaling = 40 

        # buffer waterway geometries
        water_gdf['buffer_width'] = water_gdf['waterway'].apply(lambda x: waterway_buffer.get(x, 5))  # Default to 5 if not 

        # update buffer width based on 'width' attribute if available
        def width_conversion(row):
            try:
                if pd.notnull(row['width']):
                    # convert width to float
                    width_val = float(row['width'])
                    return width_val * width_scaling
            except (ValueError, TypeError):
                pass
            return row['buffer_width']

        water_gdf['buffer_width'] = water_gdf.apply(width_conversion, axis=1)

        # ensure buffer_width is numeric
        water_gdf['buffer_width'] = pd.to_numeric(water_gdf['buffer_width'], errors='coerce').fillna(5)

        # convert to projected coordinates (UTM 33N)
        water_gdf = water_gdf.to_crs(utm_crs)

        # Create the buffer polygons with an appropriate width
        water_gdf["geometry"] = water_gdf.apply(lambda row: row['geometry'].buffer(row['buffer_width']), axis=1)

        # clip corine raster to water bodies
        corine_raster_waterclip = corine_raster.rio.clip(water_gdf.geometry, crs=utm_crs)

        ##### drop non-forest and non-agricultural classes
        print("   Filtering Corine Landcover data by rural classes...")
        # legend mapping
        legend_path = [f for f in os.listdir(f"{repo_dir}/data/corine/{corine_folder}/Legend") if f.endswith(".txt")][0]
        legend = pd.read_csv(f"{repo_dir}/data/corine/{corine_folder}/Legend/{legend_path}", sep=",", header=None)

        # rename last column
        column_count = legend.shape[1]
        legend = legend.rename(columns={column_count - 1: "Class_Name"})

        drop_classes = [
            "Non-irrigated arable land", 
            "Permanently irrigated land", 
            "Rice fields", 
            "Vineyards", 
            "Fruit trees and berry plantations", 
            "Olive groves", 
            "Annual crops associated with permanent crops", 
            "Complex cultivation patterns", 
            "Land principally occupied by agriculture with significant areas of natural vegetation", 
            "Agro-forestry areas", 
            "Broad-leaved forest", 
            "Coniferous forest", 
            "Mixed forest", 
            "Natural grasslands", 
            "Moors and heathland", 
            "Sclerophyllous vegetation", 
            "Transitional woodland-shrub", 
            "Beaches dunes sands", 
            "Inland marshes", 
            "Peat bogs", 
            "Salt marshes", 
            "Salines", 
            "Intertidal flats"
            ]

        # drop by id
        for class_name in drop_classes:
            class_df = legend[legend["Class_Name"].str.strip() == class_name.strip()]
            # if empty continue
            if class_df.empty:
                print(f"Class {class_name} not found in legend, skipping...")
                continue
            class_id = class_df.index[0]
            corine_raster = corine_raster.where(corine_raster != class_id, other=np.nan)
            
        #### create a buffer to include rural edges
        print("   Creating buffer around urban areas...")    
        # polygonize corine raster
        corine_vectors = xr_vectorize(
            corine_raster,
            attribute_col="land_cover",
            crs=utm_crs,
            dtype="float32",
        )
        
        # filter out nan
        corine_vectors = corine_vectors[~corine_vectors["land_cover"].isnull()]

        # add a buffer around corine non-NaN areas
        corine_vectors["geometry"] = corine_vectors.apply(lambda row: row['geometry'].buffer(max_distance), axis=1)
        
        # clip corine raster to buffered polygons
        corine_raster_copy = corine_raster_copy.rio.clip(corine_vectors.geometry, crs=utm_crs)

        #### merge water bodies back into urban areas
        print("   Merging water bodies back into urban areas mask...")

        # reproject waterclip to match corine_raster_copy grid
        corine_raster_waterclip = corine_raster_waterclip.rio.reproject_match(corine_raster_copy)

        # add waterclip where corine raster is nan
        corine_raster = corine_raster_copy.where(~corine_raster_copy.isnull(), other=corine_raster_waterclip)

        # convert corine raster to binary mask
        urban_mask = xr.where(corine_raster.isnull(), 0, 1)

        # clip to bbox again
        urban_mask = urban_mask.rio.clip_box(minx=bbox_gdf.geometry.bounds.minx.values[0],
                                            miny=bbox_gdf.geometry.bounds.miny.values[0],
                                            maxx=bbox_gdf.geometry.bounds.maxx.values[0],
                                            maxy=bbox_gdf.geometry.bounds.maxy.values[0])

        # urban_mask.plot()
        # plt.show()
        
        #### saving urban areas file
        print(f"   Saving urban areas file...")
        # save urban areas file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urban_mask.name = "urban_mask" # set name for saving
        urban_mask.to_zarr(filename)
        print(f"Urban areas file saved at {filename}.")
        return urban_mask

    else:
        print(f"Urban areas file already exists at {filename}. Loading existing file.")
        ds = xr.open_zarr(filename, consolidated=True)
        if "urban_mask" in ds:
            return ds["urban_mask"]
        else:
            # if no name (previous versions), return first data variable
            return ds[list(ds.data_vars)[0]]
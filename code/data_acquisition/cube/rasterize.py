######## Rasterization functions to convert vector data to an Xarray raster image cube #######

##### Import libraries ######
# system
import os

# data manipulation
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.api.extensions import register_dataframe_accessor
import xarray as xr
import rioxarray as rxr # needed for all rasterio operations on xarray even if not called directly
from rasterio.transform import from_bounds
from rasterio.features import rasterize


#### Extend geopandas DataFrame object by a conversion class to xarray raster dataarrays ######
def register_xarray_accessor():
    """
    Register the XarrayAccessor for GeoDataFrames.
    Call this function to enable .to_raster accessor on GeoDataFrames.
    """
    if "to_raster" in pd.DataFrame._accessors:
        # in case you need to change the accessor function, this removes the old one
        # you'd have to reassign old gdfs afterwards, however --> new_gdf = gpd.GeoDataFrame(old_gdf)
        pd.DataFrame._accessors.remove("to_raster")
        delattr(pd.DataFrame, "to_raster")

    @register_dataframe_accessor("to_raster")
    class XarrayAccessor:
        def __init__(self, gdf: gpd.GeoDataFrame):
            self._gdf = gdf

        def rasterize_gdf(self, bbox, image_size=1024, col=None, nodata=0):
            """
            Rasterizes the GeoDataFrame with rio rasterize.
            
            Parameters:
            -----------
            self (GeoDataFrame): 
                The GeoDataFrame to rasterize.
            bbox (tuple):
                The bounding box of the area to rasterize (xmin, ymin, xmax, ymax).
            image_size (int): 
                The size of the output raster image.
            col (str): 
                Optional column name to use for rasterization values.
            nodata (int/float):
                NoData value for the raster.
                
            Returns:
            --------
            numpy.ndarray
                The rasterized array
            """
            
            shapes = None
            if col:
                #remove none/nan values from the column
                self._gdf = self._gdf[self._gdf[col].notnull()]
                if not pd.api.types.is_numeric_dtype(self._gdf[col]):
                    #enumerate the unique values in the non-numeric column
                    df_mapping = {value: idx+1 for idx, value in enumerate(self._gdf[col].unique())}  
                    self._gdf.loc[:,f"{col}_int"] = self._gdf[col].map(df_mapping)
                    shapes = ((geom, value) for geom, value in zip(self._gdf.geometry, self._gdf[f"{col}_int"]))
                else:
                    shapes = ((geom, value) for geom, value in zip(self._gdf.geometry, self._gdf[col]))
            else:
                shapes = self._gdf.geometry.values
                
            xmin, ymin, xmax, ymax = bbox

            return rasterize(
                shapes,
                out_shape=(image_size, image_size),
                fill=nodata,
                transform=from_bounds(xmin, ymin, xmax, ymax, image_size, image_size),
                all_touched=True,
                dtype=np.float32
            )

        def to_xr_dataarray(self, bbox, image_size, x_coords, y_coords, name="data", long_name=None, description=None, 
                        mapping_col=None, output_path=None, crs="EPSG:4326", x_dim="lon", y_dim="lat", units="1", nodata=0):
        
            """
            Process the GeoDataFrame, convert to a xarray DataArray, and optionally save to a zarr file. 
            
            Parameters:
            -----------
            self (GeoDataFrame):
                The GeoDataFrame to convert
            bbox (tuple):
                The bounding box of the area to rasterize (xmin, ymin, xmax, ymax)
            image_size (int):
                The size of the output raster image
            x_coords (numpy.ndarray):
                The x coordinate space
            y_coords (numpy.ndarray):
                The y coordinate space
            name (str):
                Name for the DataArray
            long_name (str, optional):
                Long name for the DataArray
            description (str, optional):
                Description for the DataArray
            mapping_col (str, optional):
                Column name for mapping categorical values to integers
            output_path (str, optional):
                Path to save the output zarr file
            crs (str):
                Coordinate reference system (default: "EPSG:4326")
            x_dim (str):
                Name of the x dimension (default: "lon")
            y_dim (str):
                Name of the y dimension (default: "lat")
            units (str):
                Units of the data (default: "1")
            nodata (int/float):
                NoData value for the raster (default: 0)
                
            Returns:
            --------
            xarray.DataArray
                The xarray DataArray with the rasterized data
            """
        
            xmin, ymin, xmax, ymax = bbox
            
            shapes = None
            df_mapping = None
            if mapping_col:
                # remove none/nan values from the column
                self._gdf = self._gdf[self._gdf[mapping_col].notnull()]
                if not pd.api.types.is_numeric_dtype(self._gdf[mapping_col]):
                    #enumerate the unique values in the non-numeric column
                    df_mapping = {value: idx+1 for idx, value in enumerate(self._gdf[mapping_col].unique())}  
                    self._gdf.loc[:,f"{mapping_col}_int"] = self._gdf[mapping_col].map(df_mapping)
                    shapes = ((geom, value) for geom, value in zip(self._gdf.geometry, self._gdf[f"{mapping_col}_int"]))
                else:
                    shapes = ((geom, value) for geom, value in zip(self._gdf.geometry, self._gdf[mapping_col]))
            else:
                shapes = self._gdf.geometry.values
                
            

            raster = rasterize(
                shapes,
                out_shape=(image_size, image_size),
                fill=nodata,
                transform=from_bounds(xmin, ymin, xmax, ymax, image_size, image_size),
                all_touched=True,
                dtype=np.float32
            )

            #y_coords = np.flipud(y_coords) # flip y coordinates to match the raster
            coords = {y_dim: y_coords, x_dim: x_coords}
            da = xr.DataArray(raster, dims=[y_dim, x_dim], coords=coords)
            da = da.rio.write_crs(crs)
            da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            da = da.rio.set_nodata(nodata)

            da.name = name
            da.attrs.update({
                "long_name": long_name or name,
                "description": description or f"Rasterized {name} data",
                "units": units,
                "spatial_ref": crs,
                "crs": crs,
            })
        
            if df_mapping:
                da.attrs[f"{name}_mapping"] = json.dumps(df_mapping)

            if output_path:
                da.to_zarr(output_path, mode="w", consolidated=True, compute=True)

            return da


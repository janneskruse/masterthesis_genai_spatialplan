"""
==============================================================
Functions to vectorize raster data into vector data (e.g. GeoDataFrames)

Mostly adapted from 
https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Rasterise_vectorise.html
Huge Thank you! 
================================================================
"""
## Import libraries

# data manipulation
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.transform import from_bounds
import xarray as xr
import odc.geo.xr  # adds  `.odc.x` attributes to xarray objects.
from shapely.geometry import shape
import dask

# dask friendliness for large rasters
dask.config.set({"array.slicing.split_large_chunks": True})


# this is taken from https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Rasterise_vectorise.html
def add_geobox(ds: xr.Dataset, crs=None):
    """
    Ensure that an xarray DataArray has a GeoBox and .odc.* accessor
    using `odc.geo`.

    If `ds` is missing a Coordinate Reference System (CRS), this can be
    supplied using the `crs` param.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input xarray object that needs to be checked for spatial
        information.
    crs : str, optional
        Coordinate Reference System (CRS) information for the input `ds`
        array. If `ds` already has a CRS, then `crs` is not required.
        Default is None.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input xarray object with added `.odc.x` attributes to access
        spatial information.

    """
    # If a CRS is not found, use custom provided CRS
    if ds.odc.crs is None and crs is not None:
        ds = ds.odc.assign_crs(crs)
    elif ds.odc.crs is None and crs is None:
        raise ValueError(
            "Unable to determine `ds`'s coordinate "
            "reference system (CRS). Please provide a "
            "CRS using the `crs` parameter "
            "(e.g. `crs='EPSG:3577'`)."
        )

    return ds

# this is also taken from https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Rasterise_vectorise.html
def xr_vectorize(
    da: xr.DataArray,
    attribute_col=None,
    crs=None,
    dtype="float32",
    output_path=None,
    verbose=True,
    **rasterio_kwargs,
):
    """
    Vectorises a raster ``xarray.DataArray`` into a vector
    ``geopandas.GeoDataFrame``.

    Parameters
    ----------
    da : xarray.DataArray
        The input ``xarray.DataArray`` data to vectorise.
    attribute_col : str, optional
        Name of the attribute column in the resulting
        ``geopandas.GeoDataFrame``. Values from ``da`` converted
        to polygons will be assigned to this column. If None,
        the column name will default to 'attribute'.
    crs : str or CRS object, optional
        If ``da``'s coordinate reference system (CRS) cannot be
        determined, provide a CRS using this parameter.
        (e.g. 'EPSG:3577').
    dtype : str, optional
         Data type  of  must be one of int16, int32, uint8, uint16,
         or float32
    output_path : string, optional
        Provide an optional string file path to export the vectorised
        data to file. Supports any vector file formats supported by
        ``geopandas.GeoDataFrame.to_file()``.
    verbose : bool, optional
        Print debugging messages. Default True.
    **rasterio_kwargs :
        A set of keyword arguments to ``rasterio.features.shapes``.
        Can include `mask` and `connectivity`.

    Returns
    -------
    gdf : geopandas.GeoDataFrame

    """

    # Add GeoBox and odc.* accessor to array using `odc-geo`
    da = add_geobox(da, crs)

    # Run the vectorizing function
    vectors = rio.features.shapes(
        source=da.data.astype(dtype), transform=da.odc.transform, **rasterio_kwargs
    )

    # Convert the generator into a list
    vectors = list(vectors)

    # Extract the polygon coordinates and values from the list
    polygons = [polygon for polygon, value in vectors]
    values = [value for polygon, value in vectors]

    # Convert polygon coordinates into polygon shapes
    polygons = [shape(polygon) for polygon in polygons]

    # Create a geopandas dataframe populated with the polygon shapes
    attribute_name = attribute_col if attribute_col is not None else "attribute"
    gdf = gpd.GeoDataFrame(data={attribute_name: values}, geometry=polygons, crs=da.odc.crs)

    # If a file path is supplied, export to file
    if output_path is not None:
        if verbose:
            print(f"Exporting vector data to {output_path}")
        gdf.to_file(output_path)

    return gdf


def vectorize_array(
    array: np.ndarray,
    transform: rio.transform.Affine | None = None,
    crs: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    attribute_col: str | None = None,
    dtype: str = "int32",
    mask_background: bool = True,
    connectivity: int = 4,
    output_path: str | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Vectorise a 2D numpy array (e.g. a labeled array from
    ``scipy.ndimage.label``) into a ``geopandas.GeoDataFrame``.

    Either ``transform`` or ``bounds`` must be provided so that pixel
    coordinates can be mapped to geographic / projected coordinates.
    If neither is given the function falls back to an identity
    transform (pixel coordinates).

    Parameters
    ----------
    array : numpy.ndarray
        2D array to vectorise. Typically an integer labeled array where
        each unique value represents a distinct feature / region.
    transform : rasterio.transform.Affine, optional
        Affine transform that maps pixel coordinates to CRS coordinates.
        Takes precedence over ``bounds`` if both are supplied.
    crs : str or CRS object, optional
        Coordinate reference system for the output GeoDataFrame
        (e.g. ``'EPSG:4326'``).
    bounds : tuple of (xmin, ymin, xmax, ymax), optional
        Geographic bounds used to derive the affine transform when
        ``transform`` is not provided.
    attribute_col : str, optional
        Name of the attribute column in the resulting GeoDataFrame.
        Defaults to ``'label'``.
    dtype : str, optional
        Data type to cast the array to before vectorising. Must be one
        of int16, int32, uint8, uint16, or float32. Default ``'int32'``.
    mask_background : bool, optional
        If True, pixels with value 0 are treated as background and
        excluded from the output. Default True.
    connectivity : int, optional
        Pixel connectivity (4 or 8) passed to
        ``rasterio.features.shapes``. Default 4.
    output_path : str, optional
        Optional file path to export the vectorised data.
    verbose : bool, optional
        Print debugging messages. Default True.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with one row per contiguous region.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")

    # Build the affine transform
    if transform is not None:
        affine = transform
    elif bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        height, width = array.shape
        affine = from_bounds(xmin, ymin, xmax, ymax, width, height)
    else:
        # identity transform – output will be in pixel coordinates
        affine = rio.transform.Affine(1, 0, 0, 0, -1, array.shape[0])

    # Optionally mask background (0-valued pixels)
    mask = array != 0 if mask_background else None

    # Run rasterio vectorisation
    vectors = rio.features.shapes(
        source=array.astype(dtype),
        mask=mask,
        transform=affine,
        connectivity=connectivity,
    )

    # Unpack polygons and values
    polygons, values = zip(*vectors) if vectors else ([], [])
    polygons = [shape(p) for p in polygons]

    # Build GeoDataFrame
    attribute_name = attribute_col if attribute_col is not None else "label"
    gdf = gpd.GeoDataFrame(
        data={attribute_name: list(values)},
        geometry=polygons,
        crs=crs,
    )

    # Export if requested
    if output_path is not None:
        if verbose:
            print(f"Exporting vector data to {output_path}")
        gdf.to_file(output_path)

    return gdf
## Import libraries
# data manipulation
import rasterio as rio
from rasterio.enums import Resampling
import rioxarray as rxr # adds rioxarray capabilities to xarray
import xarray as xr

# helper: reproject/resample every variable in a Dataset
def reproject_ds(ds: xr.Dataset, template_da: xr.DataArray, resampling: Resampling) -> xr.Dataset:
    out_vars = {}
    for name, da in ds.data_vars.items():
        # carry nodata forward
        if da.rio.nodata is None and getattr(da, "_FillValue", None) is not None:
            da = da.rio.write_nodata(da._FillValue, encoded=True)
        out_vars[name] = da.rio.reproject_match(template_da, resampling=resampling)
    out = xr.Dataset(out_vars)
    # copy non-spatial coords
    for c in ds.coords:
        if c not in ("x", "y"):
            out = out.assign_coords({c: ds[c]})
    out.attrs = ds.attrs
    return out
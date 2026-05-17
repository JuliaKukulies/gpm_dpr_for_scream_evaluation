'''
This script regrids GPM DPR overpasses n the global MERGIR (4KM) regular grid.

Output files have dim: (lat), (lon), 176 (bins)

Created by: Julia Kukulies, February 18, 2026

'''

import os
import time
import json
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import h5py
print(h5py.__file__, flush = True)
from pathlib import Path 
import sys 
import scipy
from datetime import datetime, timedelta

# special: pyflextrkr and pansat libraries
import pansat
from pansat.environment import get_index
from pansat import TimeRange
from pansat.geometry import LonLatRect 
from pansat.granule import merge_granules
from pansat.catalog import Index
from pansat.products.satellite.gpm import l2a_gpm_dpr
from pyflextrkr.ft_regrid_func import make_grid4regridder, make_weight_file, get_latlon_bounds_1d, get_latlon_bounds_2d

import warnings
warnings.filterwarnings("ignore")


### set encoding for variables

encoding = {
    "reflectivity": {
        "_FillValue": None,
        "dtype": "float32"
    },
    "column_max_reflectivity": {
        "_FillValue": None,
        "dtype": "float32"
    },
    "surface_precip": {
        "_FillValue": None,
        "dtype": "float32"
    },
    "freezing_level": {
        "_FillValue": None,
        "dtype": "float32"
    }
}




def get_swath_edge(regridded):
    '''
    Function to get the connected pixels for the edges of the satellite swath on regridded 2D field. 

    Parameters:
        regridded (xr.DataArray): a regridded 2D field that has non-zero values everywhere that falls underneath the satellite swath 

    Returns:
        edge (xr.DataArray): binary mask that defines the edge of the swath 
    '''
    from scipy.ndimage import binary_dilation, binary_erosion
    valid_mask = regridded > 0
    structure = np.ones((3,3), dtype=bool)
    #edge = mask & binary_dilation(~mask, structure=structure)
    # Erode the mask by 1 pixel, then subtract to get the swath edge
    eroded = binary_erosion(valid_mask.values).astype(int)
    # Edge = original mask minus eroded mask (edge pixels are 1, interior/exterior are 0)
    swath_edge_mask = valid_mask.copy(data=valid_mask.values - eroded)
    return swath_edge_mask

##### Configuration #####
config = dict()
config['regrid_method'] = 'bilinear'
config['regrid_input'] = True 
target_grid = Path('/glade/derecho/scratch/kukulies/gpm_dpr/target_grid.nc')
config['gridfile_dst'] = target_grid
config['x_coordname_src'] =  'longitude'
config['y_coordname_src'] =  'latitude'
config['x_coordname_dst'] =  'lon'
config['y_coordname_dst'] =  'lat'
config['write_native'] =  False

### Choose month and day, there are granule (15-16 granules per day) ###
month = str(sys.argv[1]).zfill(2)
day = str(sys.argv[2]).zfill(2)

start_time = "2020-" + month + "-"+day+"T00:00:00"
end_time = "2020-" +month+ "-"+day+"T23:00:00"
index = get_index(l2a_gpm_dpr)

# location of granule data files
path = Path('/glade/derecho/scratch/kukulies/gpm_dpr/download/') /  str(month) 
# where to save the regridded data 
outdir= Path('/glade/derecho/scratch/kukulies/gpm_dpr') / str('2020' + month )

# find granules that fall in a specific period
granules   = index.find(TimeRange(start_time, end_time))
granules = merge_granules(granules)
print(len(granules), 'found for date:', month, day, flush = True)
assert len(granules) <= 16  


clean_times = np.array(())
start_times = np.array(())
end_times = np.array(())

# loop through all granules
for granule_idx in np.arange(len(granules)):
    # open granule ds 
    granule = granules[granule_idx]
    start_time = time.time()
    granule.file_record.local_path = path / granule.file_record.filename
    granule_ds = granule.open()[{"frequencies": -2}]
    granule_name = str(granule)[18:77]
    lats = granule_ds.latitude
    lons = granule_ds.longitude
    print('Processing:    ', granule_name, flush = True)

    # Average granule time
    time_avg = granule_ds.scan_time.mean().compute()
    timestr = str(time_avg.dt.round('1h').data)[0:-10]
    timestr_day = pd.to_datetime(time_avg.values).strftime("%Y-%m-%d")
    time_stamp = pd.to_datetime(time_avg.dt.round('1h').data) 
    overpass_start = pd.to_datetime(granule_ds.scan_time[0].compute().data)
    overpass_end = pd.to_datetime(granule_ds.scan_time[-1].compute().data)
    heights = granule_ds.height.squeeze().mean(dim= ['scans', 'rays' ] )

    # Output filename (Zarr)
    outfilename = outdir / f"GPM_DPR_reflectivity_regridded_{timestr_day}.zarr"
    outfilename_tmp = outdir / f"GPM_DPR_reflectivity_regridded_{timestr_day}_tmp.zarr"

    # --- SAVE SOURCE GRID ---
    fname = outdir / f"{granule_name}_{timestr}_src-grid.nc"
 
    if not fname.is_file(): 
        granule_ds.to_netcdf(fname) 
    else: 
        print(fname, "not saving source file, because it already exists.", flush = True)

    gridfile_src = fname 

    # --- DEFINE SOURCE AND DESTINATION GRID ---
    grid_src, grid_dst = make_grid4regridder(gridfile_src, config)
    print("Got grid info, starting regridding...", flush = True )

    # --- CREATE REGRIDDER (needs to be done for every overpass)---
    print("Creating new regrid weights...")
    regridder_fname = outdir / f"{granule_name}_{timestr}_regridder.nc"
    regridder = xe.Regridder(grid_src, grid_dst, method="bilinear", reuse_weights = True,unmapped_to_nan=True, filename = regridder_fname)

    # --- MASK ORIGINAL REFLECTIVITY ---
    refl_orig = granule_ds.reflectivity  #.fillna(-9999)
    refl_orig = xr.where((refl_orig < -1000) | (~np.isfinite(refl_orig)),np.nan,refl_orig)

    refl_orig = refl_orig.assign_coords(
        lat=granule_ds.latitude,
        lon=granule_ds.longitude)

    # Get swath edge mask
    field_2d = granule_ds.height[:,:,150]
    regridded_2d = regridder(field_2d)
    edge = get_swath_edge(regridded_2d) 

    # Swath geometry mask
    # 1 = inside swath
    # 0 = outside swath
    # ---------------------------------------------------------

    swath_mask = xr.where(np.isfinite(granule_ds.surface_precip), 1.0, np.nan)
    swath_mask = swath_mask.assign_coords(
            lat=granule_ds.latitude,
                lon=granule_ds.longitude)
    swath_mask_regridded = regridder(swath_mask)
    inside_swath = np.isfinite(swath_mask_regridded)
    outside_swath = ~inside_swath 

    # --- Transpose for regridding (bins first) ---
    refl_trans = refl_orig.transpose("bins", "scans", "rays")
    # --- Regrid reflectivity and swath mask ---
    refl_regridded = regridder(refl_trans)
    # --- Transpose to final dims ---
    refl_regridded = refl_regridded.transpose("lat", "lon", "bins")
    # compute composite (column max) reflectivity (2D variable) 
    comp_refl = refl_regridded.max(dim = 'bins')

    # fix precision
    refl_regridded = refl_regridded.astype("float32")
    comp_refl = comp_refl.astype("float32")

    # 2D variables: Get freezing level and surface precipitation as well
    freezing_level = granule_ds.freezing_level
    surface_precip = granule_ds.surface_precip
    freeze_lev_regridded = regridder(freezing_level)
    surface_precip_regridded = regridder(surface_precip)
    surface_precip_regridded = surface_precip_regridded.astype("float32")
    freeze_lev_regridded = freeze_lev_regridded.astype("float32")

    # Note: after regridding, we should have NaN values inside the swath and 0 values outside of the swath
    # Therefore, we apply a swath mask 
    # We want to get:  np.nan inside the swath, physical 0 values inside the swat which exist e.g. for precip, and -9999 outside of the swath  
    refl_regridded = refl_regridded.where(~outside_swath, -9999.0)
    comp_refl = comp_refl.where(~outside_swath, -9999.0)
    freeze_lev_regridded = freeze_lev_regridded.where(~outside_swath, -9999.0)
    surface_precip_regridded = surface_precip_regridded.where(~outside_swath, -9999.0)
    
    
    # --- Create xr Dataset  ---
    # the radar relectivity is at 3D
    # surface precip, freezing level and the swath edge is saved as 2D variable 
    x_coord = xr.DataArray(grid_dst['lon'], dims=('lon'))
    y_coord = xr.DataArray(grid_dst['lat'], dims=('lat'))
    ds_out = xr.Dataset(
        data_vars={'reflectivity': ([ 'lat', 'lon', 'height'], refl_regridded.data),
                   'swath_edge': (['lat', 'lon'], edge.data),
                   'swath_mask': (['lat', 'lon'], inside_swath.data.astype("int8")),
                   'freezing_level': (['lat', 'lon'], freeze_lev_regridded.data),
                   'column_max_reflectivity': (['lat', 'lon'], comp_refl.data),
                   'surface_precip': (['lat', 'lon'], surface_precip_regridded.data)
                   },
        coords={'lat': y_coord, 'lon': x_coord, 'height': heights.data })
    
    # expand by time and add time bounds (first and last scan time)
    ds_out = ds_out.expand_dims(time=[time_stamp])
    clean_times = np.append(clean_times, time_stamp)
    start_times = np.append(start_times, overpass_start)
    end_times = np.append(end_times, overpass_end) 
    print(str(time_stamp),'   for ' ,month, day, flush = True)

    ds_out = ds_out.assign_coords(
    time=("time", np.array([time_stamp])),  # explicitly make it a coordinate
    scan_time_start=("time",np.array([overpass_start])),
        scan_time_end=("time", np.array([overpass_end])) )

    print('output data dimensions: ', ds_out.dims, flush = True)

    # chunk dataset
    granule_ds.close()
    ds_out = ds_out.chunk({"time": -1, "lat": 512, "lon": 512, "height": -1})

    if granule_idx == 0:
        ds_out.to_zarr(store=outfilename_tmp, mode="w", consolidated=False, encoding = encoding)
        print(outfilename, "written as Zarr.", flush=True)
    else:
        ds_out.to_zarr(store=outfilename_tmp, mode="a", append_dim="time", consolidated=False, encoding = None)
        print(outfilename, "appended to existing zarr file.", flush=True)

    del ds_out
    del refl_regridded
    del comp_refl
    del edge
        
    elapsed = time.time() - start_time
    print(f"Granule processed in {elapsed:.2f} seconds.\n", flush = True)

print('Fixing time dimension in Zarr storage.', flush = True)
## post processing to fix weird bug in zarr where the times appended are not encoded correctly
ds = xr.open_zarr(outfilename_tmp)

ds = ds.assign_coords(
    time=("time", clean_times),
    scan_time_start=("time", start_times),
    scan_time_end=("time", end_times),)


ds = ds.sortby("time")
ds.to_zarr(outfilename, mode="w", consolidated=True, encoding = encoding)
print('Final file saved and processed:', outfilename, flush = True)


import shutil
# Remove temporary zarr store
if outfilename_tmp.exists():
    shutil.rmtree(outfilename_tmp)
    print(f"Temporary file removed: {outfilename_tmp}", flush=True)




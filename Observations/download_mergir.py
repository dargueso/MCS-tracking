#!/usr/bin/env python
"""
Download MERGIR (NCEP/CPC merged IR, 4 km, half-hourly) brightness temperature
over the tracking region and write one netCDF per month.

Here the OPeNDAP subsetter is used rather than whole files: measured on this
machine a region subset is ~139 kB in ~1 s, against 27 MB in ~20 s for the full
global file, the opposite of what was found for IMERG.

Each remote file holds one hour, i.e. two half-hourly steps.

Output: {cfg.path_mergir}/MERGIR_30MIN_TB_YYYY-MM.nc   (Tb, K, native 4 km)

Run:  python download_mergir.py
Safe to re-run: completed months are skipped.
"""

import os
import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import obs_config as cfg
import obs_download_utils as util

# Index bounds on the MERGIR grid are constant for the whole collection, so
# they are worked out once from the first file and reused.
_IDX = {}


def opendap_url(tstamp, constraint=""):
    name = f"merg_{tstamp.strftime('%Y%m%d%H')}_4km-pixel.nc4"
    return (f"{cfg.mergir_opendap}/{tstamp.strftime('%Y')}/"
            f"{tstamp.dayofyear:03d}/{name}.nc4{constraint}"), name


def grid_indices(tstamp):
    """Index ranges covering the region, from the coordinate arrays."""
    if _IDX:
        return _IDX
    url, _ = opendap_url(tstamp, "?lat,lon")
    tmp = os.path.join(cfg.path_scratch, "mergir_coords.nc4")
    if not util.fetch(url, tmp):
        raise RuntimeError("could not read the MERGIR coordinate arrays")
    with xr.open_dataset(tmp) as dset:
        lat, lon = dset.lat.values, dset.lon.values
    os.remove(tmp)
    lat0, lat1, lon0, lon1 = util.subset_bounds()
    jj = np.where((lat >= lat0) & (lat <= lat1))[0]
    ii = np.where((lon >= lon0) & (lon <= lon1))[0]
    _IDX.update(j0=int(jj[0]), j1=int(jj[-1]), i0=int(ii[0]), i1=int(ii[-1]))
    logging.info("MERGIR subset: lat[%d:%d] lon[%d:%d] (%d x %d points)",
                 _IDX["j0"], _IDX["j1"], _IDX["i0"], _IDX["i1"], jj.size, ii.size)
    return _IDX


def get_hour(tstamp, idx):
    """Download the region subset for one hourly file (two half-hour steps)."""
    con = (f"?Tb[0:1][{idx['j0']}:{idx['j1']}][{idx['i0']}:{idx['i1']}],"
           f"lat[{idx['j0']}:{idx['j1']}],lon[{idx['i0']}:{idx['i1']}],time[0:1]")
    url, name = opendap_url(tstamp, con)
    tmp = os.path.join(cfg.path_scratch, f"{name}.{os.getpid()}.sub")
    if not util.fetch(url, tmp):
        return None
    try:
        with xr.open_dataset(tmp) as dset:
            return dset.Tb.load()
    except Exception as err:
        logging.error("could not read %s (%s)", name, err)
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def do_month(year, month):
    fileout = f"{cfg.path_mergir}/MERGIR_30MIN_TB_{year}-{month:02d}.nc"
    if util.already_done(fileout):
        logging.info("%s already there, skipping", os.path.basename(fileout))
        return

    stamps = []
    for day in util.daterange(year, month):
        stamps.extend(pd.date_range(day, periods=24, freq="H"))
    idx = grid_indices(stamps[0])
    logging.info("%d-%02d: %d hourly files", year, month, len(stamps))

    util.warm_up(opendap_url(stamps[0], "?lat")[0])

    slices = Parallel(n_jobs=cfg.nworkers, backend="threading")(
        delayed(get_hour)(t, idx) for t in stamps)

    good = [s for s in slices if s is not None]
    if len(good) != len(stamps):
        logging.warning("%d-%02d: %d/%d files retrieved",
                        year, month, len(good), len(stamps))
    if not good:
        logging.error("%d-%02d: nothing retrieved, no file written", year, month)
        return

    tb = xr.concat(good, dim="time").sortby("time")
    tb.name = "TB"
    tb.attrs.update(units="K", long_name="MERGIR cloud-top brightness temperature",
                    source="GPM_MERGIR.1")
    dset = tb.to_dataset()
    lat0, lat1, lon0, lon1 = util.subset_bounds()
    dset.attrs.update(
        title="MERGIR half-hourly brightness temperature over the tracking region",
        source=cfg.mergir_opendap, region=cfg.reg,
        domain=f"lat {lat0}..{lat1}, lon {lon0}..{lon1}",
        files=f"{len(good)}/{len(stamps)}")
    tmp = f"{fileout}.tmp"
    dset.to_netcdf(tmp, encoding={"TB": {"zlib": True, "complevel": 5}})
    os.replace(tmp, fileout)
    logging.info("wrote %s", os.path.basename(fileout))


def main():
    util.start_logger("download_mergir")
    os.makedirs(cfg.path_mergir, exist_ok=True)
    os.makedirs(cfg.path_scratch, exist_ok=True)
    for year, month in util.months_to_do():
        do_month(year, month)


if __name__ == "__main__":
    main()

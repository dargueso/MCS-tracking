#!/usr/bin/env python
"""
Download GPM IMERG Final Run V07 half-hourly precipitation over the tracking
region and write one netCDF per month.

Whole granules are downloaded and subsetted locally rather than going through
OPeNDAP: measured on this machine, a granule takes ~1.3-3 s (7.7 MB) while the
equivalent OPeNDAP region request takes ~18 s, because the server has to
decompress the global field to extract the box.

Output: {cfg.path_imerg}/IMERG_30MIN_PR_YYYY-MM.nc   (precipitation, mm/h)

Run:  python download_imerg.py
Safe to re-run: completed months are skipped, partial files are never kept.
"""

import os
import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

import obs_config as cfg
import obs_download_utils as util


def granule_url(tstamp):
    """URL of the half-hourly granule starting at tstamp."""
    start = tstamp.strftime("%H%M%S")
    end = (tstamp + np.timedelta64(29, "m") + np.timedelta64(59, "s")).strftime("%H%M%S")
    minutes = f"{tstamp.hour * 60 + tstamp.minute:04d}"
    name = (f"3B-HHR.MS.MRG.3IMERG.{tstamp.strftime('%Y%m%d')}"
            f"-S{start}-E{end}.{minutes}.{cfg.imerg_version}.HDF5")
    return (f"{cfg.imerg_root}/{tstamp.strftime('%Y')}/"
            f"{tstamp.dayofyear:03d}/{name}"), name


def get_granule(tstamp, lat0, lat1, lon0, lon1):
    """Download one granule, return the regional precipitation slice."""
    url, name = granule_url(tstamp)
    tmp = os.path.join(cfg.path_scratch, name)
    if not util.fetch(url, tmp):
        return None
    try:
        with xr.open_dataset(tmp, group="Grid") as dset:
            # precipitation comes as (time, lon, lat); transpose to the usual order
            prec = (dset.precipitation
                    .sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
                    .transpose("time", "lat", "lon")
                    .load())
        prec["time"] = [np.datetime64(tstamp.to_datetime64())]
        return prec
    except Exception as err:
        logging.error("could not read %s (%s)", name, err)
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def do_month(year, month):
    fileout = f"{cfg.path_imerg}/IMERG_30MIN_PR_{year}-{month:02d}.nc"
    if util.already_done(fileout):
        logging.info("%s already there, skipping", os.path.basename(fileout))
        return
    lat0, lat1, lon0, lon1 = util.subset_bounds()

    stamps = []
    for day in util.daterange(year, month):
        stamps.extend(pd.date_range(day, periods=48, freq="30min"))
    logging.info("%d-%02d: %d granules", year, month, len(stamps))

    util.warm_up(granule_url(stamps[0])[0])

    slices = Parallel(n_jobs=cfg.nworkers, backend="threading")(
        delayed(get_granule)(t, lat0, lat1, lon0, lon1) for t in stamps)

    good = [s for s in slices if s is not None]
    if len(good) != len(stamps):
        logging.warning("%d-%02d: %d/%d granules retrieved",
                        year, month, len(good), len(stamps))
    if not good:
        logging.error("%d-%02d: nothing retrieved, no file written", year, month)
        return

    prec = xr.concat(good, dim="time").sortby("time")
    prec.name = "PR"
    prec.attrs.update(units="mm h-1", long_name="IMERG precipitation rate",
                      source="GPM_3IMERGHH.07")
    dset = prec.to_dataset()
    dset.attrs.update(
        title="IMERG half-hourly precipitation over the tracking region",
        source=cfg.imerg_root, region=cfg.reg,
        domain=f"lat {lat0}..{lat1}, lon {lon0}..{lon1}",
        granules=f"{len(good)}/{len(stamps)}")
    tmp = f"{fileout}.tmp"
    dset.to_netcdf(tmp, encoding={"PR": {"zlib": True, "complevel": 5}})
    os.replace(tmp, fileout)
    logging.info("wrote %s", os.path.basename(fileout))


def main():
    util.start_logger("download_imerg")
    os.makedirs(cfg.path_imerg, exist_ok=True)
    os.makedirs(cfg.path_scratch, exist_ok=True)
    for year, month in util.months_to_do():
        do_month(year, month)


if __name__ == "__main__":
    main()

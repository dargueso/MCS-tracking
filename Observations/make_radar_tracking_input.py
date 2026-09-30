#!/usr/bin/env python
"""
Radar-based inputs for a symmetric storm tracking on the 0.1 deg grid.

The observed storms of the satellite comparison are defined by IMERG rain,
whose hourly intensities at a point are far from the gauges. To define the
observed storms with a rain field of the model's own quality, EURADCLIM (on
the 2 km model grid, quality-masked) is block-averaged onto the 0.1 deg
tracker grid, and the coarsened model rain is cut to exactly the same
coverage, cell by cell and hour by hour. Tracking both with MERGIR / the
model Tb then gives storms of the same definition on both sides, over the
same area and hours:

    RADCOV_01H_RAIN_YYYY-MM.nc      in {cfg.path_track}: EURADCLIM at 0.1 deg,
                                    NaN outside the radar coverage
    MODRADCOV_01H_RAIN_YYYY-MM.nc   in {cfg.path_modcoarse}/<run>: MOD_01H_RAIN
                                    with the same NaN pattern

A 0.1 deg cell is valid in an hour when at least half of its 2 km cells are
valid radar (quality mask and that hour's availability). The tracker turns
NaN rain into no rain, so nothing outside the coverage can become a storm.

    python make_radar_tracking_input.py
"""

import os
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import obs_config as cfg
import radar_config as rcfg
from make_model_tracking_input import target_grid, cell_mapping

NWORKERS = 8
MIN_FRAC = 0.5
_G = {}


def coarsen_valid(field, flat_idx, inside, ncell, nlat, nlon, nfull):
    """Block mean of the valid 2 km cells; NaN where fewer than MIN_FRAC are valid."""
    out = np.full((field.shape[0], nlat, nlon), np.nan, "float32")
    for it in range(field.shape[0]):
        vals = field[it].ravel()[inside].astype("float64")
        good = np.isfinite(vals)
        total = np.bincount(flat_idx[good], weights=vals[good], minlength=ncell)
        n = np.bincount(flat_idx[good], minlength=ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[it] = np.where(n >= MIN_FRAC * nfull, total / np.maximum(n, 1), np.nan).reshape(nlat, nlon)
    return out


def write(fileout, data, times, lat, lon, attrs, gattrs):
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = xr.Dataset({"RAIN": (["time", "y", "x"], data)},
                    coords={"time": times, "lat": (["y", "x"], lat2d.astype("float32")),
                            "lon": (["y", "x"], lon2d.astype("float32"))})
    ds.RAIN.attrs.update(attrs)
    ds.attrs.update(gattrs)
    ds.to_netcdf(f"{fileout}.tmp", encoding={"RAIN": {"zlib": True, "complevel": 3}})
    os.replace(f"{fileout}.tmp", fileout)


def do_month(tag):
    fin = f"{rcfg.path_rad_grid}/RAD_01H_RAIN_{tag}.nc"
    fmod = f"{cfg.path_modcoarse}/{cfg.model_runs['pres']}/MOD_01H_RAIN_{tag}.nc"
    out_r = f"{cfg.path_track}/RADCOV_01H_RAIN_{tag}.nc"
    out_m = f"{cfg.path_modcoarse}/{cfg.model_runs['pres']}/MODRADCOV_01H_RAIN_{tag}.nc"
    if not (os.path.exists(fin) and os.path.exists(fmod)):
        return tag, "missing input"
    if os.path.exists(out_r) and os.path.exists(out_m):
        return tag, "done already"
    g = _G
    with xr.open_dataset(fin) as d:
        rain = np.where(g["mask"][None], d.RAIN.values, np.nan)
        times = pd.to_datetime(d.time.values)
    rad = coarsen_valid(rain, g["flat"], g["inside"], g["ncell"], g["nlat"], g["nlon"], g["nfull"])
    with xr.open_dataset(fmod) as d:
        mt = pd.to_datetime(d.time.values).floor("h")
        mod = d.RAIN.values
        mattrs = dict(d.RAIN.attrs)
    pos = mt.get_indexer(times)
    if (pos < 0).any():
        return tag, f"{(pos < 0).sum()} radar hours not in the model file"
    mod = np.where(np.isfinite(rad), mod[pos], np.nan).astype("float32")
    cov = 100 * np.isfinite(rad).mean()
    write(out_r, rad, times, g["lat"], g["lon"],
          {"units": "mm h-1", "long_name": "EURADCLIM hourly rain, block averaged to the 0.1 deg tracker grid",
           "source": fin, "rain_hour_window": "clock hour HH:00-HH+1:00 (EURADCLIM accumulation)"},
          {"title": "EURADCLIM on the 0.1 deg tracker grid, quality-masked radar coverage only",
           "coverage": f"valid where >= {MIN_FRAC:.0%} of the 2 km cells are valid radar (mask_s1 and that hour)",
           "time_convention": "stamped at the start of the hour"})
    write(out_m, mod, times, g["lat"], g["lon"],
          {**mattrs, "long_name": "EPICC precipitation, coarsened, cut to the radar coverage"},
          {"title": "MOD_01H_RAIN with NaN wherever RADCOV_01H_RAIN is NaN (same cell-hours as the radar)",
           "source": fmod, "coverage_from": out_r, "time_convention": "stamped at the start of the hour"})
    return tag, f"{times.size} h, coverage {cov:.1f}% of cell-hours"


def main():
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    lat, lon, lat_e, lon_e = target_grid()
    with xr.open_dataset(rcfg.mask_file) as m:
        mask = m.mask_s1.values.astype(bool)
        rlat, rlon = m.lat.values, m.lon.values
    flat, inside, ncell = cell_mapping(rlat, rlon, lat_e, lon_e)
    nlat, nlon = lat.size, lon.size
    # cells per 0.1 deg block when the block is fully inside the radar window
    counts = np.bincount(flat, minlength=ncell)
    nfull = np.percentile(counts[counts > 0], 90)
    _G.update(mask=mask, flat=flat, inside=inside, ncell=ncell, nlat=nlat, nlon=nlon, lat=lat, lon=lon,
              nfull=nfull)
    logging.info("2 km cells per 0.1 deg block: %.0f (full block); %d blocks touched", nfull, (counts > 0).sum())
    tags = [f"{y}-{m:02d}" for y in range(cfg.syear, cfg.eyear + 1) for m in range(1, 13)]
    with Pool(NWORKERS) as pool:
        for tag, msg in pool.imap_unordered(do_month, tags):
            logging.info("%s: %s", tag, msg)


if __name__ == "__main__":
    main()

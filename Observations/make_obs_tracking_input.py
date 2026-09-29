#!/usr/bin/env python
"""
Turn the downloaded IMERG and MERGIR subsets into tracker-ready monthly files:
hourly, on the common 0.1 deg IMERG grid, with 2D lat/lon as MCStracking
expects.

The Methods state the evaluation is done "on the same 0.1 deg grid and hourly
resolution", so IMERG's native grid is the target and MERGIR (4 km) is block
averaged onto it. Averaging is done over the half-hourly steps of each hour for
both variables.

Outputs, one pair per month, in {cfg.path_track}:
    OBS_01H_RAIN_YYYY-MM.nc   RAIN (mm h-1), lat(y,x), lon(y,x)
    OBS_01H_TB_YYYY-MM.nc     TB   (K),      lat(y,x), lon(y,x)

Run:  python make_obs_tracking_input.py
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import obs_config as cfg
import obs_download_utils as util


def hourly_mean(var):
    """Average the half-hourly steps of each hour.

    xarray's resample is used only as a fallback: the MERGIR time axis carries
    microsecond jitter (steps come out as 00:30:00.000013), which pushes
    resample onto a slow groupby path and costs ~9 minutes per month against a
    few seconds here. The steps are exactly two per hour and in order, so a
    reshape does the same arithmetic.

    Flooring the stamps also removes the jitter, so the IMERG and MERGIR hourly
    axes come out exactly equal and line up in the intersection below.
    """
    stamps = pd.to_datetime(var.time.values)
    hours = stamps.floor("h")
    regular = (var.sizes["time"] % 2 == 0
               and (hours[::2] == hours[1::2]).all()
               and hours[::2].is_unique)
    if not regular:
        logging.warning("irregular time axis, falling back to resample")
        return var.resample(time="1H").mean()
    with warnings.catch_warnings():
        # an hour with both half-hours missing is a legitimate NaN, not a problem
        warnings.simplefilter("ignore", RuntimeWarning)
        vals = np.nanmean(var.values.reshape(-1, 2, *var.shape[1:]), axis=1)
    return xr.DataArray(vals, dims=var.dims,
                        coords={"time": hours[::2],
                                **{d: var[d] for d in var.dims[1:] if d in var.coords}})


def target_grid(imerg):
    """Cell centres and edges of the IMERG grid the month was cut to."""
    lat, lon = imerg.lat.values, imerg.lon.values
    half = cfg.target_res / 2.0
    lat_edges = np.append(lat - half, lat[-1] + half)
    lon_edges = np.append(lon - half, lon[-1] + half)
    return lat, lon, lat_edges, lon_edges


def block_average(tb, lat_edges, lon_edges):
    """Average the 4 km Tb field onto the coarse grid defined by the edges.

    A straight block mean (each fine cell assigned to the coarse cell it falls
    in) rather than an interpolation: going from 4 km to 0.1 deg is a
    coarsening, and interpolating would throw away most of the fine cells
    instead of averaging them. Cells with no valid fine data come back as NaN.
    """
    nlat, nlon = lat_edges.size - 1, lon_edges.size - 1
    jj = np.digitize(tb.lat.values, lat_edges) - 1
    ii = np.digitize(tb.lon.values, lon_edges) - 1
    inj = (jj >= 0) & (jj < nlat)
    ini = (ii >= 0) & (ii < nlon)

    tb = tb.isel(lat=np.where(inj)[0], lon=np.where(ini)[0])
    flat = (jj[inj][:, None] * nlon + ii[ini][None, :]).ravel()
    ncell = nlat * nlon

    out = np.full((tb.sizes["time"], nlat, nlon), np.nan, dtype="float32")
    for it in range(tb.sizes["time"]):
        vals = tb.isel(time=it).values.ravel().astype("float64")
        good = np.isfinite(vals)
        total = np.bincount(flat[good], weights=vals[good], minlength=ncell)
        count = np.bincount(flat[good], minlength=ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / count, np.nan)
        out[it] = mean.reshape(nlat, nlon)
    return out


def write(fileout, name, data, times, lat, lon, attrs):
    """Write one variable with 2D coordinates, as the tracker expects."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    dset = xr.Dataset(
        {name: (["time", "y", "x"], data)},
        coords={"time": times,
                "lat": (["y", "x"], lat2d.astype("float32")),
                "lon": (["y", "x"], lon2d.astype("float32"))})
    dset[name].attrs.update(attrs)
    dset.attrs.update(title="Observational input for convective storm tracking",
                      region=cfg.reg, grid=f"{cfg.target_res} deg (IMERG native)")
    tmp = f"{fileout}.tmp"
    dset.to_netcdf(tmp, encoding={name: {"zlib": True, "complevel": 5}})
    os.replace(tmp, fileout)
    logging.info("wrote %s", os.path.basename(fileout))


def do_month(year, month):
    tag = f"{year}-{month:02d}"
    fin_pr = f"{cfg.path_imerg}/IMERG_30MIN_PR_{tag}.nc"
    fin_tb = f"{cfg.path_mergir}/MERGIR_30MIN_TB_{tag}.nc"
    out_pr = f"{cfg.path_track}/OBS_01H_RAIN_{tag}.nc"
    out_tb = f"{cfg.path_track}/OBS_01H_TB_{tag}.nc"

    for fin in (fin_pr, fin_tb):
        if not os.path.exists(fin):
            logging.warning("%s missing, skipping %s", os.path.basename(fin), tag)
            return
    if util.already_done(out_pr) and util.already_done(out_tb):
        logging.info("%s already done, skipping", tag)
        return

    with xr.open_dataset(fin_pr) as dset:
        pr = dset.PR.where(dset.PR >= 0)          # -9999.9 is the IMERG fill value
        pr_h = hourly_mean(pr.load())
    lat, lon, lat_edges, lon_edges = target_grid(pr_h)

    with xr.open_dataset(fin_tb) as dset:
        tb_h = hourly_mean(dset.TB.load())

    # keep only hours present in both datasets
    times = np.intersect1d(pr_h.time.values, tb_h.time.values)
    if times.size != pr_h.sizes["time"] or times.size != tb_h.sizes["time"]:
        logging.warning("%s: %d common hours (IMERG %d, MERGIR %d)", tag,
                        times.size, pr_h.sizes["time"], tb_h.sizes["time"])
    pr_h = pr_h.sel(time=times)
    tb_h = tb_h.sel(time=times)

    tb_grid = block_average(tb_h, lat_edges, lon_edges)
    logging.info("%s: %d hours, grid %d x %d, Tb valid %.1f%%", tag, times.size,
                 lat.size, lon.size, 100 * np.isfinite(tb_grid).mean())

    os.makedirs(cfg.path_track, exist_ok=True)
    write(out_pr, "RAIN", pr_h.values.astype("float32"), times, lat, lon,
          {"units": "mm h-1", "long_name": "IMERG hourly precipitation",
           "source": "GPM_3IMERGHH.07"})
    write(out_tb, "TB", tb_grid, times, lat, lon,
          {"units": "K", "long_name": "MERGIR hourly brightness temperature, "
                                      "block averaged to the IMERG grid",
           "source": "GPM_MERGIR.1"})


def main():
    util.start_logger("make_obs_tracking_input")
    os.makedirs(cfg.path_track, exist_ok=True)
    for year, month in util.months_to_do():
        do_month(year, month)


if __name__ == "__main__":
    main()

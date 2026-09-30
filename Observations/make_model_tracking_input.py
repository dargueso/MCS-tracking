#!/usr/bin/env python
"""
Coarsen the EPICC 2 km output onto the observational 0.1 deg grid, so the
tracker can be run with identical settings on model and observations and the
resulting storm statistics compared like for like.

The target grid is read straight out of an OBS_01H_* file rather than rebuilt,
so the two sides are guaranteed to be on exactly the same points.

Brightness temperature is obtained from OLR at 2 km and only then averaged onto
the coarse grid. Averaging Tb rather than the radiance is deliberate: MERGIR
supplies Tb and nothing else, so its coarse values are block means of Tb, and
the model is treated the same way. Converting after averaging would make the
two sides differ by the nonlinearity of the T^4 relation, which is the sort of
difference this comparison is meant to detect rather than introduce.

Outputs, per run, in {cfg.path_modcoarse}/<wrun>:
    MOD_01H_RAIN_YYYY-MM.nc   RAIN (mm h-1), lat(y,x), lon(y,x)
    MOD_01H_TB_YYYY-MM.nc     TB   (K),      lat(y,x), lon(y,x)

Run:  python make_model_tracking_input.py [pres|fut]     (default: both)
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import const

import obs_config as cfg
import obs_download_utils as util


def olr_to_tb(olr, method):
    """Brightness temperature from OLR.

    SB: the plain grey-body inversion, Tb = (OLR/sigma)^0.25. This is what the
    WRF driver uses, but it returns an effective broadband emission
    temperature, not a window-channel Tb: it is ~22 K colder than MERGIR at
    OLR 240.

    YS: Ohring et al. (1984) as given in Yang and Slingo (2001), the relation
    used by PyFLEXTRKR and others for exactly this comparison. It defines a
    flux-equivalent temperature Tf = Tb (a + b Tb) with OLR = sigma Tf^4, and
    is inverted here for Tb. The two agree to within 2 K in cold cloud, so the
    choice matters far more for the warm background than for the cloud shield.
    """
    tf = (olr / const.SB_sigma) ** 0.25
    if method == "SB":
        return tf
    if method == "YS":
        a, b = 1.228, -1.106e-3      # K^-1
        return (-a + np.sqrt(a ** 2 + 4 * b * tf)) / (2 * b)
    raise ValueError(f"unknown brightness-temperature method: {method}")


def target_grid():
    """Cell centres and edges of the observational grid, taken from an OBS file."""
    ref = sorted(f for f in os.listdir(cfg.path_track) if f.startswith("OBS_01H_RAIN"))
    if not ref:
        raise RuntimeError(f"no OBS_01H_RAIN_*.nc in {cfg.path_track}; "
                           "run make_obs_tracking_input.py first")
    with xr.open_dataset(os.path.join(cfg.path_track, ref[0])) as dset:
        lat = dset.lat.values[:, 0]          # 2D but constant along each axis
        lon = dset.lon.values[0, :]
    half = cfg.target_res / 2.0
    return lat, lon, np.append(lat - half, lat[-1] + half), \
        np.append(lon - half, lon[-1] + half)


def cell_mapping(mod_lat, mod_lon, lat_edges, lon_edges):
    """Flat target-cell index for every model cell, and a mask of those inside.

    The model grid is curvilinear, so lat and lon are both 2D and each cell is
    placed individually. Cells falling outside the observational box are
    dropped.
    """
    nlat, nlon = lat_edges.size - 1, lon_edges.size - 1
    jj = np.digitize(mod_lat.ravel(), lat_edges) - 1
    ii = np.digitize(mod_lon.ravel(), lon_edges) - 1
    inside = (jj >= 0) & (jj < nlat) & (ii >= 0) & (ii < nlon)
    return (jj[inside] * nlon + ii[inside]), inside, nlat * nlon


def coarsen(var, flat_idx, inside, ncell, nlat, nlon):
    """Block-average a (time, y, x) model field onto the target grid."""
    out = np.full((var.shape[0], nlat, nlon), np.nan, dtype="float32")
    counts = np.bincount(flat_idx, minlength=ncell)
    for it in range(var.shape[0]):
        vals = var[it].ravel()[inside].astype("float64")
        good = np.isfinite(vals)
        total = np.bincount(flat_idx[good], weights=vals[good], minlength=ncell)
        n = counts if good.all() else np.bincount(flat_idx[good], minlength=ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[it] = np.where(n > 0, total / n, np.nan).reshape(nlat, nlon)
    return out


def write(fileout, name, data, times, lat, lon, attrs, wrun):
    lon2d, lat2d = np.meshgrid(lon, lat)
    dset = xr.Dataset(
        {name: (["time", "y", "x"], data)},
        coords={"time": times,
                "lat": (["y", "x"], lat2d.astype("float32")),
                "lon": (["y", "x"], lon2d.astype("float32"))})
    dset[name].attrs.update(attrs)
    dset.attrs.update(
        title="EPICC 2 km output coarsened to the observational grid",
        run=wrun, region=cfg.reg,
        grid=f"{cfg.target_res} deg, identical to the OBS_01H_* files",
        time_convention="stamped at the start of the hour, as the OBS files are "
                        "(the source files are stamped mid-hour)")
    tmp = f"{fileout}.tmp"
    dset.to_netcdf(tmp, encoding={name: {"zlib": True, "complevel": 5}})
    os.replace(tmp, fileout)
    logging.info("wrote %s", os.path.basename(fileout))


def do_month(wrun, year, month, grid, mapping):
    tag = f"{year}-{month:02d}"
    dirout = f"{cfg.path_modcoarse}/{wrun}"
    out_pr = f"{dirout}/MOD_01H_RAIN_{tag}.nc"
    out_tb = {m: f"{dirout}/MOD_01H_TB_{m}_{tag}.nc" for m in cfg.bt_methods}
    if util.already_done(out_pr) and all(util.already_done(f) for f in out_tb.values()):
        logging.info("%s %s already done, skipping", wrun, tag)
        return

    fin_pr = f"{cfg.path_model}/{wrun}/RAIN/UIB_01H_RAIN_{tag}.nc"
    fin_ol = f"{cfg.path_model}/{wrun}/OLR/UIB_01H_OLR_{tag}.nc"
    for fin in (fin_pr, fin_ol):
        if not os.path.exists(fin):
            logging.warning("%s missing, skipping %s %s", os.path.basename(fin), wrun, tag)
            return

    lat, lon, _, _ = grid
    flat_idx, inside, ncell = mapping
    nlat, nlon = lat.size, lon.size

    with xr.open_dataset(fin_pr) as dpr, xr.open_dataset(fin_ol) as dol:
        # The EPICC hourly files are stamped mid-hour (HH:30 with time_bnds
        # HH:00-HH+1:00 in the clock-hour files on /scratch3; HH:25 in the
        # original, 10-min-early files), whereas the observations are stamped at
        # the start of the hour. Floor the model stamps to match; otherwise
        # nothing downstream that joins the two on time would line up.
        times = pd.to_datetime(dpr.time.values).floor("h").values
        # which hourly rain this is, recorded in the output and, through
        # track_storms.py, in the tracked storms
        rain_window = ("clock hour HH:00-HH+1:00" if "correction" in dpr.attrs
                       else "original cdo hoursum: HH-1:50 to HH:50 (10 min early)")
        nt = times.size
        pr_out = np.empty((nt, nlat, nlon), dtype="float32")
        tb_out = {m: np.empty((nt, nlat, nlon), dtype="float32") for m in cfg.bt_methods}
        # chunked over time: a whole month at 2 km does not fit comfortably
        for i0 in range(0, nt, cfg.chunk_hours):
            i1 = min(i0 + cfg.chunk_hours, nt)
            pr = dpr.RAIN.isel(time=slice(i0, i1)).values
            olr = dol.OLR.isel(time=slice(i0, i1)).squeeze().values
            pr_out[i0:i1] = coarsen(pr, flat_idx, inside, ncell, nlat, nlon)
            for method in cfg.bt_methods:
                # convert at 2 km and only then average, to match the way the
                # MERGIR Tb is block averaged
                tb = olr_to_tb(olr, method)
                tb_out[method][i0:i1] = coarsen(tb, flat_idx, inside, ncell, nlat, nlon)

    logging.info("%s %s: %d hours, grid %d x %d", wrun, tag, nt, nlat, nlon)
    os.makedirs(dirout, exist_ok=True)
    write(out_pr, "RAIN", pr_out, times, lat, lon,
          {"units": "mm h-1", "long_name": "EPICC precipitation, coarsened",
           "source": fin_pr, "rain_hour_window": rain_window}, wrun)
    longname = {"SB": "Brightness temperature from OLR (Stefan-Boltzmann), coarsened",
                "YS": "Brightness temperature from OLR (Yang & Slingo 2001), coarsened"}
    for method in cfg.bt_methods:
        write(out_tb[method], "TB", tb_out[method], times, lat, lon,
              {"units": "K", "long_name": longname[method],
               "source": f"{wrun} UIB_01H_OLR", "bt_method": method,
               "note": "Tb computed at 2 km, then block averaged"}, wrun)


def main():
    util.start_logger("make_model_tracking_input")
    which = sys.argv[1:] or ["pres", "fut"]
    grid = target_grid()
    lat, lon, lat_edges, lon_edges = grid

    for key in which:
        wrun = cfg.model_runs[key]
        with xr.open_dataset(
                f"{cfg.path_model}/{wrun}/RAIN/UIB_01H_RAIN_{cfg.syear}-01.nc") as dset:
            mapping = cell_mapping(dset.lat.values, dset.lon.values, lat_edges, lon_edges)
        logging.info("%s: %d of %d model cells fall inside the target box",
                     wrun, int(mapping[1].sum()), mapping[1].size)
        for year, month in util.months_to_do():
            do_month(wrun, year, month, grid, mapping)


if __name__ == "__main__":
    main()

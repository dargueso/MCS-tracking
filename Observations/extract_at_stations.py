#!/usr/bin/env python
"""
Hourly model (and EURADCLIM) rain at the AEMET stations.

Stations: every station of the combined 10-min dataset plus the Arnau-only
stations (daily evaluation). Each is placed in the model cell containing it, and
three values are kept per hour:

    cell   the station's own 2 km cell
    nmax   the maximum over the (2k+1)^2 cells around it (k = nbhd_half)
    nmean  the mean over the same block

A gauge is a point; the model and the radar are 4 km2 averages that may also
put a storm a few km off. Cell and neighbourhood maximum bracket that: an
extreme missed by both is missed by the model, not by the sampling.

EURADCLIM comes from its files on the model grid (make_radar_input.py) and is
added for whatever months exist; re-run once more months are processed.

Outputs, in {cfg.path_eval}:
    AT_STATIONS_<run>_01H_2011-2020.nc
    AT_STATIONS_EURADCLIM_01H_2011-2020.nc

    python extract_at_stations.py            # model and EURADCLIM
    python extract_at_stations.py --radar    # EURADCLIM only
"""

import os
import argparse
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import station_config as cfg
import radar_config as rcfg
import radar_utils as ru
import obs_download_utils as util

NWORKERS = 12
_S = {}


def stations():
    """Codes, lat, lon of all stations: 10-min dataset first, then Arnau-only."""
    with xr.open_dataset(cfg.file_01h) as h, xr.open_dataset(cfg.file_arnau) as a:
        c1 = [str(c) for c in h.code.values]
        extra = [i for i, c in enumerate(a.code.values) if str(c) not in set(c1)]
        codes = c1 + [str(a.code.values[i]) for i in extra]
        lat = np.r_[h.lat.values, a.lat.values[extra]]
        lon = np.r_[h.lon.values, a.lon.values[extra]]
        names = np.r_[h.name.values, a.name.values[extra]]
    return codes, lat, lon, names


def model_cells(lat, lon):
    """(j, i) of the model cell containing each station, and distance to its centre."""
    with xr.open_dataset(cfg.geofile) as geo:
        glat, glon = geo.XLAT_M[0].values, geo.XLONG_M[0].values
        attrs = dict(geo.attrs)
    proj = ru.model_projection(attrs)
    gx, gy = proj(glon, glat)
    dx = float(attrs["DX"])
    sx, sy = proj(lon, lat)
    i = np.round((sx - gx[0, 0]) / dx).astype(int)
    j = np.round((sy - gy[0, 0]) / dx).astype(int)
    ny, nx = glat.shape
    k = cfg.nbhd_half
    inside = (i >= k) & (i < nx - k) & (j >= k) & (j < ny - k)
    dist = np.full(lat.size, np.nan)
    dist[inside] = np.hypot(sx[inside] - gx[j[inside], i[inside]],
                            sy[inside] - gy[j[inside], i[inside]])
    return j, i, inside, dist


def sample(field, j, i):
    """field (nt, ny, nx) -> cell, nmax, nmean at each (j, i), shape (nt, nst)."""
    k = cfg.nbhd_half
    blocks = np.stack([field[:, j + dj, i + di]
                       for dj in range(-k, k + 1) for di in range(-k, k + 1)], axis=-1)
    with np.errstate(invalid="ignore"):
        # a radar block with gaps: max/mean over what is valid, NaN if the cell is
        return field[:, j, i], np.nanmax(blocks, -1), np.nanmean(blocks, -1)


def do_model_month(tag):
    j, i = _S["j"], _S["i"]
    fin = f"{cfg.path_model_rain}/UIB_01H_RAIN_{tag}.nc"
    if not os.path.exists(fin):
        return tag, None
    with xr.open_dataset(fin) as ds:
        times = pd.to_datetime(ds.time.values).floor("h")
        field = ds.RAIN.values
    return tag, (times, *sample(field, j, i))


def do_radar_month(tag):
    j, i = _S["j"], _S["i"]
    fin = f"{rcfg.path_rad_grid}/RAD_01H_RAIN_{tag}.nc"
    if not os.path.exists(fin):
        return tag, None
    with xr.open_dataset(fin) as ds:
        times = pd.to_datetime(ds.time.values)
        field = ds.RAIN.values
    return tag, (times, *sample(field, j, i))


def run(worker, jj, ii, label, tags):
    _S.update(j=jj, i=ii)          # inherited by the forked workers
    hours = pd.date_range(f"{cfg.syear}-01-01", f"{cfg.eyear}-12-31 23:00", freq="h")
    out = {k: np.full((jj.size, hours.size), np.nan, "float32") for k in ("cell", "nmax", "nmean")}
    got = 0
    with Pool(NWORKERS) as pool:
        for tag, res in pool.imap_unordered(worker, tags):
            if res is None:
                continue
            times, *vals = res
            pos = hours.get_indexer(times)
            ok = pos >= 0
            for k, v in zip(("cell", "nmax", "nmean"), vals):
                out[k][:, pos[ok]] = v[ok].T
            got += 1
    logging.info("%s: %d of %d months", label, got, len(tags))
    return hours, out, got


def write(fout, hours, out, codes, names, lat, lon, j, i, dist, attrs):
    ds = xr.Dataset(
        {k: (("station", "time"), v, {"units": "mm", "long_name": f"rain in the clock hour, {k}"})
         for k, v in out.items()},
        coords={"time": hours, "code": ("station", np.array(codes, dtype=object)),
                "name": ("station", names), "lat": ("station", lat.astype("float32")),
                "lon": ("station", lon.astype("float32")),
                "model_j": ("station", j), "model_i": ("station", i),
                "dist_to_cell_centre_m": ("station", dist.astype("float32"))},
        attrs={**attrs, "neighbourhood": f"{2 * cfg.nbhd_half + 1}x{2 * cfg.nbhd_half + 1} cells",
               "time_convention": "START of the clock hour, UTC"})
    enc = {k: {"zlib": True, "complevel": 4} for k in out}
    ds.to_netcdf(f"{fout}.tmp", encoding=enc)
    os.replace(f"{fout}.tmp", fout)
    logging.info("wrote %s", fout)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--radar", action="store_true", help="EURADCLIM only")
    args = par.parse_args()
    util.start_logger("extract_at_stations")
    os.makedirs(cfg.path_eval, exist_ok=True)
    codes, lat, lon, names = stations()
    j, i, inside, dist = model_cells(lat, lon)
    if (dist[inside] > 1500).any():
        raise RuntimeError("a station is > 1.5 km from its cell centre: grid placement is wrong")
    logging.info("%d stations, %d inside the model grid (max %.0f m from cell centre)",
                 len(codes), inside.sum(), np.nanmax(dist))
    keep = np.where(inside)[0]
    codes = [codes[k] for k in keep]
    lat, lon, names, j, i, dist = lat[keep], lon[keep], names[keep], j[keep], i[keep], dist[keep]
    tags = [f"{y}-{m:02d}" for y in range(cfg.syear, cfg.eyear + 1) for m in range(1, 13)]

    if not args.radar:
        hours, out, _ = run(do_model_month, j, i, cfg.model_run, tags)
        write(f"{cfg.path_eval}/AT_STATIONS_{cfg.model_run}_01H_{cfg.syear}-{cfg.eyear}.nc",
              hours, out, codes, names, lat, lon, j, i, dist,
              {"title": f"{cfg.model_run} hourly rain at AEMET stations",
               "source": cfg.path_model_rain,
               "note": "model stamps HH:25 (window HH:00-HH:50) floored to HH:00"})

    # EURADCLIM is on the cropped model grid: shift the indices
    grid = ru.model_grid()
    jr, ir = j - grid["ys"].start, i - grid["xs"].start
    ny, nx = grid["lat"].shape
    k = cfg.nbhd_half
    rin = (jr >= k) & (jr < ny - k) & (ir >= k) & (ir < nx - k)
    jr, ir = np.where(rin, jr, k), np.where(rin, ir, k)     # outside the crop: dummy, masked below
    hours, out, got = run(do_radar_month, jr, ir, "EURADCLIM", tags)
    if got:
        for v in out.values():
            v[~rin] = np.nan
        write(f"{cfg.path_eval}/AT_STATIONS_EURADCLIM_01H_{cfg.syear}-{cfg.eyear}.nc",
              hours, out, codes, names, lat, lon, j, i, dist,
              {"title": "EURADCLIM hourly rain at AEMET stations (on the model grid)",
               "source": rcfg.path_rad_grid, "months_available": got,
               "note": "NaN where EURADCLIM has no data or the station is outside the "
                       "radar evaluation box; the static radar quality mask is NOT applied"})


if __name__ == "__main__":
    main()

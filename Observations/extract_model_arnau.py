#!/usr/bin/env python
"""
Model daily quantities at the Arnau stations, computed like Arnau's.

AEMET's climatological daily record (Arnau) gives, per station and UTC day
(00-24), the total P24 and the maximum rain in sliding windows of 10, 20 and
30 min and 1, 2, 6 and 12 h, each window inside the day. Rebuilding them from
the verified 10-min gauge data shows that is the definition: sliding windows
inside the day reproduce Arnau on 99.8-100% of days, clock-aligned blocks on
53-77%, and windows reaching into the previous day on 83-99%.

The model gets the same treatment from its 10-min rain (UIB_10MIN_RAIN.zarr,
WRF PREC_ACC_NC, END-stamped: the value at t is rain from t-10 min to t). The
day D is the 144 steps stamped D 00:10 .. D+1 00:00.

Each station is placed in the model cell containing it (the cells of
AT_STATIONS_<run>_01H, so the hourly and daily evaluations use the same
cells). Two values are kept per quantity:

    cell   the station's own 2 km cell
    nmax   the maximum of that quantity over the (2k+1)^2 cells around it

The zarr is read tile by tile (its 50x50-cell chunks are raw float32), only
the tiles holding a station's neighbourhood.

Output, in {cfg.path_eval}:
    AT_ARNAU_<run>_DAILY_2011-2019.nc

    python extract_model_arnau.py
"""

import os
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

import station_config as cfg

NWORKERS = 16
STEPS = 144
# Arnau variable -> window length in 10-min steps
WINDOWS = {"p24": STEPS, "pmax10": 1, "pmax20": 2, "pmax30": 3, "pmax60": 6,
           "pmax2h": 12, "pmax6h": 36, "pmax12h": 72}
ZARR = f"/scratch1/dargueso/postprocessed/EPICC/{cfg.model_run}/UIB_10MIN_RAIN.zarr"
T0 = pd.Timestamp("2011-01-01")          # first zarr stamp (END of 2010-12-31 23:50-24:00)
MOD_01H = f"{cfg.path_eval}/AT_STATIONS_{cfg.model_run}_01H_{cfg.syear}-{cfg.eyear}.nc"
FOUT = f"{cfg.path_eval}/AT_ARNAU_{cfg.model_run}_DAILY_2011-2019.nc"


def daily_quantities(s, nday):
    """s: start-stamped 10-min series from day 0 -> (nday, nvar) Arnau-style values."""
    x = s[:nday * STEPS].reshape(nday, STEPS).astype("float64")
    cs = np.concatenate([np.zeros((nday, 1)), np.cumsum(x, 1)], 1)
    return np.stack([(cs[:, w:] - cs[:, :-w]).max(1) for w in WINDOWS.values()], -1)


def do_tile(args):
    """All needed cells of one 50x50 tile -> {(j, i): (nday, nvar)}."""
    (tj, ti), cells, nday, meta = args
    nt, cy, cx, ct = meta
    series = np.full((nt, len(cells)), np.nan, "float32")
    yy = np.array([j - tj * cy for j, _ in cells])
    xx = np.array([i - ti * cx for _, i in cells])
    for k in range(-(-nt // ct)):
        f = f"{ZARR}/RAIN/{k}.{tj}.{ti}"
        if not os.path.exists(f):                  # a chunk never written is all fill (NaN)
            continue
        n = min(ct, nt - k * ct)
        block = np.fromfile(f, dtype="<f4").reshape(ct, cy, cx)[:n]
        series[k * ct:k * ct + n] = block[:, yy, xx]
    # END-stamped -> START-stamped: step k' (starting T0 + k'*10 min) is zarr index k'+1
    return {c: daily_quantities(series[1:, n], nday) for n, c in enumerate(cells)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
    with xr.open_dataset(cfg.file_arnau) as a:
        codes = [str(c) for c in a.code.values]
        days = pd.to_datetime(a.day.values)
        lat, lon, names = a.lat.values, a.lon.values, a.name.values
    with xr.open_dataset(MOD_01H) as m:
        mc = [str(c) for c in m.code.values]
        idx = [mc.index(c) for c in codes]
        j, i = m.model_j.values[idx], m.model_i.values[idx]
    zarray = pd.read_json(f"{ZARR}/RAIN/.zarray", typ="series")
    nt, ny, nx = zarray["shape"]
    ct, cy, cx = zarray["chunks"]
    day0 = (days[0] - T0).days
    nday = day0 + len(days)
    assert day0 == 0 and (nday * STEPS + 1) <= nt, "Arnau days must start 2011-01-01 and fit the zarr"

    k = cfg.nbhd_half
    offs = [(dj, di) for dj in range(-k, k + 1) for di in range(-k, k + 1)]
    need = sorted({(jj + dj, ii + di) for jj, ii in zip(j, i) for dj, di in offs})
    tiles = {}
    for c in need:
        tiles.setdefault((c[0] // cy, c[1] // cx), []).append(c)
    logging.info("%d stations, %d cells in %d tiles; reading %s", len(codes), len(need),
                 len(tiles), ZARR)

    vals = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(NWORKERS) as pool:
        jobs = [(t, cells, nday, (nt, cy, cx, ct)) for t, cells in tiles.items()]
        for n, res in enumerate(pool.imap_unordered(do_tile, jobs), 1):
            vals.update(res)
            if n % 10 == 0 or n == len(jobs):
                logging.info("tiles %d/%d", n, len(jobs))

    nvar = len(WINDOWS)
    cell = np.stack([vals[(jj, ii)] for jj, ii in zip(j, i)])                  # (nst, nday, nvar)
    block = np.stack([np.stack([vals[(jj + dj, ii + di)] for dj, di in offs])
                      for jj, ii in zip(j, i)])                                  # (nst, 9, nday, nvar)
    with np.errstate(invalid="ignore"):
        nmax = np.nanmax(block, 1)

    data = {}
    for v, key in enumerate(WINDOWS):
        data[f"{key}_cell"] = (("station", "day"), cell[..., v].astype("float32"),
                               {"units": "mm", "long_name": f"{key}, model cell"})
        data[f"{key}_nmax"] = (("station", "day"), nmax[..., v].astype("float32"),
                               {"units": "mm", "long_name": f"{key}, max over the "
                                f"{2 * k + 1}x{2 * k + 1} cells"})
    ds = xr.Dataset(data, coords={
        "day": days, "code": ("station", np.array(codes, dtype=object)), "name": ("station", names),
        "lat": ("station", lat.astype("float32")), "lon": ("station", lon.astype("float32")),
        "model_j": ("station", j), "model_i": ("station", i)},
        attrs={"title": f"{cfg.model_run} daily quantities at the Arnau stations, as Arnau defines them",
               "source": ZARR,
               "day_convention": "00-24 UTC; model 10-min steps stamped D 00:10 .. D+1 00:00 (END stamps)",
               "windows": "p24 daily total; pmaxNN maximum rain in a sliding window of that length "
                          "inside the day",
               "neighbourhood": f"{2 * k + 1}x{2 * k + 1} cells",
               "history": "extract_model_arnau.py (MCS-tracking/Observations)"})
    enc = {v: {"zlib": True, "complevel": 4} for v in data}
    ds.to_netcdf(FOUT, encoding=enc)
    logging.info("wrote %s", FOUT)


if __name__ == "__main__":
    main()

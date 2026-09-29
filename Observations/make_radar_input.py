#!/usr/bin/env python
"""
Put EURADCLIM on the EPICC 2 km grid, one file per month, hourly, stamped at the
START of the hour like the model and the IMERG/MERGIR files.

EURADCLIM stamps the END of the accumulation hour (file ..._201409011400.h5 is
13:00-14:00 UTC); the EPICC files stamp the centre (13:25, bounds 13:00-13:50),
floored to 13:00 downstream. Both therefore end up on 13:00.

Remapping is the supersampled conservative scheme in radar_utils.remap_weights,
onto the model grid cropped to the evaluation box. Hours missing from the
archive are written as all-NaN, so every file has the full time axis and a gap
can never be mistaken for a dry hour.

The radars contributing to each hour (ODIM /how/nodes) are counted and written
as an attribute; make_radar_mask.py uses them for the distance-to-radar mask.

Output, in {cfg.path_rad_grid}:
    RAD_01H_RAIN_YYYY-MM.nc   RAIN (mm h-1) on the cropped model grid

    python make_radar_input.py                 # all months
    python make_radar_input.py 2014-09 2014-10 # just these
"""

import os
import sys
import shutil
import logging
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import radar_config as cfg
import radar_utils as ru
import obs_download_utils as util

NWORKERS = 12       # ~2.5 GB per worker; the machine has 1.5 TB

_G = {}             # per-process grid, weights and index, built once


def init():
    grid = ru.model_grid()
    W, window = ru.remap_weights(grid)
    _G.update(grid=grid, W=W, window=window, index=ru.index_hours())


def do_month(tag):
    grid, W, window, index = _G["grid"], _G["W"], _G["window"], _G["index"]
    fileout = f"{cfg.path_rad_grid}/RAD_01H_RAIN_{tag}.nc"
    if os.path.exists(fileout):
        logging.info("%s already done, skipping", tag)
        return tag, None
    period = pd.Period(tag, "M")
    hours = pd.date_range(period.start_time, period.end_time.floor("h"), freq="h")
    present = [h for h in hours if h in index]
    if not present:
        logging.warning("%s: no EURADCLIM hours found, skipping", tag)
        return tag, 0
    # The last hour of a month is stamped 00:00 on the 1st and lives in the NEXT
    # month's zip. Writing now would freeze a gap (done months are skipped), so
    # wait for that archive -- except after the last month of the period, whose
    # successor is not downloaded at all.
    last_month = period == pd.Period(f"{cfg.eyear}-{max(cfg.months):02d}", "M")
    if hours[-1] not in index and not last_month:
        logging.warning("%s: last hour not available yet (in the next month's "
                        "archive), skipping for now", tag)
        return tag, None

    ny, nx = grid["lat"].shape
    r0, r1, c0, c1 = window
    out = np.full((hours.size, ny, nx), np.nan, dtype="float32")
    nodes = Counter()
    scratch = f"{cfg.path_rad_scratch}/{tag}"
    os.makedirs(scratch, exist_ok=True)
    try:
        for i0 in range(0, hours.size, 48):
            chunk = hours[i0:i0 + 48]
            paths = ru.extract(index, chunk, scratch)
            stack = np.full((chunk.size, r1 - r0, c1 - c0), np.nan)
            for j, h in enumerate(chunk):
                if h not in paths:
                    continue
                try:
                    stack[j], used = ru.read_hour(paths[h], window)
                    nodes.update(used)
                except (OSError, KeyError, ValueError) as err:
                    logging.error("%s unreadable, left missing: %s", paths[h], err)
                if index[h][0] is not None:        # extracted copy, not the source
                    os.remove(paths[h])
            out[i0:i0 + chunk.size] = ru.remap(W, stack, (ny, nx))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    nmiss = hours.size - len(present)
    dset = xr.Dataset(
        {"RAIN": (("time", "y", "x"), out,
                  {"units": "mm h-1", "long_name": "EURADCLIM 1-h accumulation",
                   "_FillValue": np.float32(np.nan)})},
        coords={"time": hours.values,
                "lat": (("y", "x"), grid["lat"].astype("float32")),
                "lon": (("y", "x"), grid["lon"].astype("float32"))})
    dset.attrs.update(
        title="EURADCLIM on the EPICC 2 km grid",
        source="EURADCLIM (Overeem et al. 2023), KNMI Data Platform, "
               f"{cfg.knmi_dataset} v{cfg.knmi_version}",
        remapping=f"supersampled conservative, {cfg.nsub}x{cfg.nsub} per model cell; "
                  f"valid if >= {cfg.min_valid_frac} of the cell is valid radar",
        model_grid=f"{cfg.geofile} rows {grid['ys'].start}:{grid['ys'].stop} "
                   f"cols {grid['xs'].start}:{grid['xs'].stop}",
        radar_window=f"rows {r0}:{r1} cols {c0}:{c1}",
        time_convention="stamped at the START of the hour; EURADCLIM filenames "
                        "carry the end",
        hours_missing=nmiss,
        nodes=",".join(f"{k}:{v}" for k, v in sorted(nodes.items())))
    os.makedirs(cfg.path_rad_grid, exist_ok=True)
    tmp = f"{fileout}.tmp"
    dset.to_netcdf(tmp, encoding={"RAIN": {"zlib": True, "complevel": 4,
                                           "chunksizes": (24, ny, nx)}})
    os.replace(tmp, fileout)
    logging.info("wrote %s (%d hours missing)", os.path.basename(fileout), nmiss)
    return tag, nmiss


def main():
    util.start_logger("make_radar_input")
    tags = sys.argv[1:] or [f"{y}-{m:02d}" for y in range(cfg.syear, cfg.eyear + 1)
                            for m in cfg.months]
    with Pool(NWORKERS, initializer=init) as pool:
        for tag, nmiss in pool.imap_unordered(do_month, tags):
            if nmiss:
                logging.warning("%s: %d hours missing", tag, nmiss)


if __name__ == "__main__":
    main()

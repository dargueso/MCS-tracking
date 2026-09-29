#!/usr/bin/env python
"""
Per-cell hourly-rain statistics for EURADCLIM and EPICC, month by month, on the
same grid and over exactly the same hours and cells.

Like with like: the model is only counted where and when the radar is valid.
A radar gap is removed from BOTH sides, cell by cell and hour by hour, so the
two sets of statistics are always over an identical sample. At the aggregated
scales the model is block-averaged over the same valid 2 km cells as the radar.

The static quality mask is NOT applied here. Everything is stored in additive
form (counts, sums, per-cell intensity histograms), so the mask can be revised
in make_radar_mask.py and the plots re-made without touching these files.

Output, in {cfg.path_rad_stats}/<dataset>/:
    RADSTATS_<dataset>_s<scale>_YYYY-MM.nc

    python radar_model_stats.py                  # all months
    python radar_model_stats.py 2014-09          # just these
"""

import os
import sys
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import radar_config as cfg
import radar_utils as ru
import obs_download_utils as util

NWORKERS = 12       # ~3 GB per worker at the 2 km scale; the machine has 1.5 TB
RAD, MOD = "EURADCLIM", cfg.model_run


def do_month(tag):
    outs = {(d, k): ru.stats_file(d, k, int(tag[:4]), int(tag[5:]))
            for d in (RAD, MOD) for k in cfg.scales}
    if all(os.path.exists(f) for f in outs.values()):
        logging.info("%s already done, skipping", tag)
        return tag, None
    fin_rad = f"{cfg.path_rad_grid}/RAD_01H_RAIN_{tag}.nc"
    fin_mod = f"{cfg.path_model_rain}/UIB_01H_RAIN_{tag}.nc"
    for fin in (fin_rad, fin_mod):
        if not os.path.exists(fin):
            logging.warning("%s missing, skipping %s", os.path.basename(fin), tag)
            return tag, None

    grid = ru.model_grid()
    with xr.open_dataset(fin_rad) as drad, xr.open_dataset(fin_mod) as dmod:
        # model stamps are HH:25 (centre of HH:00-HH:50): floor to the hour start
        tmod = pd.to_datetime(dmod.time.values).floor("h")
        trad = pd.to_datetime(drad.time.values)
        common = trad.intersection(tmod)
        if common.size < trad.size:
            logging.warning("%s: %d radar hours have no model hour", tag,
                            trad.size - common.size)
        irad = trad.get_indexer(common)
        imod = tmod.get_indexer(common)
        # the RAD file is on the cropped grid; crop the model identically
        if drad.RAIN.shape[1:] != grid["lat"].shape or not np.allclose(
                drad.lat.values, grid["lat"]):
            raise RuntimeError(f"{fin_rad} is not on the current model crop")

        stats = {}
        for k in cfg.scales:
            shape = ru.aggregate_static(grid["lat"], k).shape
            stats[(RAD, k)] = ru.CellStats(shape)
            stats[(MOD, k)] = ru.CellStats(shape)

        for i0 in range(0, common.size, cfg.chunk_hours):
            sl = slice(i0, i0 + cfg.chunk_hours)
            rad = drad.RAIN.isel(time=irad[sl]).values.astype("float64")
            mod = dmod.RAIN.isel(time=imod[sl], y=grid["ys"], x=grid["xs"]).values
            mod = mod.astype("float64")
            valid = np.isfinite(rad) & np.isfinite(mod)
            hours = common[sl].hour.values
            for k in cfg.scales:
                stats[(RAD, k)].add(ru.aggregate(rad, valid, k), hours)
                stats[(MOD, k)].add(ru.aggregate(mod, valid, k), hours)

    for (dataset, k), st in stats.items():
        attrs = {"dataset": dataset, "month": tag, "scale_cells": k,
                 "scale_km": 2 * k, "hours": int(common.size),
                 "wet_thres_mm_h": cfg.wet_thres,
                 "sampling": "only hours and cells where EURADCLIM is valid, "
                             "identically for both datasets",
                 "time_convention": "hour of day is UTC, hour START"}
        ds = st.to_dataset(ru.aggregate_static(grid["lat"], k),
                           ru.aggregate_static(grid["lon"], k), attrs)
        fout = outs[(dataset, k)]
        os.makedirs(os.path.dirname(fout), exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(f"{fout}.tmp", encoding=enc)
        os.replace(f"{fout}.tmp", fout)
    logging.info("%s: %d common hours", tag, common.size)
    return tag, common.size


def main():
    util.start_logger("radar_model_stats")
    tags = sys.argv[1:] or [f"{y}-{m:02d}" for y in range(cfg.syear, cfg.eyear + 1)
                            for m in cfg.months]
    with Pool(NWORKERS) as pool:
        for _ in pool.imap_unordered(do_month, tags):
            pass


if __name__ == "__main__":
    main()

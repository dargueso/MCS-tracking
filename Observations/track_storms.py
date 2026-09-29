#!/usr/bin/env python
"""
Run the storm tracker on any of the 0.1 deg datasets: the observations, or the
coarsened model with either OLR-to-BT conversion.

This is the counterpart of MCS_tracking_WRF.py for data that already supplies
brightness temperature. The only real difference is that nothing is derived
from OLR here: OBS_01H_TB_* is MERGIR window BT, and MOD_01H_TB_<method>_* was
converted at 2 km before being coarsened. Everything else -- thresholds, object
definitions -- comes from mcs_config, so the tracker configuration is identical
to the one used on the native 2 km output.

    python track_storms.py obs
    python track_storms.py mod0.1_YS_pres
    python track_storms.py obs mod0.1_YS_pres mod0.1_SB_pres
    python track_storms.py --list

Output goes to {cfg.path_tracking}/<dataset>/<exp>/, where <exp> is read from
mcs_config so it always matches the thresholds actually in force.
"""

import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as mcfg
from tracking_functions_optimized import MCStracking

import obs_config as cfg
import obs_download_utils as util

# Which mcs_config threshold set is loaded. Keyed on the values that actually
# drive the tracker, so a config that has drifted is caught rather than
# mislabelled -- exactly the exp5-masquerading-as-exp1 case from before.
KNOWN_EXPS = {
    (5, 500, 1000, 5, 15): "exp1",
    (5, 1000, 2000, 5, 15): "exp2",
    (5, 1000, 10000, 5, 10): "exp3",
    (15, 500, 1000, 15, 30): "exp4",
    (15, 1000, 2000, 15, 30): "exp5",
}


def current_exp():
    """Label for the threshold set currently in mcs_config."""
    key = (mcfg.thres_pr, mcfg.min_area_pr, mcfg.min_area_bt,
           mcfg.MCS_thres_pr, mcfg.MCS_thres_peak_pr)
    name = KNOWN_EXPS.get(key)
    if name is None:
        raise SystemExit(
            f"mcs_config thresholds {key} match none of the known "
            f"configurations {sorted(KNOWN_EXPS.values())}. Either restore one "
            "of them or add this set to KNOWN_EXPS with a name of its own.")
    return name


def track_month(dataset, year, month, exp):
    tag = f"{year}-{month:02d}"
    pat_pr, pat_tb = cfg.datasets[dataset]
    fin_pr, fin_tb = pat_pr.format(tag=tag), pat_tb.format(tag=tag)
    dirout = f"{cfg.path_tracking}/{dataset}/{exp}"
    fileout = f"{dirout}/Storms_{tag}.nc"

    if os.path.exists(f"{dirout}/MCS_{year}{month:02d}") and not cfg.overwrite:
        logging.info("%s %s %s already tracked, skipping", dataset, exp, tag)
        return
    for fin in (fin_pr, fin_tb):
        if not os.path.exists(fin):
            logging.warning("%s missing, skipping %s %s", os.path.basename(fin),
                            dataset, tag)
            return

    start = time.time()
    with xr.open_dataset(fin_pr) as dpr, xr.open_dataset(fin_tb) as dtb:
        pr_data = dpr.RAIN.values
        bt_data = dtb.TB.values
        lat, lon = dpr.lat.values, dpr.lon.values
        times = pd.to_datetime(dpr.time.values)
        if not np.array_equal(dpr.time.values, dtb.time.values):
            logging.error("%s %s: RAIN and TB time axes differ, skipping",
                          dataset, tag)
            return

    # The tracker treats NaN as a value; observations legitimately have gaps.
    # Rain gaps become zero (no precipitation object) and BT gaps become warm
    # (no cloud object), so a missing pixel can never create a storm.
    nan_pr, nan_bt = np.isnan(pr_data), np.isnan(bt_data)
    if nan_pr.any() or nan_bt.any():
        logging.info("%s %s: filling %.3f%% missing RAIN, %.3f%% missing TB",
                     dataset, tag, 100 * nan_pr.mean(), 100 * nan_bt.mean())
        pr_data = np.where(nan_pr, 0.0, pr_data)
        bt_data = np.where(nan_bt, 300.0, bt_data)

    os.makedirs(dirout, exist_ok=True)
    MCStracking(pr_data, bt_data, times, lon, lat,
                nc_file=fileout, path_out=dirout)

    # record what produced this, alongside the thresholds the tracker writes
    with xr.open_dataset(fileout) as dset:
        extra = dict(dset.attrs)
    extra.update(dataset=dataset, experiment=exp,
                 bt_source=os.path.basename(fin_tb),
                 pr_source=os.path.basename(fin_pr))
    with xr.open_dataset(fileout) as dset:
        dset.load().assign_attrs(extra).to_netcdf(f"{fileout}.tmp")
    os.replace(f"{fileout}.tmp", fileout)
    logging.info("%s %s %s done in %.0f s", dataset, exp, tag, time.time() - start)


def main():
    if "--list" in sys.argv:
        print("datasets:", ", ".join(cfg.datasets))
        print("current mcs_config =", current_exp())
        return
    which = [a for a in sys.argv[1:] if not a.startswith("-")]
    unknown = [d for d in which if d not in cfg.datasets]
    if not which or unknown:
        raise SystemExit(f"usage: track_storms.py <dataset> [...]\n"
                         f"unknown: {unknown}\n"
                         f"available: {', '.join(cfg.datasets)}")

    util.start_logger("track_storms")
    exp = current_exp()
    logging.info("tracking %s with %s thresholds (thres_pr=%s, min_area_pr=%s, "
                 "min_area_bt=%s, peak=%s)", which, exp, mcfg.thres_pr,
                 mcfg.min_area_pr, mcfg.min_area_bt, mcfg.MCS_thres_peak_pr)

    for dataset in which:
        months = util.months_to_do()
        Parallel(n_jobs=cfg.ntrack_jobs)(
            delayed(track_month)(dataset, y, m, exp) for y, m in months)


if __name__ == "__main__":
    main()

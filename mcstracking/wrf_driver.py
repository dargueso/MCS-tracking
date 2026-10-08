#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-03-28T11:27:09+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-03-28T11:27:14+02:00
#
# @Project@ EPICC
# Version: 2.0
# Description: Driver that ingests hourly RAIN and OLR files (WRF postprocessed
# layout, one month per file) and tracks storms with mcstracking.MCStracking.
# Run as `mcstracking-wrf` or `python -m mcstracking.wrf_driver`.
#
# Dependencies:
#
# Files:
#
# Based on Andreas Prein version 2022
# (https://colab.research.google.com/drive/1MrQFujQCFhesk0MCUSqB41Mx3AHEd1ua?usp=sharing)
#####################################################################
"""
import os
from glob import glob
import time
import logging

import xarray as xr
import pandas as pd

from joblib import Parallel, delayed

# The tracking settings come from the module resolved by load_config(): the
# one named by MCS_CONFIG (module name or .py path), else mcs_config, else the
# packaged defaults. Run-time overrides, all optional:
#   MCS_MONTHS    comma-separated YYYY-MM list: track only these months
#   MCS_NJOBS     number of months tracked in parallel (default 10)
#   MCS_PATH_OUT  output directory, instead of cfg.path_out
#   MCS_WRITE_NC  "0" skips the Storms netCDF whatever cfg.write_nc says
from .config import load_config
from .tracking import MCStracking, olr_to_tb

cfg = load_config()



#logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)

###########################################################
###########################################################
def start_logger_if_necessary():
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler('out.log', mode='w')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


###########################################################
###########################################################


def main():
    """ Main program: loops over available files and parallelize storm tracking
    """
    #start_logger_if_necessary()
    filesin = sorted(
        glob(f"{cfg.path_in}/RAIN/UIB_01H_RAIN_20??-??.nc")
    )
    months = os.environ.get("MCS_MONTHS")
    if months:
        wanted = {m.strip() for m in months.split(",") if m.strip()}
        filesin = [f for f in filesin
                   if os.path.basename(f)[len("UIB_01H_RAIN_"):-3] in wanted]
    if not filesin:
        raise SystemExit(f"no RAIN files to track under {cfg.path_in}/RAIN"
                         + (f" for MCS_MONTHS={months}" if months else ""))
    logging.info(f"{len(filesin)} months to track from {cfg.path_in}/RAIN -> {output_dir()}")

    n_jobs = int(os.environ.get("MCS_NJOBS", 10))
    Parallel(n_jobs=n_jobs)(delayed(storm_tracking)(fin_name) for fin_name in filesin)


def output_dir():
    """Where the pickles and netCDF go: MCS_PATH_OUT, else cfg.path_out."""
    return os.environ.get("MCS_PATH_OUT") or cfg.path_out


def write_nc():
    return getattr(cfg, "write_nc", True) and os.environ.get("MCS_WRITE_NC", "1") != "0"

###########################################################
###########################################################


def rain_window_label(pr):
    """Human-readable accumulation window of an hourly RAIN file."""
    if "correction" in pr.attrs and "time_bnds" in pr:
        b0, b1 = pd.to_datetime(pr.time_bnds.values[0])
        return f"clock hour, {b0:%H:%M}-{b1:%H:%M} UTC for the first step"
    return ("original cdo hoursum of end-stamped 10-min values: "
            "HH-1:50 to HH:50 (10 min early)")


def storm_tracking(pr_finname):
    """ Initialize the algorithm loading data from postprocessed WRF
    """
    logger = start_logger_if_necessary()
    logger.info(f"Analyzing {pr_finname}")
    #logging.info(f"Analyzing {pr_finname}")
    start_time = time.time()

    olr = xr.open_dataset(f"{pr_finname.replace('RAIN','OLR')}").squeeze()
    pr = xr.open_dataset(f"{pr_finname}").squeeze()
    # WSPD  = xr.open_dataset(f"{pr_finname.replace('RAIN','WSPD10')}").isel(time=slice(216,240)).squeeze()

    # Record which rain went in, read from the file itself (not asserted in the
    # config): its path, and the accumulation window its time bounds describe.
    # The clock-hour files (2026-09-29) have bounds HH:00-HH+1:00; the original
    # files have HH:00-HH:50 bounds from cdo, but their hours are 10 min early.
    cfg.rain_source = pr_finname
    cfg.rain_hour_window = rain_window_label(pr)

    pr_data = pr.RAIN.values
    # Window brightness temperature; method set in mcs_config and recorded
    # in the output attributes.
    bt_data = olr_to_tb(olr.OLR.values, cfg.bt_method)

    lat = pr.lat.values
    lon = pr.lon.values

    times = pd.date_range(pr.time.isel(time=0).values, end=pr.time.isel(time=-1).values, freq='1h')

    end_time = time.time()
    logging.debug(f"======> 'Loading data: {(end_time-start_time):.2f} seconds \n")

    ###########################################################
    ###########################################################



    # Its own directory per BT method, so an SB run and a YS run with the
    # same thresholds never overwrite one another.
    path_out = output_dir()
    os.makedirs(path_out, exist_ok=True)
    fileout = None
    if write_nc():
        fileout = f"{path_out}/" + os.path.basename(pr_finname).replace("RAIN", "Storms")


    _,_ = MCStracking(
        pr_data,
        bt_data,
        times,
        lon,
        lat,
        nc_file          =   fileout,
        path_out         =   path_out,
        cfg              =   cfg,
    )

    end_time = time.time()
    logging.info(f"======> DONE in {(end_time-start_time):.2f} seconds \n")

    #fout_name = f'{cfg.path_in}/{wrun}/Storm_properties_{sdate.year}-{sdate.month:02d}.pkl'
    #pickle.dump(grMCSs,open(fout_name,'wb'))
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
    main()

###########################################################
###########################################################

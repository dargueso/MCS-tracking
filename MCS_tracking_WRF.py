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
# Version: 1.0 (Beta)
# Description: This program ingest RAIN and OLR postprocessed files to identify
# and track storms. It uses RAIN and WINDSPEED to calculate storm statistics
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

# The tracking settings come from mcs_config.py unless MCS_CONFIG names another
# module (the sensitivity experiments use mcs_config_sens.py); mcs_config.py
# itself stays as the analyses that focus on exp1 expect it.
import importlib
cfg = importlib.import_module(os.environ.get("MCS_CONFIG", "mcs_config"))
from constants import const
from tracking_functions_optimized import MCStracking, olr_to_tb



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



    Parallel(n_jobs=10)(delayed(storm_tracking)(fin_name) for fin_name in filesin)

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

    times = pd.date_range(pr.time.isel(time=0).values, end=pr.time.isel(time=-1).values, freq='1H')

    end_time = time.time()
    logging.debug(f"======> 'Loading data: {(end_time-start_time):.2f} seconds \n")

    ###########################################################
    ###########################################################



    # Its own directory per BT method, so an SB run and a YS run with the
    # same thresholds never overwrite one another.
    os.makedirs(cfg.path_out, exist_ok=True)
    fileout = None
    if getattr(cfg, "write_nc", True):
        fileout = f"{cfg.path_out}/" + os.path.basename(pr_finname).replace("RAIN", "Storms")


    _,_ = MCStracking(
        pr_data,
        bt_data,
        times,
        lon,
        lat,
        nc_file          =   fileout,
        path_out         =   cfg.path_out,
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

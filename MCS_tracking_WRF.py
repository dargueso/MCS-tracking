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
from glob import glob
import time
import logging

import xarray as xr
import pandas as pd

from joblib import Parallel, delayed

import mcs_config as cfg
from constants import const
from tracking_functions_optimized import MCStracking



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
        glob(f"{cfg.path_in}/UIB_01H_RAIN_20??-??.nc")
    )



    Parallel(n_jobs=10)(delayed(storm_tracking)(fin_name) for fin_name in filesin)

###########################################################
###########################################################


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

    pr_data = pr.RAIN.values
    bt_data = (olr.OLR.values / const.SB_sigma) ** (0.25)

    lat = pr.lat.values
    lon = pr.lon.values

    times = pd.date_range(pr.time.isel(time=0).values, end=pr.time.isel(time=-1).values, freq='1H')

    end_time = time.time()
    logging.debug(f"======> 'Loading data: {(end_time-start_time):.2f} seconds \n")

    ###########################################################
    ###########################################################



    fileout = pr_finname.replace("RAIN", "Storms")


    _,_ = MCStracking(
        pr_data,
        bt_data,
        times,
        lon,
        lat,
        nc_file          =   fileout,
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

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
import datetime
import time

import numpy as np
import xarray as xr
import pandas as pd

from joblib import Parallel, delayed

import mcs_config as cfg
from constants import const
from tracking_functions_optimized import MCStracking

###########################################################
############# USER MODIF ##################################

wrf_runs = cfg.wrf_runs
wrun = wrf_runs[0]
FREQ = "01H"

DT = 1

############# END OF USER MODIF ###########################
###########################################################



###########################################################
###########################################################


def main():
    """ Main program: loops over available files and parallelize storm tracking
    """

    filesin = sorted(
        glob(f"{cfg.path_in}/{wrun}/{cfg.patt_in}_{FREQ}_RAIN_201[3-4]-09.nc")
    )
    Parallel(n_jobs=1)(delayed(storm_tracking)(fin_name) for fin_name in filesin)


###########################################################
###########################################################


def storm_tracking(pr_finname):
    """ Initialize the algorithm loading data from postprocessed WRF
    """
    olr = (
        xr.open_dataset(f"{pr_finname.replace('RAIN','OLR')}")
        .isel(time=slice(216, 240))
        .squeeze()
    )
    rain = xr.open_dataset(f"{pr_finname}").isel(time=slice(216, 240)).squeeze()
    # WSPD  = xr.open_dataset(f"{pr_finname.replace('RAIN','WSPD10')}").isel(time=slice(216,240)).squeeze()

    bt = (olr.OLR.values / const.SB_sigma) ** (0.25)

    data_all = np.stack([rain.RAIN.values, bt], axis=3)

    lat = rain.lat.values
    lon = rain.lon.values

    sdate = datetime.datetime(
        int(rain.time.isel(time=0).dt.year),
        int(rain.time.isel(time=0).dt.month),
        int(rain.time.isel(time=0).dt.day),
        int(rain.time.isel(time=0).dt.hour),
    )
    edate = datetime.datetime(
        int(rain.time.isel(time=-1).dt.year),
        int(rain.time.isel(time=-1).dt.month),
        int(rain.time.isel(time=-1).dt.day),
        int(rain.time.isel(time=-1).dt.hour),
    )
    times = pd.date_range(sdate, end=edate, freq=FREQ)

    ###########################################################
    ###########################################################

    start_time = time.time()

    fileout = pr_finname.replace("RAIN", "Storms")
    _,_ = MCStracking(
        data_all,
        times,
        lon,
        lat,
        cfg.Variables,
        DT,
        SmoothSigmaP    =   cfg.SmoothSigmaP,
        Pthreshold      =   cfg.Pthreshold,
        MinTimePR       =   cfg.MinTimePR,
        MinAreaPR       =   cfg.MinAreaPR,
        SmoothSigmaC    =   cfg.SmoothSigmaC,
        Cthreshold      =   cfg.Cthreshold,
        MinTimeC        =   cfg.MinTimeC,
        MinAreaC        =   cfg.MinAreaC,
        MCS_Minsize     =   cfg.MCS_Minsize,
        MCS_minPR       =   cfg.MCS_minPR,
        MCS_MinPeakPR   =   cfg.MCS_MinPeakPR,
        CL_MaxT         =   cfg.CL_MaxT,
        CL_Area         =   cfg.CL_Area,
        MCS_minTime     =   cfg.MCS_minTime,
        NCfile          =   fileout,
    )

    end_time = time.time()
    print(f"======> DONE in {(end_time-start_time):.2f} seconds \n")

    #fout_name = f'{cfg.path_in}/{wrun}/Storm_properties_{sdate.year}-{sdate.month:02d}.pkl'
    #pickle.dump(grMCSs,open(fout_name,'wb'))
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    main()

###########################################################
###########################################################

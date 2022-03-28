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

import numpy as np
import xarray as xr
from glob import glob
import time
import pickle

from joblib import Parallel, delayed

import epicc_config as cfg
from constants import const as const
from Tracking_Functions_optimized import MCStrack

###########################################################
############# USER MODIF ##################################

wrf_runs = cfg.wrf_runs
syear = cfg.syear
eyear = cfg.eyear

# Tracking parameters
dT = 1                    # temporal resolution of data for tracking in hours

# MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# precipitation tracking options
SmoothSigmaP = 0          # Gaussion std for precipitation smoothing
Pthreshold = 5            # precipitation threshold [mm/h]
MinTimePR = 3             # minum lifetime of PR feature in hours
MinAreaPR = 500          # minimum area of precipitation feature in km2

# Brightness temperature (Tb) tracking setup
SmoothSigmaC = 0          # Gaussion std for Tb smoothing
Cthreshold = 241          # minimum Tb of cloud shield
MinTimeC = 3              # minium lifetime of cloud shield in hours
MinAreaC = 5000          # minimum area of cloud shield in km2

# MCs detection
MCS_Minsize = MinAreaPR   # minimum area of MCS precipitation object in km2
MCS_minPR = 10            # minimum max precipitation in mm/h
MCS_MinPeakPR = 10        # Minimum lifetime peak of MCS precipitation
CL_MaxT = 225             # minimum brightness temperature
CL_Area = MinAreaC        # min cloud area size in km2
MCS_minTime = 4           # minimum lifetime of MCS

############# END OF USER MODIF ###########################
###########################################################

wrun = wrf_runs[0]

###########################################################
###########################################################

filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))



###########################################################
###########################################################

OLR   = xr.open_dataset(f'{cfg.path_in}/{wrun}/UIB_01H_OLR_2013-09.nc').squeeze()
RAIN  = xr.open_dataset(f'{cfg.path_in}/{wrun}/UIB_01H_RAIN_2013-09.nc').squeeze()
#WSPD  = xr.open_dataset(f'{cfg.path_in}/{wrun}/UIB_01H_WSPD10_2013-09.nc').squeeze()


BT = (OLR.OLR.values/const.SB_sigma)**(0.25)

DATA_all = np.stack([RAIN.RAIN.values,BT], axis=3)

Lat = RAIN.lat.values
Lon = RAIN.lon.values

StartDay = datetime.datetime(RAIN.time.isel(time=0).dt.year,RAIN.time.isel(time=0).dt.month,RAIN.time.isel(time=0).dt.day,RAIN.time.isel(time=0).dt.hour)
StopDay = datetime.datetime(RAIN.time.isel(time=-1).dt.year,RAIN.time.isel(time=-1).dt.month,RAIN.time.isel(time=-1).dt.day,RAIN.time.isel(time=-1).dt.hour)
Time = pd.date_range(StartDay, end=StopDay, freq='1H')

###########################################################
###########################################################

NCfile = f'EPICC-MCS-tracking_{year}{month:02d}_.nc'
grMCSs, MCS_obj = MCStracking(DATA_all,
                              Time,
                              Lon,
                              Lat,
                              Variables,
                              dT,
                              SmoothSigmaP = SmoothSigmaP,
                              Pthreshold = Pthreshold,
                              MinTimePR = MinTimePR,
                              MinAreaPR = MinAreaPR,
                              SmoothSigmaC = SmoothSigmaC,
                              Cthreshold = Cthreshold,
                              MinTimeC = MinTimeC,
                              MinAreaC = MinAreaC,
                              MCS_Minsize = MCS_Minsize,
                              MCS_minPR = MCS_minPR,
                              MCS_MinPeakPR = MCS_MinPeakPR,
                              CL_MaxT = CL_MaxT,
                              CL_Area = CL_Area,
                              MCS_minTime = MCS_minTime,
                              NCfile = NCfile)

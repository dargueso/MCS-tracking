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

###########################################################
###########################################################



###########################################################
###########################################################


############# END OF USER MODIF ###########################
###########################################################

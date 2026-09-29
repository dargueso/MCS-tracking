#!/usr/bin/env python
"""
Configuration for the combined AEMET rain-gauge dataset and its use to evaluate
EPICC (and to check EURADCLIM).

Three AEMET sources, all measuring with the same automatic stations, combined
into ONE quality-controlled 10-minute dataset, from which the clock-hour
dataset used in this study is derived:

  HyMEX      10-min rain, real-time AWS database, NOT quality controlled,
             2010-2022, ~395 stations (Catalonia, Ebro, Valencia, Murcia,
             Balearics). Restricted to HyMeX projects.
  Balearic   10-min rain from AEMET's historical archive (cuenca B),
             2009-2024, 47 stations.
  Arnau      daily totals plus maximum 10/20/30/60-min and 2/6/12-h amounts
             per day, AEMET climatological database WITH quality flags,
             2000-2019, ~400 stations. Mostly derived from the same 10-min
             record (ID_FLAG_P = 1 for 87% of days).

Arnau is therefore the quality control for the 10-min data: a station-day is
accepted only where it reproduces AEMET's own validated daily total and
maxima. See make_station_dataset.py for the rules.

    python make_station_dataset.py      # -> 10-min, 1-h and Arnau daily netCDF
    python extract_at_stations.py       # model (and EURADCLIM) at the stations
    python plot_station_model.py        # evaluation figures and numbers
"""

import os
import sys

import obs_config as ocfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as mcfg

###########################################################
# Inputs
###########################################################

path_obs_raw = "/scratch3/dargueso/OBS"
path_hymex = f"{path_obs_raw}/AEMET_HyMEX"          # prec_10min_all_stations_YYYY.pkl
hymex_zip = f"{path_obs_raw}/AEMET/AEMET_HyMEX.zip"   # 0_Documentation/AWS_AEMET-stations.csv
path_balear = f"{path_obs_raw}/AEMET/BALEARES_10min_2009-2024/datos"
balear_master = f"{path_balear}/Maestrohistorico_diezmin_cuenca_B_2009_2024_Cuenca_B.csv"
path_arnau_rar = f"{path_obs_raw}/Arnau"            # .rar archives, unpacked on first use
arnau_archives = ["Arnau_2010_2014", "Arnau_2015_2019", "Arnau_Balears_2000_2019"]
path_hrly_meta = f"{path_obs_raw}/AEMET/hrly_obs"    # headers used only for coordinates

###########################################################
# Period and stations
###########################################################

syear, eyear = 2011, 2020     # the EPICC evaluation period; Arnau ends in 2019
# The dataset keeps every station that has coordinates: it is not tied to the
# model or to WME. The evaluation selects stations inside the model grid and
# its region.

###########################################################
# Outputs
###########################################################

# Everything about the stations lives under one directory:
#   STATIONS/AEMET_combined/   the database itself - data files, station list,
#                              README and build summary, nothing else
#   STATIONS/evaluation/       model (and radar) at the stations, figures, numbers
path_st = f"{ocfg.path_obs}/STATIONS"
path_db = f"{path_st}/AEMET_combined"
path_arnau = f"{path_db}/arnau_raw"                  # unpacked Arnau text files
file_10min = f"{path_db}/AEMET_10MIN_PREC_{syear}-{eyear}.nc"
file_01h = f"{path_db}/AEMET_01H_PREC_{syear}-{eyear}.nc"
file_daily = f"{path_db}/AEMET_DAILY_{syear}-{eyear}.nc"
file_arnau = f"{path_db}/AEMET_ARNAU_DAILY_{syear}-2019.nc"
file_stations = f"{path_db}/stations.csv"
path_eval = f"{path_st}/evaluation"
path_st_figs = path_eval

###########################################################
# Time convention
###########################################################
#
# Both 10-min sources stamp the END of the interval (AEMET convention). Measured,
# not assumed: daily sums of the stamps 00:10..24:00 reproduce Arnau's daily
# total exactly (to 0.05 mm) on 99.6% of rain days for HyMEX and 100% for the
# Balearic archive, against 85% and 75% for stamps 00:00..23:50.
# The OUTPUT is stamped at the START of the interval, like every other file in
# this project: the value at 13:00 is 13:00-13:10 UTC, and the hourly value at
# 13:00 is 13:00-14:00 UTC.
source_stamp = "end"

###########################################################
# Quality control
###########################################################
#
# Day tiers (per station and day), written to the files:
TIER_NONE, TIER_A, TIER_B, TIER_REJ_ARNAU, TIER_REJ_AUTO = 0, 1, 2, 8, 9
tier_meaning = {
    TIER_NONE: "no data",
    TIER_A: "verified: reproduces Arnau (AEMET-validated) daily total and maxima",
    TIER_B: "automatic checks only (no usable Arnau record for that day)",
    TIER_REJ_ARNAU: "rejected: disagrees with Arnau, or Arnau flags it doubtful",
    TIER_REJ_AUTO: "rejected: failed an automatic check",
}

# Arnau agreement, tier A. Tolerances allow for rounding to 0.1 mm only: Arnau
# is computed from the same 10-min record, so a larger difference means AEMET
# corrected the day (ID_FLAG_E = 10) and the raw copy is wrong.
tol_abs = 0.2            # mm
tol_rel = 0.05           # fraction of the Arnau value
arnau_good_flags = (0, 1)        # validated manually / automatically
arnau_bad_flags = (20, 21)       # doubtful -> reject the day

# Automatic checks, tier B (and applied to tier A days as well).
# The largest AEMET-validated 10-min amount in 2011-2019 in these data is
# 44.7 mm (60-min: 159 mm), so the cap sits just above it: a lower cap would
# remove real extremes.
max_10min = 50.0         # mm in 10 min
max_day = 400.0          # mm in a day
# A non-zero value repeated unchanged this many times in a row is a stuck sensor.
stuck_min_value, stuck_steps = 0.5, 6
# An isolated spike: >= spike_value with nothing within +-spike_window steps.
spike_value, spike_window = 10.0, 3

# Valid 10-min steps for a day to count as complete. A complete day that fails
# to reproduce Arnau is rejected; an incomplete one below Arnau's total is only
# unverifiable (tier B), since the missing rain may be in the gaps.
min_steps_day = 144

###########################################################
# Evaluation
###########################################################

model_run = ocfg.model_runs["pres"]
# Clock-hour model rain (HH:00-HH+1:00), the definitive version on /scratch3.
# (The original, 10-min-early files on /scratch1 are not used: README, "Timing checks".)
path_model_rain = f"{ocfg.path_model}/{model_run}/RAIN"
geofile = mcfg.geofile_ref
# Neighbourhood for the model value at a station: the station's cell, and the
# max / mean over a (2k+1)^2 block around it, which brackets small location
# errors in where the model puts the rain.
nbhd_half = 1
eval_tiers = (TIER_A, TIER_B)    # which day tiers the evaluation accepts
subregions = {
    "ALL": [ocfg.lat_min, ocfg.lon_min, ocfg.lat_max, ocfg.lon_max],
    "CAT": mcfg.reg_coords["CAT"],
    "LEV": mcfg.reg_coords["LEV"],
    "BAL": mcfg.reg_coords["BAL"],
}
wet_thres = 0.1                  # mm/h
durations_h = [1, 2, 3, 6, 12, 24]

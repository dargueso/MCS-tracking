

# All EPICC postprocessed input is on /scratch3 (moved 2026-09-29). Its RAIN/
# holds the clock-hour rain (HH:00-HH+1:00); the original, 10-min-early RAIN
# stays on /scratch1 only for manuscripts that used it. Keep path_in on one
# line: run_*_tracking.sh rewrite it with sed.
path_in = "/scratch3/dargueso/postprocessed/EPICC/EPICC_2km_ERA5"
wrun = "EPICC_2km_ERA5"
#path_in = "/home/dargueso/Scripts/MCS-tracking"

## Brightness temperature from OLR
#
# "YS"  Ohring et al. (1984) as given in Yang and Slingo (2001). The 241 K and
#       225 K cloud thresholds below come from the satellite MCS literature and
#       are defined on WINDOW brightness temperature, which is what this returns.
# "SB"  Tb = (OLR/sigma)^0.25, what exp1..exp5 were produced with. It returns a
#       broadband effective emission temperature, ~22 K colder than a window Tb
#       at OLR 240, so the thresholds do not mean there what they mean in the
#       literature they came from.
# The method is written into the tracked netCDF attributes, so runs that differ
# only in this are distinguishable from the files themselves.
bt_method = "YS"

# Tracked output. exp<N> keeps meaning the threshold set and nothing else; the
# BT method is a separate axis, so it names the directory rather than the exp.
exp_label = "exp1"

# Whether to write the full Storms netCDF (3D object masks plus PR and BT).
# Every storm property used by the analysis comes from the MCS_/PR_/BT_ pickles,
# which are a few hundred kB per month; the netCDF is ~2.2 GB per month and is
# only needed to plot object masks or look at individual cases. At 240
# run-months that is ~520 GB, which is why the output goes to /scratch3 rather
# than beside the input.
write_nc = True

# Tracked output sits beside the input on /scratch3.
path_track_root = "/scratch3/dargueso/postprocessed/EPICC"
path_out = f"{path_track_root}/{wrun}/ConvStormTracking_{bt_method}/{exp_label}"

## MCS config
#
# Reference configuration: exp1 (ST1 in the manuscript). These values are the
# ones recorded in the global attributes of the tracked output actually used
# for the paper:
#   EPICC_2km_ERA5/ConvStormTracking/exp1           (reference)
#   EPICC_2km_ERA5_CMIP6anom/ConvStormTracking/exp1 (PGW)
# Both runs used identical settings. The other configurations on disk are:
#   exp1 (ST1): thres_pr  5, min_area_pr  500, min_area_bt  1000, peak 15
#   exp2 (ST2): thres_pr  5, min_area_pr 1000, min_area_bt  2000, peak 15
#   exp3 (ST3): thres_pr  5, min_area_pr 1000, min_area_bt 10000, peak 10
#   exp4      : thres_pr 15, min_area_pr  500, min_area_bt  1000, peak 30
#   exp5      : thres_pr 15, min_area_pr 1000, min_area_bt  2000, peak 30
#
DT = 1 # time step of data in hours
Variables = ["PR", "Tb"]

# MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# precipitation tracking options
smooth_sigma_pr = 0 # Gaussion std for precipitation smoothing
thres_pr = 5  # precipitation threshold [mm/h]
min_time_pr= 3  # minum lifetime of PR feature in hours
min_area_pr = 500  # minimum area of precipitation feature in km2

# Brightness temperature (Tb) tracking setup
smooth_sigma_bt = 0  # Gaussion std for Tb smoothing
thres_bt = 241  # minimum Tb of cloud shield
min_time_bt = 5  # minium lifetime of cloud shield in hours
min_area_bt = 1000  # minimum area of cloud shield in km2

# MCs detection
MCS_min_area = min_area_pr  # minimum area of MCS precipitation object in km2
MCS_thres_pr = 5  # minimum max precipitation in mm/h
MCS_thres_peak_pr = 15  # Minimum lifetime peak of MCS precipitation
MCS_thres_bt = 225  # minimum brightness temperature
MCS_min_area_bt = min_area_bt  # min cloud area size in km2
MCS_min_time = 5  # minimum lifetime of MCS





#


###########################################################
## Analysis and plotting
###########################################################
#
# These were previously imported from epicc_config, which lives in a separate
# repository (EPICC_scripts). The tracking has no business depending on it, so
# the handful of values the analysis actually needs are kept here. Copied from
# epicc_config on 2026-09-28; if the EPICC paths or region boxes move, they have
# to be updated here too.
#
# NOTE the names: `path_in` and `path_out` above mean the run being tracked and
# where its output goes. These are deliberately different names, because they
# mean different things - `path_postproc` is the root holding every run.

path_postproc = "/scratch3/dargueso/postprocessed/EPICC"     # NOT /home/dargueso/postprocessed: that links to /scratch1
path_figures = "/home/dargueso/Analyses/EPICC"

geoem_in = "/home/dargueso/share/geo_em_files/EPICC"
geofile_ref = f"{geoem_in}/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"

wrf_runs = ["EPICC_2km_ERA5", "EPICC_2km_ERA5_CMIP6anom"]

# [lat_min, lon_min, lat_max, lon_max]
reg_coords = {
    "BAL": [38.6, 0.9, 40.3, 4.7],
    "LEV": [36.4, -3.30, 40.20, 1.0],
    "CAT": [40.3, 0, 43, 3.5],
    "SAR": [38.7, 7.74, 41.35, 10.3],
    "WME": [36, -5, 45, 15],
    "SWM": [36, -5, 40.3, 15],
    "NWM": [40.5, -5, 45.0, 15],
}

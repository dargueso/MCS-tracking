#!/usr/bin/env python
"""
Configuration for the observational (IMERG + MERGIR) branch of the
convective-storm tracking, used for the evaluation of the EPICC simulations.

Both datasets come from NASA GES DISC and need Earthdata Login credentials in
~/.netrc:

    machine urs.earthdata.nasa.gov
        login <username>
        password <password>

(chmod 600 ~/.netrc). A cookie jar at ~/.urs_cookies is created automatically.
"""

###########################################################
# Region and period
###########################################################

# Tracking region. These are epicc_config.reg_coords["WME"] = [lat0,lon0,lat1,lon1]
reg = "WME"
lat_min, lat_max = 36.0, 45.0
lon_min, lon_max = -5.0, 15.0

# Extra margin around the region. Storms are tracked on the larger domain and
# only afterwards filtered on where their centre lies, exactly as is done with
# the model output, so that systems entering the region are already tracked.
margin = 1.0  # degrees

# Period. The manuscript Methods state 2011-2020 for the evaluation.
syear, eyear = 2011, 2020

# Months to download. Whole years: the analysis currently focuses on ASO, but
# the study is expected to extend to other months, and this also matches the
# "2011-2020" in the Methods, which does not restrict the season.
months = list(range(1, 13))

###########################################################
# Paths
###########################################################

# Everything lives on /scratch3: /me01 (which is where /home/dargueso sits) is
# at ~1.76 TB of a 2 TB quota, while /scratch3 has ~8.9 TB free.
path_obs = "/scratch3/dargueso/obs-mcs-tracking"
path_imerg = f"{path_obs}/IMERG"          # half-hourly subsets, native 0.1 deg
path_mergir = f"{path_obs}/MERGIR"        # half-hourly subsets, native 4 km
path_track = f"{path_obs}/ConvStormTracking"  # hourly, common grid, ready to track
# All figures (and the numbers/CSV files that go with them), one subfolder per
# evaluation: satellite, radar, stations.
path_figs = f"{path_obs}/figures"
path_figs_sat = f"{path_figs}/satellite"

# Scratch space for the full IMERG granules before subsetting. They are deleted
# as soon as the region has been extracted, so this never holds more than
# nworkers files at a time.
path_scratch = f"{path_obs}/scratch"

###########################################################
# Download behaviour
###########################################################

nworkers = 4      # parallel downloads. Do not raise this: GES DISC throttles
                  # bursts, and measured on 24 IMERG granules, 4 workers gave
                  # 0.84 s/granule while 8 gave 1.97 s and 12 gave 1.79 s --
                  # both spent more time backing off from 503s than they gained
                  # in concurrency. IMERG and MERGIR sit on different hosts
                  # (gpm1 / disc2), so the two download scripts CAN be run at
                  # the same time without competing for the same throttle.
nretries = 5      # attempts per file, with jittered backoff, before giving up
timeout = 300     # seconds per request
overwrite = False # True re-downloads months that are already complete

###########################################################
# Remote collections (verified against GES DISC)
###########################################################

# IMERG Final Run V07, half-hourly, 0.1 deg global.
# Downloaded as whole granules and subsetted locally: measured ~1.3-3 s per
# 7.7 MB granule, against ~18 s for the equivalent OPeNDAP subset request.
imerg_root = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"
imerg_version = "V07B"

# MERGIR (NCEP/CPC merged IR), half-hourly, 4 km, 60S-60N.
# Fetched through OPeNDAP: a region subset is ~139 kB in ~1 s, against 27 MB in
# ~20 s for the whole file, so here the subsetter is the clear win.
mergir_opendap = "https://disc2.gesdisc.eosdis.nasa.gov/opendap/MERGED_IR/GPM_MERGIR.1"

###########################################################
# Model side of the comparison
###########################################################

# The EPICC 2 km output is coarsened onto exactly the same 0.1 deg grid as the
# observations, so the tracker can be run with identical settings on both and
# the storm statistics compared like for like.
# /scratch3: RAIN there is the clock-hour rain. (/home/dargueso/postprocessed
# links to /scratch1, which keeps only the original, 10-min-early rain.)
path_model = "/scratch3/dargueso/postprocessed/EPICC"
model_runs = {"pres": "EPICC_2km_ERA5", "fut": "EPICC_2km_ERA5_CMIP6anom"}
path_modcoarse = f"{path_obs}/MODEL_0.1deg"   # then /<wrun>/MOD_01H_{RAIN,TB}_YYYY-MM.nc

# How OLR is turned into brightness temperature, for the coarsened model.
# Both are written, as MOD_01H_TB_<method>_YYYY-MM.nc, so either can be tracked
# without redoing the coarsening.
#   SB  plain Stefan-Boltzmann, Tb = (OLR/sigma)^0.25. What MCS_tracking_WRF.py
#       does, and what produced exp1.
#   YS  Ohring et al. (1984) as given in Yang and Slingo (2001), the standard
#       correction for comparing model OLR with window-channel Tb. MERGIR
#       reports 10.8 um window Tb, which SB underestimates by ~22 K at OLR 240
#       while agreeing to within 2 K in cold cloud (OLR 90).
bt_methods = ["SB", "YS"]

# Process the time axis in chunks of this many hours. A month of 2 km RAIN is
# 720 x 749 x 1249 floats (~2.7 GB), too much to hold at once alongside OLR.
chunk_hours = 48

# Target grid for the comparison: the native IMERG 0.1 deg grid, as stated in
# the Methods. MERGIR is block-averaged onto it by make_obs_tracking_input.py.
target_res = 0.1


###########################################################
# Tracking datasets
###########################################################
#
# Storm tracking is indexed on two independent axes:
#
#   dataset  what was tracked  (obs, coarsened model, native 2 km model)
#   exp      the tracker threshold configuration (exp1..exp5 in mcs_config)
#
# They are kept separate on purpose. exp1..exp5 are recorded in the global
# attributes of the tracked output, so two runs that differ only in, say, the
# OLR-to-BT conversion would carry byte-identical tracker attributes and be
# indistinguishable from the files themselves. Naming such a run "exp1b" would
# hide that difference in a label that says nothing about what changed; naming
# the dataset instead keeps every exp label meaning exactly one thing.
#
# Output goes to {path_tracking}/{dataset}/{exp}/, holding the usual
# PR_/BT_/MCS_ pickles plus the Storms netCDF.

path_tracking = f"{path_obs}/tracking"

# Months tracked in parallel. The 0.1 deg grid is small (110 x 220), so the
# limit is memory per worker rather than CPU.
ntrack_jobs = 6

# dataset -> (rain file pattern, brightness temperature file pattern)
def _mod(wrun, method):
    d = f"{path_modcoarse}/{wrun}"
    return (f"{d}/MOD_01H_RAIN_{{tag}}.nc", f"{d}/MOD_01H_TB_{method}_{{tag}}.nc")

datasets = {
    "obs": (f"{path_track}/OBS_01H_RAIN_{{tag}}.nc",
            f"{path_track}/OBS_01H_TB_{{tag}}.nc"),
}
for _key, _wrun in model_runs.items():
    for _m in bt_methods:
        datasets[f"mod0.1_{_m}_{_key}"] = _mod(_wrun, _m)
# Symmetric radar-based pair (make_radar_tracking_input.py): EURADCLIM coarsened
# to 0.1 deg with MERGIR, and the present-day model cut to the radar coverage
# with its YS Tb. Same storm definition, same area and hours on both sides.
datasets["rad"] = (f"{path_track}/RADCOV_01H_RAIN_{{tag}}.nc",
                   f"{path_track}/OBS_01H_TB_{{tag}}.nc")
datasets["mod0.1_YS_pres_radcov"] = (
    f"{path_modcoarse}/{model_runs['pres']}/MODRADCOV_01H_RAIN_{{tag}}.nc",
    f"{path_modcoarse}/{model_runs['pres']}/MOD_01H_TB_YS_{{tag}}.nc")

# The native 2 km tracking is not run from here: it already lives under
# {path_model}/<wrun>/ConvStormTracking/<exp>/ and keeps that layout.

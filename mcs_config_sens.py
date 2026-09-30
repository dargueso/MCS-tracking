"""
Sensitivity experiments of the storm tracking, exp6 onwards.

exp1..exp5 (mcs_config.py, run_st2_st5.sh) raise the entry bar: intensity,
area, cloud shield. These change the definition of a storm in kind, each
starting from the exp1 reference:

    exp6   rain only: no cold-cloud shield required (require_bt = False)
    exp7   loose: thres_pr 3 mm/h, min_area_pr 250 km2, min_time_pr 2 h,
           MCS_thres_pr 3 mm/h, peak 10 mm/h
    exp8   persistence: min_time_pr 6 h, min_time_bt 8 h, MCS_min_time 8 h
    exp9   smoothing: Gaussian sigma 1 cell on rain and Tb before thresholding
    exp10  linking: objects continue in time only with >= 30% overlap
           (2-D components linked step by step; the reference links any
           touching cell through a 3-D labelling)

Selected by environment variables, so mcs_config.py is never touched:

    MCS_CONFIG=mcs_config_sens SENS_EXP=exp6 MCS_RUN=EPICC_2km_ERA5 python MCS_tracking_WRF.py
    MCS_CONFIG=mcs_config_sens SENS_EXP=exp6 python Observations/track_storms.py obs mod0.1_YS_pres

Every tracking setting is written here explicitly (the exp1 values, then the
experiment's changes), so the result does not depend on what mcs_config.py
holds at the time. Paths, region boxes and the like are taken from
mcs_config.py.
"""

import os
from mcs_config import *          # noqa: F401,F403  paths, DT, geofile, regions, ...

SENS = {
    "exp6": dict(require_bt=False),
    "exp7": dict(thres_pr=3, MCS_thres_pr=3, min_area_pr=250, min_time_pr=2, MCS_thres_peak_pr=10),
    "exp8": dict(min_time_pr=6, min_time_bt=8, MCS_min_time=8),
    "exp9": dict(smooth_sigma_pr=1, smooth_sigma_bt=1),
    "exp10": dict(min_overlap=0.3),
}
DESCRIPTION = {
    "exp6": "rain only, no cloud shield required",
    "exp7": "loose thresholds: 3 mm/h, 250 km2, 2 h, peak 10 mm/h",
    "exp8": "persistence: 6 h rain, 8 h cloud, 8 h storm",
    "exp9": "Gaussian smoothing, sigma 1 cell",
    "exp10": "linking with >= 30% overlap",
}

exp_label = os.environ.get("SENS_EXP", "exp6")
if exp_label not in SENS:
    raise SystemExit(f"SENS_EXP={exp_label!r}: choose one of {sorted(SENS)}")

# the exp1 reference, stated explicitly
DT = 1
smooth_sigma_pr = 0
thres_pr = 5
min_time_pr = 3
min_area_pr = 500
smooth_sigma_bt = 0
thres_bt = 241
min_time_bt = 5
min_area_bt = 1000
MCS_thres_pr = 5
MCS_thres_peak_pr = 15
MCS_thres_bt = 225
MCS_min_time = 5
require_bt = True
min_overlap = 0.0

globals().update(SENS[exp_label])
MCS_min_area = min_area_pr
MCS_min_area_bt = min_area_bt

bt_method = "YS"
wrun = os.environ.get("MCS_RUN", "EPICC_2km_ERA5")
path_in = f"{path_track_root}/{wrun}"
path_out = f"{path_track_root}/{wrun}/ConvStormTracking_{bt_method}/{exp_label}"
write_nc = True

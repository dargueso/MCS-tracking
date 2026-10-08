"""Example tracking configuration.

This is the module the tracker uses when MCS_CONFIG is not set and this file
is on sys.path or in the working directory. Copy it, edit the paths (and the
thresholds if needed) and point MCS_CONFIG at your copy:

    MCS_CONFIG=/path/to/my_config.py mcstracking-wrf

All thresholds and options, with the reference (exp1) values, are documented
in mcstracking/default_config.py.
"""
from mcstracking.default_config import *   # noqa: F401,F403

# Input: {path_in}/RAIN/UIB_01H_RAIN_YYYY-MM.nc (variable RAIN, mm/h) and
#        {path_in}/OLR/UIB_01H_OLR_YYYY-MM.nc  (variable OLR, W m-2),
# one month per file, hourly, with 2-D `lat`/`lon` (see README, "Input").
path_in = "example_input"
wrun = "example_run"

bt_method = "YS"     # "YS" (Yang & Slingo, window Tb) or "SB" (grey-body inversion)
exp_label = "exp1"   # the threshold set; name a different set differently
write_nc = True      # Storms_YYYY-MM.nc with the object masks (large: ~2 GB per month at 2 km)

path_out = f"tracking_output/{wrun}/ConvStormTracking_{bt_method}/{exp_label}"

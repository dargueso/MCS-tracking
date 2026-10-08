"""Reference tracking configuration (exp1 of the EPICC Mediterranean storm study).

Every setting the tracker reads, with the reference value. A user configuration
module normally starts with ``from mcstracking.default_config import *`` and
overrides what it needs; it must also set the paths at the bottom.
"""

DT = 1                     # time step of the input data in hours
Variables = ["PR", "Tb"]

# Precipitation objects
smooth_sigma_pr = 0        # Gaussian std (cells) for precipitation smoothing; 0 = none
thres_pr = 5               # precipitation threshold [mm/h]
min_time_pr = 3            # minimum lifetime of a precipitation feature [h]
min_area_pr = 500          # minimum area of a precipitation feature [km2]

# Brightness-temperature (cloud shield) objects
smooth_sigma_bt = 0        # Gaussian std (cells) for Tb smoothing; 0 = none
thres_bt = 241             # maximum Tb of the cloud shield [K]
min_time_bt = 5            # minimum lifetime of a cloud shield [h]
min_area_bt = 1000         # minimum area of a cloud shield [km2]

# Storm (MCS) detection
MCS_min_area = min_area_pr      # minimum area of the storm precipitation object [km2]
MCS_thres_pr = 5                # minimum maximum precipitation [mm/h]
MCS_thres_peak_pr = 15          # minimum lifetime peak precipitation [mm/h]
MCS_thres_bt = 225              # cold-core brightness temperature [K]
MCS_min_area_bt = min_area_bt   # minimum cloud-shield area [km2]
MCS_min_time = 5                # minimum lifetime of a storm [h]

# Options (reference behaviour)
require_bt = True          # False: rain-only storms, no cloud shield required
min_overlap = 0.0          # >0: objects continue in time only with this overlap fraction

# Provenance written into the output netCDF attributes
bt_method = "YS"           # OLR -> Tb conversion used for bt_data: "YS" (Yang & Slingo) or "SB" (Stefan-Boltzmann)
exp_label = "exp1"         # name of this threshold set
write_nc = True            # write the Storms_YYYY-MM.nc file (object masks, PR, BT)

# Paths used by the WRF driver (mcstracking-wrf)
wrun = "unknown"                       # run name, written into the netCDF `source` attribute
path_in = "."                          # holds RAIN/UIB_01H_RAIN_YYYY-MM.nc and OLR/UIB_01H_OLR_YYYY-MM.nc
path_out = "./tracking_output"         # where PR_/BT_/MCS_ pickles and Storms netCDF go

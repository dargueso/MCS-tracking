
#path_in = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/EPICC_2km_ERA5_HVC_GWD/"
path_in = "/Users/daniel/Scripts_local/MCS-tracking"
## MCS config
DT = 1 # time step of data in hours
Variables = ["PR", "Tb"]

# MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# precipitation tracking options
smooth_sigma_pr = 0 # Gaussion std for precipitation smoothing
thres_pr = 5  # precipitation threshold [mm/h]
min_time_pr= 3  # minum lifetime of PR feature in hours
min_area_pr = 1000  # minimum area of precipitation feature in km2

# Brightness temperature (Tb) tracking setup
smooth_sigma_bt = 0  # Gaussion std for Tb smoothing
thres_bt = 241  # minimum Tb of cloud shield
min_time_bt = 5  # minium lifetime of cloud shield in hours
min_area_bt = 10000  # minimum area of cloud shield in km2

# MCs detection
MCS_min_area = min_area_pr  # minimum area of MCS precipitation object in km2
MCS_thres_pr = 10  # minimum max precipitation in mm/h
MCS_thres_peak_pr = 10  # Minimum lifetime peak of MCS precipitation
MCS_thres_bt = 225  # minimum brightness temperature
MCS_min_area_bt = min_area_bt  # min cloud area size in km2
MCS_min_time = 5  # minimum lifetime of MCS





#

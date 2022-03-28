
path_in = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/EPICC_2km_ERA5_HVC_GWD/"

## MCS config
DT = 1 # time step of data in hours
Variables = ["PR", "Tb"]

# MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# precipitation tracking options
SmoothSigmaP = 0 # Gaussion std for precipitation smoothing
Pthreshold = 5  # precipitation threshold [mm/h]
MinTimePR = 3  # minum lifetime of PR feature in hours
MinAreaPR = 500  # minimum area of precipitation feature in km2

# Brightness temperature (Tb) tracking setup
SmoothSigmaC = 0  # Gaussion std for Tb smoothing
Cthreshold = 241  # minimum Tb of cloud shield
MinTimeC = 3  # minium lifetime of cloud shield in hours
MinAreaC = 5000  # minimum area of cloud shield in km2

# MCs detection
MCS_Minsize = MinAreaPR  # minimum area of MCS precipitation object in km2
MCS_minPR = 10  # minimum max precipitation in mm/h
MCS_MinPeakPR = 10  # Minimum lifetime peak of MCS precipitation
CL_MaxT = 225  # minimum brightness temperature
CL_Area = MinAreaC  # min cloud area size in km2
MCS_minTime = 4  # minimum lifetime of MCS





#

path_in = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/"
patt_in = "UIB"
path_out = "/home/dargueso/Analyses/EPICC/"
path_cmaps = "/home/dargueso/share/colormaps/"
geoem_in = "/home/dargueso/share/geo_em_files/EPICC"
path_bdy = "/home/dargueso/OBS_DATA/ERA5/"
patt_bdy = "era5_monthly_prec"
bdy_data = "ERA5"
path_wrfout = "/vg6/dargueso-NO-BKUP/WRF_OUT/EPICC"
path_postproc = "/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/"

syear = 2013
eyear = 2020

smonth = 1
emonth = 12

wrf_runs = ['EPICC_2km_ERA5_HVC_GWD','EPICC_2km_ERA5_CMIP6anom_HVC_GWD']
wrun_ref = 'EPICC_2km_ERA5_HVC_GWD'
geofile_ref = f'{geoem_in}/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc'
file_ref = 'wrfout_d01_2020-08-01_00:00:00'


## MCS config

Variables = ["PR", "Tb"]

# MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# precipitation tracking options
SmoothSigmaP = 0  # Gaussion std for precipitation smoothing
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

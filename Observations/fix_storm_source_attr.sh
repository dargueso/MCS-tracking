#!/bin/bash
# The tracker stamps every Storms_*.nc with source="WRF Model outputs (...)",
# also on the observational trackings. Set it right; the data are untouched.
export PATH=/home/dargueso/anaconda3/envs/wrfprocessing/bin:$PATH
T=/scratch3/dargueso/obs-mcs-tracking/tracking
for f in $T/obs/exp1/Storms_*.nc; do ncatted -h -a source,global,o,c,"IMERG (GPM_3IMERGHH.07) rain and MERGIR (GPM_MERGIR.1) Tb, hourly on the 0.1 deg IMERG grid" $f; done
for f in $T/rad/exp1/Storms_*.nc; do ncatted -h -a source,global,o,c,"EURADCLIM rain (quality-masked radar coverage, block-averaged to 0.1 deg) and MERGIR Tb, hourly" $f; done
echo "done: $(ls $T/obs/exp1/Storms_*.nc | wc -l) obs, $(ls $T/rad/exp1/Storms_*.nc | wc -l) rad files"

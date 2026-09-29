#!/bin/bash
# Full YS tracking at 2 km, both runs. Output (pickles + Storms netCDF) goes to
# /scratch3, mirroring the /scratch1 layout.
set -u
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
CFG=/me01/dargueso/Scripts/MCS-tracking/mcs_config.py
cd /me01/dargueso/Scripts/MCS-tracking

for RUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
  echo "[$(date +%H:%M:%S)] ===== $RUN ====="
  sed -i "s|^path_in = .*|path_in = \"/scratch3/dargueso/postprocessed/EPICC/$RUN\"|" $CFG
  sed -i "s|^wrun = .*|wrun = \"$RUN\"|" $CFG
  $PY -c "import sys;sys.path.insert(0,'.');import mcs_config as c;print('  ->',c.wrun,c.bt_method,'->',c.path_out)"
  $PY MCS_tracking_WRF.py
  D=/scratch3/dargueso/postprocessed/EPICC/$RUN/ConvStormTracking_YS/exp1
  echo "[$(date +%H:%M:%S)] $RUN: $(ls $D/MCS_* 2>/dev/null | wc -l)/120 months, $(du -sh $D 2>/dev/null | cut -f1)"
  df -h /scratch3 | tail -1
done
echo "[$(date +%H:%M:%S)] ALL DONE"

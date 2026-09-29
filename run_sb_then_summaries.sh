#!/bin/bash
# 1. Re-run SB tracking at 2 km with the current code, so SB and YS differ ONLY
#    in the brightness-temperature conversion (the old ConvStormTracking/ output
#    predates the area-weighted rain volume).
# 2. Build ASON 2011-2020 summaries and the main figure for both methods.
set -u
PYT=/home/dargueso/anaconda3/envs/MCStracking/bin/python   # tracking: no wrf needed
PYP=/home/dargueso/anaconda3/envs/py310/bin/python          # plotting: needs wrf + dask
ROOT=/me01/dargueso/Scripts/MCS-tracking
CFG=$ROOT/mcs_config.py
PLOT=$ROOT/Plotting/plot_scatter_hist_storm_characteristics.py

cd $ROOT
sed -i 's|^bt_method = .*|bt_method = "SB"|' $CFG
for RUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
  echo "[$(date +%H:%M:%S)] ===== SB tracking: $RUN ====="
  sed -i "s|^path_in = .*|path_in = \"/scratch3/dargueso/postprocessed/EPICC/$RUN\"|" $CFG
  sed -i "s|^wrun = .*|wrun = \"$RUN\"|" $CFG
  $PYT -c "import sys;sys.path.insert(0,'.');import mcs_config as c;print('  ->',c.wrun,c.bt_method,'->',c.path_out)"
  $PYT MCS_tracking_WRF.py
  D=/scratch3/dargueso/postprocessed/EPICC/$RUN/ConvStormTracking_SB/exp1
  echo "[$(date +%H:%M:%S)] $RUN: $(ls $D/MCS_* 2>/dev/null | wc -l)/120 months"
done

cd $ROOT/Plotting
sed -i 's/^calc_summary=.*/calc_summary=True/' $PLOT
for BT in SB YS; do
  echo "[$(date +%H:%M:%S)] ===== summaries + figure: $BT ====="
  sed -i "s/^bt_method = .*/bt_method = '$BT'/" $PLOT
  $PYP plot_scatter_hist_storm_characteristics.py 2>&1 | grep -vi "userwarning\|warnings.warn\|pkg_resources\|getfattr" | tail -3
  ls -la --time-style='+%H:%M' storms_*_${BT}_m3.pkl 2>/dev/null | awk '{print "   ", $5, $NF}'
done
sed -i 's/^calc_summary=.*/calc_summary=False/' $PLOT
echo "[$(date +%H:%M:%S)] ALL DONE"

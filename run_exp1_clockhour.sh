#!/bin/bash
# Re-run the reference tracking (exp1 = ST1) on the clock-hour rain, both
# brightness-temperature conversions and both climates, so that every result
# comes from the same code and the same, corrected input:
#     YS present, YS PGW, SB present, SB PGW      (~75 min each)
# then the ASON summaries for both conversions and the SB/YS robustness check.
#
# Sequential on purpose: each run is configured by rewriting mcs_config.py
# with sed, so two runs at once would overwrite each other's settings.
# Before each run a guard checks that the thresholds are exp1 and that the
# rain it is about to read is the clock-hour version; a failed guard skips
# that run rather than producing mislabelled output.
#
#     nohup ./run_exp1_clockhour.sh > exp1_clockhour.out 2>&1 &
set -u
PYT=/home/dargueso/anaconda3/envs/MCStracking/bin/python   # tracking
PYP=/home/dargueso/anaconda3/envs/py310/bin/python          # plotting: needs wrf + dask
ROOT=/me01/dargueso/Scripts/MCS-tracking
CFG=$ROOT/mcs_config.py
PLOT=$ROOT/Plotting/plot_scatter_hist_storm_characteristics.py
POST=/scratch3/dargueso/postprocessed/EPICC

cd $ROOT
for BT in YS SB; do
  sed -i "s|^bt_method = .*|bt_method = \"$BT\"|" $CFG
  for RUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
    echo "[$(date +%H:%M:%S)] ===== $BT tracking: $RUN ====="
    sed -i "s|^path_in = .*|path_in = \"$POST/$RUN\"|" $CFG
    sed -i "s|^wrun = .*|wrun = \"$RUN\"|" $CFG
    $PYT - <<'EOF' || { echo "   GUARD FAILED: skipping this run"; continue; }
import sys, glob
sys.path.insert(0, '.')
import xarray as xr
import mcs_config as c
exp1 = dict(thres_pr=5, min_area_pr=500, min_area_bt=1000, MCS_thres_peak_pr=15)
got = {k: getattr(c, k) for k in exp1}
assert got == exp1 and c.exp_label == "exp1", f"not exp1: {got}, label {c.exp_label}"
files = sorted(glob.glob(f"{c.path_in}/RAIN/UIB_01H_RAIN_20??-??.nc"))
assert len(files) == 120, f"{len(files)} rain files in {c.path_in}/RAIN"
assert "correction" in xr.open_dataset(files[0]).attrs, "rain is not the clock-hour version"
print("   ->", c.wrun, c.bt_method, c.exp_label, "| rain", c.path_in + "/RAIN (clock hour) ->", c.path_out)
EOF
    $PYT MCS_tracking_WRF.py > $ROOT/exp1_clockhour_${BT}_${RUN}.log 2>&1
    D=$POST/$RUN/ConvStormTracking_$BT/exp1
    echo "[$(date +%H:%M:%S)] $RUN $BT: $(ls $D/MCS_* 2>/dev/null | wc -l) MCS pickles, $(ls $D/*.nc 2>/dev/null | wc -l)/120 netCDF"
  done
done
# leave the production default: YS, present day
sed -i "s|^bt_method = .*|bt_method = \"YS\"|" $CFG
sed -i "s|^path_in = .*|path_in = \"$POST/EPICC_2km_ERA5\"|" $CFG
sed -i "s|^wrun = .*|wrun = \"EPICC_2km_ERA5\"|" $CFG

cd $ROOT/Plotting
sed -i 's/^calc_summary=.*/calc_summary=True/' $PLOT
for BT in SB YS; do
  echo "[$(date +%H:%M:%S)] ===== summaries + figure: $BT ====="
  sed -i "s/^bt_method = .*/bt_method = '$BT'/" $PLOT
  $PYP plot_scatter_hist_storm_characteristics.py 2>&1 | grep -vi "userwarning\|warnings.warn\|pkg_resources\|getfattr" | tail -3
  ls -la --time-style='+%H:%M' storms_*_${BT}_m3.pkl 2>/dev/null | awk '{print "   ", $5, $NF}'
done
sed -i 's/^calc_summary=.*/calc_summary=False/' $PLOT

echo "[$(date +%H:%M:%S)] ===== SB/YS robustness check (ASON) ====="
$PYT check_bt_robustness.py 2>&1 | grep -vi warn | tail -30
echo "[$(date +%H:%M:%S)] ALL DONE"

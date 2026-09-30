#!/bin/bash
# Threshold configurations beyond the reference (exp1 = ST1), all on the
# clock-hour rain and the Yang & Slingo Tb:
#   part 1: the 0.1 deg pair, IMERG + MERGIR and the coarsened present-day
#           model, exp2..exp5 (tracking/<dataset>/<exp>)
#   part 2: the native 2 km present and PGW runs, exp2 and exp3 (ST2, ST3 of
#           the manuscript), into <run>/ConvStormTracking_YS/<exp>
#
# Strictly sequential: both trackers read the thresholds from mcs_config.py
# inside their worker processes, so the file is rewritten before each run and
# must not change while one is going. A guard checks, before every run, that
# mcs_config holds exactly the intended set and that the 2 km rain is the
# clock-hour version. exp1 is restored at the end.
#
#     nohup ./run_st2_st5.sh > st2_st5.out 2>&1 &
set -u
PYT=/home/dargueso/anaconda3/envs/MCStracking/bin/python
ROOT=/me01/dargueso/Scripts/MCS-tracking
CFG=$ROOT/mcs_config.py
POST=/scratch3/dargueso/postprocessed/EPICC
OBS=/scratch3/dargueso/obs-mcs-tracking/tracking
cd $ROOT

# exp: thres_pr min_area_pr min_area_bt MCS_thres_pr MCS_thres_peak_pr
declare -A SET=( [exp1]="5 500 1000 5 15" [exp2]="5 1000 2000 5 15" [exp3]="5 1000 10000 5 10"
                 [exp4]="15 500 1000 15 30" [exp5]="15 1000 2000 15 30" )

set_exp() {   # rewrite the threshold lines and the label
  read -r a b c d e <<< "${SET[$1]}"
  sed -i "s|^thres_pr = .*|thres_pr = $a  # precipitation threshold [mm/h]|" $CFG
  sed -i "s|^min_area_pr = .*|min_area_pr = $b  # minimum area of precipitation feature in km2|" $CFG
  sed -i "s|^min_area_bt = .*|min_area_bt = $c  # minimum area of cloud shield in km2|" $CFG
  sed -i "s|^MCS_thres_pr = .*|MCS_thres_pr = $d  # minimum max precipitation in mm/h|" $CFG
  sed -i "s|^MCS_thres_peak_pr = .*|MCS_thres_peak_pr = $e  # Minimum lifetime peak of MCS precipitation|" $CFG
  sed -i "s|^exp_label = .*|exp_label = \"$1\"|" $CFG
  sed -i "s|^bt_method = .*|bt_method = \"YS\"|" $CFG
}

guard() {   # $1 exp, $2 run or "-" for the 0.1 deg pair
  EXP=$1 RUN=$2 $PYT - <<'EOF'
import os, sys, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'Observations')
import mcs_config as c
exp, run = os.environ["EXP"], os.environ["RUN"]
sets = {"exp1": (5, 500, 1000, 5, 15), "exp2": (5, 1000, 2000, 5, 15), "exp3": (5, 1000, 10000, 5, 10),
        "exp4": (15, 500, 1000, 15, 30), "exp5": (15, 1000, 2000, 15, 30)}
got = (c.thres_pr, c.min_area_pr, c.min_area_bt, c.MCS_thres_pr, c.MCS_thres_peak_pr)
assert got == sets[exp] and c.exp_label == exp and c.bt_method == "YS", f"config is {got} {c.exp_label} {c.bt_method}, wanted {exp}"
assert c.MCS_min_area == c.min_area_pr and c.MCS_min_area_bt == c.min_area_bt
if run != "-":
    import xarray as xr
    assert c.wrun == run and c.path_in.endswith(run), f"run {c.wrun} {c.path_in}"
    files = sorted(glob.glob(f"{c.path_in}/RAIN/UIB_01H_RAIN_20??-??.nc"))
    assert len(files) == 120 and "correction" in xr.open_dataset(files[0]).attrs, "rain is not the 120 clock-hour files"
    print(f"   -> {c.wrun} {c.bt_method} {c.exp_label} thresholds {got} -> {c.path_out}")
else:
    import track_storms as t
    assert t.current_exp() == exp, f"track_storms sees {t.current_exp()}"
    print(f"   -> 0.1 deg pair, {exp}, thresholds {got}")
EOF
}

echo "[$(date +%H:%M:%S)] ===== part 1: 0.1 deg pair, exp2..exp5 ====="
for E in exp2 exp3 exp4 exp5; do
  set_exp $E
  guard $E - || { echo "   GUARD FAILED for $E: skipping"; continue; }
  echo "[$(date +%H:%M:%S)] tracking obs + mod0.1_YS_pres, $E"
  (cd Observations && $PYT track_storms.py obs mod0.1_YS_pres > $ROOT/st_${E}_0.1deg.log 2>&1)
  for D in obs mod0.1_YS_pres; do echo "   $D $E: $(ls $OBS/$D/$E/MCS_* 2>/dev/null | wc -l) months with storms, $(ls $OBS/$D/$E/Storms_*.nc 2>/dev/null | wc -l)/120 netCDF"; done
done

echo "[$(date +%H:%M:%S)] ===== part 2: 2 km present and PGW, exp2 and exp3 ====="
for E in exp2 exp3; do
  set_exp $E
  for RUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
    sed -i "s|^path_in = .*|path_in = \"$POST/$RUN\"|" $CFG
    sed -i "s|^wrun = .*|wrun = \"$RUN\"|" $CFG
    guard $E $RUN || { echo "   GUARD FAILED for $E $RUN: skipping"; continue; }
    echo "[$(date +%H:%M:%S)] tracking $RUN, $E"
    $PYT MCS_tracking_WRF.py > $ROOT/st_${E}_${RUN}.log 2>&1
    D=$POST/$RUN/ConvStormTracking_YS/$E
    echo "[$(date +%H:%M:%S)] $RUN $E: $(ls $D/MCS_* 2>/dev/null | wc -l) MCS pickles, $(ls $D/*.nc 2>/dev/null | wc -l)/120 netCDF"
  done
done

# leave the production default: exp1, YS, present day
set_exp exp1
sed -i "s|^path_in = .*|path_in = \"$POST/EPICC_2km_ERA5\"|" $CFG
sed -i "s|^wrun = .*|wrun = \"EPICC_2km_ERA5\"|" $CFG
echo "[$(date +%H:%M:%S)] config restored to exp1 / YS / present; ALL DONE"

#!/bin/bash
# Sensitivity experiments exp6..exp10 (mcs_config_sens.py), all on the
# clock-hour rain with the Yang & Slingo Tb:
#   part 1: the 0.1 deg pair, IMERG + MERGIR and the coarsened present-day model
#   part 2: the native 2 km present and PGW runs, present and PGW together
#           (the settings come from the environment, so nothing is rewritten
#           and two runs can share the machine)
# mcs_config.py is never touched. A guard checks the loaded settings and the
# clock-hour rain before every run.
#
#     nohup ./run_sens.sh > sens.out 2>&1 &
set -u
PYT=/home/dargueso/anaconda3/envs/MCStracking/bin/python
ROOT=/me01/dargueso/Scripts/MCS-tracking
POST=/scratch3/dargueso/postprocessed/EPICC
OBS=/scratch3/dargueso/obs-mcs-tracking/tracking
export MCS_CONFIG=mcs_config_sens
cd $ROOT
EXPS=${@:-exp6 exp7 exp8 exp9 exp10}     # experiments to run, e.g. ./run_sens.sh exp11

guard() {   # $1 exp, $2 run or "-"
  SENS_EXP=$1 MCS_RUN=${2/-/EPICC_2km_ERA5} $PYT - <<'EOF'
import os, sys, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'Observations')
import mcs_config_sens as c
exp, run = os.environ["SENS_EXP"], os.environ["MCS_RUN"]
assert c.exp_label == exp and c.bt_method == "YS", (c.exp_label, c.bt_method)
want = {"exp4": dict(thres_pr=15, MCS_thres_pr=15, MCS_thres_peak_pr=30, min_area_pr=500, min_area_bt=1000),
        "exp5": dict(thres_pr=15, MCS_thres_pr=15, MCS_thres_peak_pr=30, min_area_pr=1000, min_area_bt=2000),
        "exp6": dict(require_bt=False), "exp7": dict(thres_pr=3, min_area_pr=250, min_time_pr=2, MCS_thres_pr=3, MCS_thres_peak_pr=10),
        "exp8": dict(min_time_pr=6, min_time_bt=8, MCS_min_time=8), "exp9": dict(smooth_sigma_pr=1, smooth_sigma_bt=1),
        "exp10": dict(min_overlap=0.3), "exp11": dict(require_bt=False, min_area_pr=1000, MCS_thres_peak_pr=30)}[exp]
for k, v in want.items():
    assert getattr(c, k) == v, (k, getattr(c, k), v)
s = f"pr {c.thres_pr}/{c.min_area_pr}/{c.min_time_pr}h bt {c.thres_bt}/{c.min_area_bt}/{c.min_time_bt}h mcs {c.MCS_thres_pr}/{c.MCS_thres_peak_pr}/{c.MCS_min_time}h sigma {c.smooth_sigma_pr} bt_req {c.require_bt} overlap {c.min_overlap}"
if os.environ.get("CHECK_RAIN", "0") == "1":
    import xarray as xr
    assert c.wrun == run and c.path_in.endswith(run), (c.wrun, c.path_in)
    files = sorted(glob.glob(f"{c.path_in}/RAIN/UIB_01H_RAIN_20??-??.nc"))
    assert len(files) == 120 and "correction" in xr.open_dataset(files[0]).attrs, "rain is not the 120 clock-hour files"
    print(f"   -> {run} {exp}: {s} -> {c.path_out}")
else:
    import track_storms as t
    assert t.current_exp() == exp, t.current_exp()
    print(f"   -> 0.1 deg pair {exp}: {s}")
EOF
}

echo "[$(date +%H:%M:%S)] ===== part 1: 0.1 deg pair, exp6..exp10 ====="
for E in $EXPS; do
  guard $E - || { echo "   GUARD FAILED for $E: skipping"; continue; }
  echo "[$(date +%H:%M:%S)] tracking obs + mod0.1_YS_pres, $E"
  (cd Observations && SENS_EXP=$E $PYT track_storms.py obs mod0.1_YS_pres > $ROOT/sens_${E}_0.1deg.log 2>&1)
  for D in obs mod0.1_YS_pres; do echo "   $D $E: $(ls $OBS/$D/$E/MCS_* 2>/dev/null | wc -l) months with storms, $(ls $OBS/$D/$E/Storms_*.nc 2>/dev/null | wc -l)/120 netCDF"; done
done

echo "[$(date +%H:%M:%S)] ===== part 2: 2 km present and PGW, exp6..exp10 ====="
for E in $EXPS; do
  for RUN in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
    CHECK_RAIN=1 guard $E $RUN || { echo "   GUARD FAILED for $E $RUN: skipping"; continue; }
    echo "[$(date +%H:%M:%S)] tracking $RUN, $E"
    ( SENS_EXP=$E MCS_RUN=$RUN $PYT MCS_tracking_WRF.py > $ROOT/sens_${E}_${RUN}.log 2>&1
      D=$POST/$RUN/ConvStormTracking_YS/$E
      echo "[$(date +%H:%M:%S)] $RUN $E: $(ls $D/MCS_* 2>/dev/null | wc -l) MCS pickles, $(ls $D/*.nc 2>/dev/null | wc -l)/120 netCDF" ) &
  done
  wait
done
echo "[$(date +%H:%M:%S)] ALL DONE"

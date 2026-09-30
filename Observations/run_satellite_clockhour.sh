#!/bin/bash
# Satellite (IMERG + MERGIR) comparison on the clock-hour model rain:
#   1. coarsen the 2 km model (RAIN, OLR -> TB) onto the 0.1 deg grid, both runs
#   2. re-track the coarsened model, both BT conversions, both climates
#   3. re-make the comparison figures
# The observations and their tracking (tracking/obs) do not change. The previous
# model inputs and tracking were renamed *_origrain; the previous figures are in
# figures/before_clockhour_fix/satellite/.
#
#     nohup ./run_satellite_clockhour.sh > satellite_clockhour.out 2>&1 &
set -u
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
cd /me01/dargueso/Scripts/MCS-tracking/Observations
M=/scratch3/dargueso/obs-mcs-tracking/MODEL_0.1deg

echo "[$(date +%H:%M:%S)] ===== coarsening, present and PGW in parallel ====="
$PY make_model_tracking_input.py pres > satellite_clockhour_coarsen_pres.log 2>&1 &
$PY make_model_tracking_input.py fut  > satellite_clockhour_coarsen_fut.log 2>&1 &
wait
for R in EPICC_2km_ERA5 EPICC_2km_ERA5_CMIP6anom; do
  n=$(ls $M/$R/MOD_01H_RAIN_*.nc 2>/dev/null | wc -l)
  echo "   $R: $n/120 rain files, $(ls $M/$R/MOD_01H_TB_*.nc 2>/dev/null | wc -l)/240 TB files"
  [ "$n" -eq 120 ] || { echo "   INCOMPLETE: stopping"; exit 1; }
done
$PY - <<'EOF' || { echo "   rain provenance check FAILED: stopping"; exit 1; }
import xarray as xr
d = xr.open_dataset("/scratch3/dargueso/obs-mcs-tracking/MODEL_0.1deg/EPICC_2km_ERA5/MOD_01H_RAIN_2014-09.nc")
w = d.RAIN.attrs.get("rain_hour_window", "")
print("   coarsened rain:", w)
assert w.startswith("clock hour"), "coarsened rain is not from the clock-hour files"
EOF

echo "[$(date +%H:%M:%S)] ===== tracking the coarsened model ====="
$PY track_storms.py mod0.1_YS_pres mod0.1_SB_pres mod0.1_YS_fut mod0.1_SB_fut > satellite_clockhour_tracking.log 2>&1
for d in mod0.1_YS_pres mod0.1_SB_pres mod0.1_YS_fut mod0.1_SB_fut; do
  echo "   $d: $(ls /scratch3/dargueso/obs-mcs-tracking/tracking/$d/exp1/MCS_* 2>/dev/null | wc -l) MCS pickles"
done

echo "[$(date +%H:%M:%S)] ===== figures ====="
for P in plot_obs_model_comparison.py plot_obs_model_maps.py plot_obs_model_maps_relative.py plot_storm_structure.py; do
  echo "   $P"; $PY $P > satellite_clockhour_${P%.py}.log 2>&1 || echo "   FAILED: $P"
done
echo "[$(date +%H:%M:%S)] ALL DONE"

#!/bin/bash
# Waits for the present-day clock-hour correction, validates it, then redoes the
# model side of the evaluation on the corrected files. Previous outputs are kept.
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
LOG=/scratch3/dargueso/postprocessed/EPICC/fix_hourly_rain_clockhour.log
cd /me01/dargueso/Scripts/MCS-tracking/Observations
echo "$(date) waiting for EPICC_2km_ERA5 clock-hour files"
until grep -q "EPICC_2km_ERA5 done" $LOG; do
  if grep -q Traceback $LOG; then echo "$(date) CORRECTION FAILED, stopping"; exit 1; fi
  sleep 60
done
echo "$(date) correction done, validating"
$PY validate_clockhour.py || { echo "$(date) VALIDATION FAILED, stopping; nothing re-run"; exit 1; }
set -e
# One-off, already run on 2026-09-29. The figures have since moved to
# $S/figures/{radar,stations}, and the old ones to $S/figures/before_clockhour_fix/.
S=/scratch3/dargueso/obs-mcs-tracking
# keep what was made from the original (10-min-early) model files
mv $S/EURADCLIM/stats $S/EURADCLIM/stats_before_clockhour_fix
mv $S/STATIONS/evaluation/AT_STATIONS_EPICC_2km_ERA5_01H_2011-2020.nc \
   $S/STATIONS/evaluation/AT_STATIONS_EPICC_2km_ERA5_01H_2011-2020_before_clockhour_fix.nc
mkdir -p $S/figures_before_clockhour_fix
cp -p $S/EURADCLIM/*.png $S/EURADCLIM/*.csv $S/EURADCLIM/*.txt $S/figures_before_clockhour_fix/ 2>/dev/null || true
cp -p $S/STATIONS/evaluation/station_model_* $S/figures_before_clockhour_fix/ 2>/dev/null || true
echo "$(date) model at stations"; $PY extract_at_stations.py
echo "$(date) radar statistics";  $PY radar_model_stats.py
for SEAS in ASON ANN; do
  echo "$(date) figures $SEAS"
  $PY plot_radar_model_qq.py --season $SEAS
  $PY plot_radar_model_maps.py --season $SEAS --scale 1
  $PY plot_radar_model_maps.py --season $SEAS --scale 5
  $PY plot_station_model.py --season $SEAS
done
echo "$(date) ALL DONE ON CLOCK-HOUR MODEL RAIN"

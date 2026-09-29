#!/bin/bash
# Mask with the radial test, then every evaluation figure, ASON and whole year.
set -e
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
cd /me01/dargueso/Scripts/MCS-tracking/Observations
$PY make_radar_mask.py
for S in ASON ANN; do
  $PY plot_radar_model_qq.py --season $S
  $PY plot_radar_model_maps.py --season $S --scale 1
  $PY plot_radar_model_maps.py --season $S --scale 5
  $PY plot_station_model.py --season $S
done
echo ALL FIGURES DONE

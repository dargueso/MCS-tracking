#!/bin/bash
# EURADCLIM chain: regrid -> statistics -> quality mask -> radar at the gauges.
set -e
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
cd /me01/dargueso/Scripts/MCS-tracking/Observations
$PY make_radar_input.py
$PY radar_model_stats.py
$PY make_radar_mask.py
$PY extract_at_stations.py --radar
echo CHAIN DONE

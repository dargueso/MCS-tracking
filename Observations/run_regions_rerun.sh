#!/bin/bash
# Re-run every evaluation plot with the polygon regions (CAT, VAL, BAL, MUR, AND).
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
cd /me01/dargueso/Scripts/MCS-tracking/Observations
run() { echo "[$(date +%H:%M:%S)] start $*"; $PY "$@" > "regions_rerun_$(echo $* | tr ' /.' '___').log" 2>&1; echo "[$(date +%H:%M:%S)] exit $? $*"; }
run plot_station_model.py --season ASON &
run plot_station_model.py --season ANN &
run plot_arnau_model.py --season ASON &
run plot_arnau_model.py --season ANN &
run plot_storm_rain_stations.py --season ASON &
run plot_storm_rain_stations.py --season ANN &
run plot_radar_model_qq.py --season ASON &
run plot_radar_model_qq.py --season ANN &
wait
run plot_radar_model_maps.py --season ASON --scale 1 &
run plot_radar_model_maps.py --season ASON --scale 5 &
run plot_radar_model_maps.py --season ANN --scale 1 &
run plot_radar_model_maps.py --season ANN --scale 5 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"

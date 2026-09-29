#!/bin/bash
# Wait for the two coarsening jobs, then track the model datasets and plot.
set -u
PY=/home/dargueso/anaconda3/envs/MCStracking/bin/python
cd /me01/dargueso/Scripts/MCS-tracking/Observations

echo "[$(date +%H:%M:%S)] waiting for coarsening ($*)"
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] pid $pid finished"
done

echo "[$(date +%H:%M:%S)] tracking model datasets"
$PY track_storms.py mod0.1_YS_pres mod0.1_SB_pres mod0.1_YS_fut mod0.1_SB_fut

echo "[$(date +%H:%M:%S)] comparison figure"
$PY plot_obs_model_comparison.py
echo "[$(date +%H:%M:%S)] pipeline complete"

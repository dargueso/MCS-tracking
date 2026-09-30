#!/bin/bash
# Wait for the radar coarsening, then track the symmetric radar-based pair.
cd /me01/dargueso/Scripts/MCS-tracking/Observations
while pgrep -f "python make_radar_tracking_input.py" > /dev/null; do sleep 15; done
n=$(ls /scratch3/dargueso/obs-mcs-tracking/ConvStormTracking/RADCOV_01H_RAIN_*.nc | wc -l)
echo "[$(date +%H:%M:%S)] coarsening done: $n RADCOV files; tracking"
/home/dargueso/anaconda3/envs/MCStracking/bin/python track_storms.py rad mod0.1_YS_pres_radcov > track_storms_radcov.log 2>&1
echo "[$(date +%H:%M:%S)] tracking exit $?"
for d in rad mod0.1_YS_pres_radcov; do echo "  $d: $(ls /scratch3/dargueso/obs-mcs-tracking/tracking/$d/exp1/MCS_* 2>/dev/null | wc -l) months"; done

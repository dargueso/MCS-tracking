#!/bin/bash
# Download the test data (RAIN_test.nc, OLR_test.nc; one month of EPICC 2 km
# output, 150 MB) from the GitHub release assets into tests/data/.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/data"
gh release download "${1:-v2.0}" -R dargueso/MCS-tracking -p '*_test.nc' -D "$HERE/data" --clobber
ls -la "$HERE/data"

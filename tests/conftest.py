"""Test data location.

RAIN_test.nc and OLR_test.nc (one month of EPICC 2 km output, 150 MB together)
are not in the repository. They are assets of the GitHub release
(tests/fetch_test_data.sh downloads them into tests/data/) and are also kept at
/scratch3/dargueso/MCS-tracking_testdata on the medicane server. Point
MCSTRACKING_TEST_DATA at a directory holding both files to use another copy.
Tests that need them are skipped when they are absent.
"""
import os
import pytest

DATA = os.environ.get("MCSTRACKING_TEST_DATA",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
_have = all(os.path.exists(os.path.join(DATA, f)) for f in ("RAIN_test.nc", "OLR_test.nc"))
needs_data = pytest.mark.skipif(
    not _have,
    reason=f"RAIN_test.nc/OLR_test.nc not in {DATA}: run tests/fetch_test_data.sh "
           "or set MCSTRACKING_TEST_DATA")

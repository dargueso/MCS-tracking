# Tests

    pytest tests

`test_area_simple`, `test_area_obj` and `test_remove_small_short_objects` need no
data. `test_area` and `test_MCStracking` read `RAIN_test.nc` and `OLR_test.nc`
(one month of EPICC 2 km output, 150 MB together, not in the repository) from
`tests/data/`, or from the directory named by `MCSTRACKING_TEST_DATA`, and are
skipped when the files are absent.

    tests/fetch_test_data.sh            # downloads both from the GitHub release assets
    MCSTRACKING_TEST_DATA=/scratch3/dargueso/MCS-tracking_testdata pytest tests   # medicane

`test_MCStracking` runs the full tracker on that month and checks the number of
storms (17) and three storm statistics to full precision, so it is the
regression test for any change to `mcstracking/tracking.py`.

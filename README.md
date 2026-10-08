# MCS-tracking: mesoscale convective system and convective storm tracker

[![DOI](https://zenodo.org/badge/474651021.svg)](https://doi.org/10.5281/zenodo.15732413)

Python package `mcstracking`: identifies and tracks convective storms in gridded
hourly precipitation and brightness-temperature fields, and computes per-storm
statistics (footprint, rain volume, peak rain rate, lifetime, track). It is an
optimised version of the A. Prein MCS-tracking algorithm
([original notebook](https://colab.research.google.com/drive/1MrQFujQCFhesk0MCUSqB41Mx3AHEd1ua)),
adapted to ingest WRF postprocessed output or any other netCDF with rain and
outgoing longwave radiation (OLR) on a 2-D latitude/longitude grid.

Version 2.0 is the tracker used for the EPICC western-Mediterranean
convective-storm study (present-day and pseudo-global-warming WRF simulations
at 2 km, plus satellite observations). The evaluation, sensitivity experiments
and analysis scripts of that study live in a separate repository
(`MedConvStorms_paper`); this one holds the algorithm alone.

## Installation

Any of these works; the dependencies (numpy, scipy, pandas, xarray, netCDF4,
joblib; Python >= 3.8) are declared in `pyproject.toml`, so pip installs them.

```bash
# straight from GitHub, into any environment
pip install git+https://github.com/dargueso/MCS-tracking.git@v2.0

# a conda environment with current versions (conda-forge), then an editable install
conda env create -f environment.yml && conda activate mcstracking
pip install -e .

# the pinned Python 3.8 environment the EPICC storm study was produced with
conda env create -f MCStracking.yml && conda activate MCStracking
pip install -e . --no-build-isolation      # setuptools < 64 there; drop the flag elsewhere
```

The test suite gives the same results on both environments (numpy 1.22 to
2.5, pandas 1.4 to 3.0).

## How it works

1. **Precipitation objects**: cells with rain >= `thres_pr` are labelled as
   3-D (time, y, x) connected components; objects smaller than `min_area_pr`
   or shorter than `min_time_pr` are dropped.
2. **Cloud-shield objects**: the same on brightness temperature <= `thres_bt`
   with `min_area_bt` and `min_time_bt`.
3. **Storms (MCS)**: a precipitation object qualifies when its lifetime peak
   rain rate reaches `MCS_thres_peak_pr`, its maximum reaches `MCS_thres_pr`,
   it is at least `MCS_min_area` and `MCS_min_time`, and, unless
   `require_bt = False`, it is overlain by a cloud shield of at least
   `MCS_min_area_bt` with a cold core <= `MCS_thres_bt`.
4. Objects that touch across the date line are reconnected and objects that
   merge and split are broken up along their tracks.

With `min_overlap > 0`, objects are 2-D components linked hour to hour only
where they overlap by that fraction (one successor and one predecessor each)
instead of the 3-D labelling that links anything touching.

`olr_to_tb(olr, method)` converts OLR to brightness temperature: `"YS"`
(Ohring et al. 1984 as given in Yang & Slingo 2001, a window-channel Tb, the
reference) or `"SB"` (the grey-body inversion `(OLR/sigma)^0.25`, ~22 K
colder). The 241 K / 225 K thresholds come from the satellite literature and
are defined on window Tb, so use `"YS"` with them.

## Configuration

All thresholds and options live in a plain Python module; the reference values
(exp1 of the EPICC study) are in `mcstracking/default_config.py` and
`mcs_config.py` at the repository root is a documented example that starts
from them. Which module is used is decided by `mcstracking.load_config()`:

| `MCS_CONFIG` | configuration used |
|---|---|
| unset | `mcs_config` on `sys.path` or in the working directory; else the packaged defaults |
| a module name (`my_config`) | that module, imported from `sys.path` |
| a path (`/path/to/my_config.py`) | that file; its directory is put on `sys.path`, so `from mcs_config import *` inside it works |

| setting | reference | meaning |
|---|---|---|
| `DT` | 1 | time step of the input [h] |
| `smooth_sigma_pr`, `smooth_sigma_bt` | 0 | Gaussian smoothing of rain and Tb [cells] |
| `thres_pr`, `min_area_pr`, `min_time_pr` | 5 mm/h, 500 km2, 3 h | precipitation objects |
| `thres_bt`, `min_area_bt`, `min_time_bt` | 241 K, 1000 km2, 5 h | cloud-shield objects |
| `MCS_thres_pr`, `MCS_thres_peak_pr` | 5, 15 mm/h | storm maximum and lifetime-peak rain rate |
| `MCS_min_area`, `MCS_min_area_bt` | = `min_area_pr`, `min_area_bt` | storm rain and shield areas |
| `MCS_thres_bt`, `MCS_min_time` | 225 K, 5 h | cold core and storm lifetime |
| `require_bt`, `min_overlap` | True, 0.0 | the two definition options above |
| `bt_method`, `exp_label`, `write_nc` | "YS", "exp1", True | provenance written to the output, and whether to write the netCDF |
| `path_in`, `wrun`, `path_out` | | input root, run name and output directory for the WRF driver |

Name a different threshold set with a different `exp_label`: the label and
every threshold are written into the output netCDF attributes, so files from
different configurations can always be told apart.

## Running

### WRF driver

Input layout: `{path_in}/RAIN/UIB_01H_RAIN_YYYY-MM.nc` (variable `RAIN`,
mm/h) and `{path_in}/OLR/UIB_01H_OLR_YYYY-MM.nc` (variable `OLR`, W m-2), one
month per file, hourly, 2-D `lat` and `lon`. The driver tracks every month
found, in parallel, and writes into `path_out`.

```bash
MCS_CONFIG=/path/to/my_config.py mcstracking-wrf       # or: python -m mcstracking.wrf_driver
```

Run-time overrides (environment variables): `MCS_MONTHS=2014-09,2014-10`
tracks only those months; `MCS_NJOBS` months in parallel (default 10);
`MCS_PATH_OUT` replaces `path_out`; `MCS_WRITE_NC=0` skips the netCDF.

### Python

```python
import pandas as pd, xarray as xr
from mcstracking import MCStracking, olr_to_tb, default_config

pr = xr.open_dataset("RAIN.nc"); olr = xr.open_dataset("OLR.nc")
times = pd.DatetimeIndex(pr.time.values)
storms, mask = MCStracking(pr.RAIN.values, olr_to_tb(olr.OLR.values, "YS"), times,
                           pr.lon.values, pr.lat.values,
                           nc_file="Storms.nc", path_out="tracked", cfg=default_config)
```

`cfg` is any module-like object with the settings above (`cfg=None` resolves
it as described). Rain and Tb are `(time, y, x)` arrays; `times` a
`DatetimeIndex`.

## Output

- `PR_YYYYMM`, `BT_YYYYMM`, `MCS_YYYYMM`: pickled dicts (`pandas.read_pickle`),
  one entry per precipitation object, cloud object and storm. Each entry holds
  per-time-step arrays over the object's life: `times`, `size` [km2], `tot`
  (rain summed over the footprint), `volume` (area-weighted rain volume),
  `max`, `mean`, `min` (rain rate over the footprint), `track` and
  `mass_center_loc` (centre positions) and `speed`. Written to `path_out`; a
  month with no qualifying storm writes no `MCS_` file.
- `UIB_01H_Storms_YYYY-MM.nc` (if `write_nc`): `PR`, `BT`, and the labelled
  masks `PR_objects`, `BT_objects`, `MCS_objects` on the full grid (~2 GB per
  month at 2 km), with every threshold, `bt_method`, `exp_label`, `require_bt`,
  `min_overlap` and the rain file provenance as global attributes.

## Tests

```bash
tests/fetch_test_data.sh      # 150 MB of test data from the GitHub release assets
pytest tests
```

See `tests/README.md`. The full-tracker test checks 17 storms and three
statistics to full precision and is the regression test for any change to the
numerics.

## Citing

See `CITATION.cff`. Version 2.0 is archived at https://doi.org/10.5281/zenodo.23242955;
the concept DOI above always resolves to the latest version. Licence: CC BY 4.0.

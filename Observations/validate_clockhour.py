#!/usr/bin/env python
"""Check the corrected clock-hour model rain against the exact RAINNC hourly
totals of the two raw wrfout days on disk, and against the original files
(corrected(HH) = original(HH) - v10(HH:00) + v10(HH+1:00)). Exit 1 if not."""
import sys
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc
RUN = "EPICC_2km_ERA5"
NEW = f"/scratch3/dargueso/postprocessed/EPICC/{RUN}/RAIN"          # clock-hour, definitive
OLD = f"/scratch1/dargueso/postprocessed/EPICC/{RUN}/RAIN"          # original, 10 min early
OUT = "/scratch1/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/out"
ok = True
for day in ("2020-01-10", "2020-08-01"):
    o = nc.Dataset(f"{OUT}/wrfout_d01_{day}_00:00:00")
    # WRF resets RAINNC every BUCKET_MM and counts resets in I_RAINNC: the
    # running total is RAINNC + BUCKET_MM * I_RAINNC (168 cells reset on 2020-08-01)
    bucket = float(getattr(o, "BUCKET_MM", 1000.0))
    total = (o.variables["RAINNC"][:, ::3, ::3].astype("f8")
             + bucket * o.variables["I_RAINNC"][:, ::3, ::3].astype("f8"))
    true = np.diff(total, axis=0)                                   # HH -> HH+1, 23 hours
    m = xr.open_dataset(f"{NEW}/UIB_01H_RAIN_{day[:7]}.nc")
    t = pd.to_datetime(m.time.values).floor("h")
    k = int(np.argmax(t >= pd.Timestamp(day)))
    got = m.RAIN.values[k:k + 23, ::3, ::3]
    err = float(np.nanmax(np.abs(got - true)))
    print(f"{day}: max |clock-hour - RAINNC| = {err:.5f} mm (wettest hour {true.max():.1f} mm)")
    ok &= err < 0.01
# relation to the original files, one month, both runs (no raw wrfout for PGW)
for run in (RUN, "EPICC_2km_ERA5_CMIP6anom"):
    new = xr.open_dataset(f"/scratch3/dargueso/postprocessed/EPICC/{run}/RAIN/"
                          "UIB_01H_RAIN_2014-09.nc").RAIN.values[:, ::5, ::5]
    old = xr.open_dataset(f"/scratch1/dargueso/postprocessed/EPICC/{run}/RAIN/"
                          "UIB_01H_RAIN_2014-09.nc").RAIN.values[:, ::5, ::5]
    r = np.corrcoef(new[1:-1].ravel(), old[1:-1].ravel())[0, 1]
    ratio = np.nansum(new) / np.nansum(old)
    print(f"{run} 2014-09: monthly total new/old {ratio:.4f}, hourly correlation new vs old {r:.4f}")
    ok &= 0.98 < ratio < 1.02 and r > 0.8
print("VALIDATION", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)

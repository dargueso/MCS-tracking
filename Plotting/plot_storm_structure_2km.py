#!/usr/bin/env python
"""Storm-relative rain structure at the native 2 km resolution: present vs PGW.

The 0.1 deg version (Observations/plot_storm_structure.py) asks whether the
model arranges rain inside a storm the way the satellite sees it. This asks the
climate question instead: given that PGW storms are larger AND more intense, is
the extra rain in a narrower, fiercer core or in a broader shield? The bulk
statistics cannot separate those - both raise area and volume.

Three things differ from the 0.1 deg script and are why this is a separate
file rather than a flag:

  * tracking lives at {path_postproc}/<wrun>/ConvStormTracking_<bt>/<exp>/,
    not under the observational tracking root;
  * the EPICC grid is Lambert conformal, so lat and lon both vary along both
    axes and the 0.1 deg trick of collapsing them to 1-D (lat[:,0], lon[0,:])
    is wrong here; nearest cells come from a KD-tree on unit-sphere coordinates,
    built once;
  * the composite half-width is set in kilometres, because 36 cells is ~400 km
    at 0.1 deg but only 72 km at 2 km.

Region handling follows the project convention (2026-09-30): only timesteps
inside the WME box contribute.

    python plot_storm_structure_2km.py --half-km 300
    python plot_storm_structure_2km.py --max-storms 50      # quick test

Writes storm_structure_2km_<exp>_<bt>.npz (composites + radial profiles) and
a companion _numbers.txt. Plotting is done by the caller from the npz.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as cfg

DX_KM = 2.0
PERIOD_WRUN = {"pres": "EPICC_2km_ERA5", "fut": "EPICC_2km_ERA5_CMIP6anom"}


def unit_sphere(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def grid_tree(frain):
    """KD-tree over the 2 km grid, and its shape. Built once, reused."""
    with xr.open_dataset(frain) as d:
        lat = np.asarray(d.lat.values)
        lon = np.asarray(d.lon.values)
    if lat.ndim == 1:                      # defensive: regular grid
        lon, lat = np.meshgrid(lon, lat)
    pts = unit_sphere(lat.ravel(), lon.ravel())
    return cKDTree(pts), lat.shape


def composite(period, exp, bt, half, months, max_storms=None):
    """Mean rain rate on the storm-relative grid for one climate."""
    wrun = PERIOD_WRUN[period]
    root = f"{cfg.path_postproc}/{wrun}/ConvStormTracking_{bt}/{exp}"
    lat0, lon0, lat1, lon1 = cfg.reg_coords["WME"]
    n = 2 * half + 1
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    tree = shape = None
    nstorm = nstep = 0

    for year in range(2011, 2021):
        for month in months:
            fin = f"{root}/MCS_{year}{month:02d}"
            frain = f"{cfg.path_postproc}/{wrun}/RAIN/UIB_01H_RAIN_{year}-{month:02d}.nc"
            if not (os.path.exists(fin) and os.path.exists(frain)):
                continue
            storms = pd.read_pickle(fin)
            if not storms:
                continue
            if tree is None:
                tree, shape = grid_tree(frain)
            with xr.open_dataset(frain) as dset:
                assert "RAIN" in dset, f"no RAIN variable in {frain}"
                times = pd.to_datetime(dset.time.values)
                tindex = {t: i for i, t in enumerate(times)}
                # Collect every (timestep, centre) this month needs, then read
                # only those slices: storms occupy a minority of the hours and a
                # month of 2 km rain is ~2.7 GB.
                want = []
                for storm in storms.values():
                    track = storm["track"]
                    inside = ((track[:, 0] > lat0) & (track[:, 0] < lat1)
                              & (track[:, 1] > lon0) & (track[:, 1] < lon1))
                    if not inside.any():
                        continue
                    nstorm += 1
                    if max_storms and nstorm > max_storms:
                        break
                    for k, when in enumerate(storm["times"]):
                        if not inside[k]:
                            continue
                        it = tindex.get(pd.Timestamp(when))
                        if it is not None:
                            want.append((it, track[k, 0], track[k, 1]))
                if not want:
                    continue
                its = np.array([w[0] for w in want])
                _, jj = tree.query(unit_sphere(np.array([w[1] for w in want]),
                                               np.array([w[2] for w in want])))
                jy, jx = np.unravel_index(jj, shape)
                order = np.argsort(its)
                rain_var = dset.RAIN
                cur_it, cur = -1, None
                for idx in order:
                    it = int(its[idx])
                    if it != cur_it:
                        cur = np.asarray(rain_var.isel(time=it).values)
                        cur_it = it
                    y0, y1 = int(jy[idx]) - half, int(jy[idx]) + half + 1
                    x0, x1 = int(jx[idx]) - half, int(jx[idx]) + half + 1
                    cy0, cx0 = max(0, -y0), max(0, -x0)
                    y0c, x0c = max(0, y0), max(0, x0)
                    y1c, x1c = min(cur.shape[0], y1), min(cur.shape[1], x1)
                    if y1c <= y0c or x1c <= x0c:
                        continue
                    patch = cur[y0c:y1c, x0c:x1c]
                    sl = (slice(cy0, cy0 + patch.shape[0]),
                          slice(cx0, cx0 + patch.shape[1]))
                    good = np.isfinite(patch)
                    total[sl] += np.where(good, patch, 0.0)
                    count[sl] += good
                    nstep += 1
            print(f"  {period} {year}-{month:02d}: {nstorm} storms, {nstep} steps",
                  flush=True)
            if max_storms and nstorm > max_storms:
                break
        if max_storms and nstorm > max_storms:
            break

    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean, nstorm, nstep


def radial_profile(field, half, step_km=10.0):
    off = np.arange(-half, half + 1) * DX_KM
    xx, yy = np.meshgrid(off, off)
    r = np.sqrt(xx ** 2 + yy ** 2)
    edges = np.arange(0, half * DX_KM, step_km)
    mids, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi) & np.isfinite(field)
        if sel.sum():
            mids.append(0.5 * (lo + hi))
            vals.append(float(field[sel].mean()))
    return np.array(mids), np.array(vals)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--exp", default="exp1")
    par.add_argument("--bt", default="YS")
    par.add_argument("--half-km", type=float, default=300.0)
    par.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    par.add_argument("--max-storms", type=int, default=None)
    a = par.parse_args()
    half = int(round(a.half_km / DX_KM))

    out = {}
    lines = [f"Storm-relative rain structure at 2 km, {a.exp}, {a.bt}, "
             f"months {a.months}, 2011-2020, WME (clipped)", ""]
    for period in ("pres", "fut"):
        field, nstorm, nstep = composite(period, a.exp, a.bt, half,
                                         a.months, a.max_storms)
        r, v = radial_profile(field, half)
        out[f"{period}_field"] = field
        out[f"{period}_r"] = r
        out[f"{period}_prof"] = v
        out[f"{period}_n"] = np.array([nstorm, nstep])
        lines.append(f"{period}: {nstorm} storms, {nstep} storm-hours, "
                     f"centre {field[half, half]:.2f} mm/h")

    # Core vs shield: is the extra rain concentrated or spread?
    rp, vp = out["pres_r"], out["pres_prof"]
    rf, vf = out["fut_r"], out["fut_prof"]
    lines += ["", "Radial mean rain rate [mm/h] and PGW/present ratio:", "",
              f"  {'r (km)':>8s} {'pres':>8s} {'fut':>8s} {'ratio':>7s}"]
    for i in range(0, len(rp), max(1, len(rp) // 20)):
        lines.append(f"  {rp[i]:8.0f} {vp[i]:8.3f} {vf[i]:8.3f} {vf[i]/vp[i]:7.2f}")

    def frac_within(r, v, rad):
        sel = r <= rad
        return float(np.trapz(v[sel] * r[sel], r[sel]) / np.trapz(v * r, r))
    lines += ["",
              f"Fraction of the radially integrated rain inside 50 km: "
              f"pres {frac_within(rp, vp, 50):.3f}, fut {frac_within(rf, vf, 50):.3f}",
              f"Fraction inside 100 km: "
              f"pres {frac_within(rp, vp, 100):.3f}, fut {frac_within(rf, vf, 100):.3f}",
              "",
              "A ratio that is flat with radius means the storm simply scales up;",
              "a ratio decreasing outward means core intensification; one that",
              "increases outward means the shield expanded."]

    tag = f"storm_structure_2km_{a.exp}_{a.bt}"
    np.savez(f"{tag}.npz", half=np.array([half]), dx_km=np.array([DX_KM]), **out)
    with open(f"{tag}_numbers.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

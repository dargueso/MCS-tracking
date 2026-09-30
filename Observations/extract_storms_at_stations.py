#!/usr/bin/env python
"""
Which tracked storm, if any, covers each gauge station in each hour, with
the rain and cold-cloud fields the tracker saw there.

Three trackings, all exp1 (MCS_objects: the storms that meet every criterion):

    obs      IMERG + MERGIR, 0.1 deg                     (tracking/obs)
    mod01    EPICC coarsened to 0.1 deg, same tracker     (tracking/mod0.1_YS_pres)
    mod2km   EPICC native 2 km tracking                   (<run>/ConvStormTracking_YS)

obs and mod01 are the like-for-like pair: the same grid, the same tracker,
the same criteria, so "a storm" means the same thing on both sides. mod2km
is the tracking used for the model storms in the manuscript, kept as a
sensitivity: how much the storm definition, not the model, changes the
answer.

Also kept, per tracking: the tracker's own rain at the station's cell (PR:
IMERG for obs, the coarsened / native model rain for the model) and the
cold-cloud object (BT_objects, Tb <= 241 K for >= 5 h). A storm mask is
defined by that tracker rain, so the rain it selects is wet by construction;
comparing PR under each side's own mask, and conditioning on cold cloud only,
separate that selection from real differences.

A station is inside a storm when the storm object covers its cell: the 0.1 deg
cell containing it for obs and mod01, its 2 km model cell (the one in
AT_STATIONS_<run>_01H) for mod2km. Time is the START of the clock hour:
the 0.1 deg files are stamped so already, the 2 km ones at HH:30 and are
floored.

Output, in {cfg.path_eval}:
    STORMS_AT_STATIONS_2011-2020.nc   per station and hour: MCS object id (0 = none),
                                      tracker rain (mm/h), cold-cloud object id

    python extract_storms_at_stations.py
"""

import os
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

import station_config as cfg
import obs_config as ocfg

NWORKERS = 12
TRACK = f"{ocfg.path_obs}/tracking"
SOURCES = {
    "obs": f"{TRACK}/obs/exp1/Storms_{{tag}}.nc",
    "mod01": f"{TRACK}/mod0.1_YS_pres/exp1/Storms_{{tag}}.nc",
    "mod2km": f"{ocfg.path_model}/{cfg.model_run}/ConvStormTracking_YS/exp1/UIB_01H_Storms_{{tag}}.nc",
}
MOD_01H = f"{cfg.path_eval}/AT_STATIONS_{cfg.model_run}_01H_{cfg.syear}-{cfg.eyear}.nc"
FOUT = f"{cfg.path_eval}/STORMS_AT_STATIONS_{cfg.syear}-{cfg.eyear}.nc"
_S = {}


def grid_cells(fin, lat, lon):
    """(j, i) of the 0.1 deg cell containing each station (nearest centre)."""
    with xr.open_dataset(fin) as d:
        glat, glon = d.lat.values, d.lon.values
    j = np.empty(lat.size, int); i = np.empty(lat.size, int)
    for n, (la, lo) in enumerate(zip(lat, lon)):
        k = np.nanargmin((glat - la) ** 2 + ((glon - lo) * np.cos(np.radians(la))) ** 2)
        j[n], i[n] = np.unravel_index(k, glat.shape)
    dist = np.hypot(glat[j, i] - lat, (glon[j, i] - lon) * np.cos(np.radians(lat))) * 111.2
    return j, i, dist


def do_month(job):
    src, tag = job
    j, i = _S[src]
    fin = SOURCES[src].format(tag=tag)
    if not os.path.exists(fin):
        return src, tag, None
    with nc.Dataset(fin) as f:
        t = f.variables["time"]
        times = pd.to_datetime([str(x) for x in nc.num2date(t[:], t.units, getattr(t, "calendar", "standard"))])
        # one slab over the stations' bounding box (index lists are read point by point)
        j0, j1, i0, i1 = j.min(), j.max() + 1, i.min(), i.max() + 1
        vals = []
        for name, dtype in (("MCS_objects", "int32"), ("PR", "float32"), ("BT_objects", "int32")):
            v = f.variables[name]
            v.set_auto_mask(False)
            vals.append(v[:, j0:j1, i0:i1][:, j - j0, i - i0].astype(dtype))   # (nt, nst)
    return src, tag, (times.floor("h"), *vals)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")
    with xr.open_dataset(cfg.file_01h) as h:
        codes = [str(c) for c in h.code.values]
        lat, lon = h.lat.values, h.lon.values
    with xr.open_dataset(MOD_01H) as m:
        mc = [str(c) for c in m.code.values]
        idx = [mc.index(c) for c in codes]
        mj, mi = m.model_j.values[idx], m.model_i.values[idx]
    tags = pd.date_range(f"{cfg.syear}-01", f"{cfg.eyear}-12", freq="MS").strftime("%Y-%m").tolist()
    first = SOURCES["obs"].format(tag=tags[0])
    oj, oi, odist = grid_cells(first, lat, lon)
    j2, i2, _ = grid_cells(SOURCES["mod01"].format(tag=tags[0]), lat, lon)
    assert np.array_equal(oj, j2) and np.array_equal(oi, i2), "obs and mod01 grids differ"
    _S.update(obs=(oj, oi), mod01=(oj, oi), mod2km=(mj, mi))
    logging.info("%d stations; 0.1 deg cell centres within %.1f km", len(codes), np.nanmax(odist))

    hours = pd.date_range(f"{cfg.syear}-01-01", f"{cfg.eyear}-12-31 23:00", freq="h")
    out = {s: np.zeros((len(codes), hours.size), "int32") for s in SOURCES}
    pr = {s: np.full((len(codes), hours.size), np.nan, "float32") for s in SOURCES}
    bt = {s: np.zeros((len(codes), hours.size), "int32") for s in SOURCES}
    have = {s: np.zeros(hours.size, bool) for s in SOURCES}
    jobs = [(s, t) for s in SOURCES for t in tags]
    with Pool(NWORKERS) as pool:
        for n, (src, tag, res) in enumerate(pool.imap_unordered(do_month, jobs), 1):
            if res is None:
                logging.warning("%s %s: no file", src, tag)
                continue
            times, ids, prv, btv = res
            pos = hours.get_indexer(times)
            ok = pos >= 0
            out[src][:, pos[ok]] = ids[ok].T
            pr[src][:, pos[ok]] = prv[ok].T
            bt[src][:, pos[ok]] = btv[ok].T
            have[src][pos[ok]] = True
            if n % 60 == 0:
                logging.info("%d/%d months", n, len(jobs))
    for s in SOURCES:
        logging.info("%s: %d of %d hours covered; station-hours in a storm %.2f%%", s,
                     have[s].sum(), hours.size, 100 * (out[s] > 0).mean())

    ds = xr.Dataset(
        {s: (("station", "time"), out[s], {"long_name": f"MCS object id covering the station, {s}",
                                           "source": SOURCES[s].format(tag="YYYY-MM"), "none": 0})
         for s in SOURCES},
        coords={"time": hours, "code": ("station", np.array(codes, dtype=object)),
                "lat": ("station", lat.astype("float32")), "lon": ("station", lon.astype("float32")),
                "grid01_j": ("station", oj), "grid01_i": ("station", oi),
                "model_j": ("station", mj), "model_i": ("station", mi)},
        attrs={"title": "Tracked storms (MCS_objects, exp1) at the AEMET 10-min stations",
               "time_convention": "START of the clock hour, UTC",
               "history": "extract_storms_at_stations.py (MCS-tracking/Observations)"})
    for s in SOURCES:
        ds[f"{s}_pr"] = (("station", "time"), pr[s], {"units": "mm/h", "long_name": f"tracker rain at "
                                                      f"the station's cell, {s}"})
        ds[f"{s}_btobj"] = (("station", "time"), bt[s], {"long_name": f"cold-cloud object id, {s}",
                                                         "none": 0})
        ds[f"{s}_hour_available"] = ("time", have[s].astype("int8"))
    ds.to_netcdf(FOUT, encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars
                                 if ds[k].ndim == 2})
    logging.info("wrote %s", FOUT)


if __name__ == "__main__":
    main()

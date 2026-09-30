#!/usr/bin/env python
"""
Quality mask for the EURADCLIM evaluation, applied identically to radar and
model before any statistic is computed.

A cell is kept only if it passes all of:

  box        centre inside the evaluation region
  range      within max_range_km of a radar that actually contributed to the
             composite (from the /how/nodes of each hour), band-dependent.
             Needed, not optional: the Spanish composite reports valid zeros
             over a whole rectangle, far beyond any radar's useful range, so
             availability alone does not see where Spain's coverage ends.
  land       land, or sea within coast_buffer_km of it (gauges are on land)
  avail      valid in >= min_availability of hours
  shadow     climatological mean not anomalously LOW against its smoothed
             surroundings (beam blockage, overshooting behind mountains)
  clutter    wet-hour frequency not anomalously HIGH against its surroundings
             (persistent non-meteorological echo)
  radial     not part of a radial streak: around each radar, a direction whose
             climatology stays well below or above its neighbouring directions
             over >= 30 km of range (blockage sectors, interference spokes).
             Tested by shape, so real orographic gradients are kept.
  manual     outside every manual_exclude box

The artefact tests compare EURADCLIM with a smoothed version of itself, not with
the model: using the model to decide where the radar is wrong would bias the
evaluation towards agreement.

Also writes the mask at every aggregated scale (a block is kept if at least
agg_min_valid of its 2 km cells are) and a diagnostic figure. LOOK at the
figure: the thresholds are defaults, and anything they miss belongs in
manual_exclude.

    python make_radar_mask.py
"""

import os
import json
import glob
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import radar_config as cfg
import regions
import radar_utils as ru
import obs_download_utils as util
from plot_obs_model_comparison import INK, INK_MUTED, GRID
from plot_obs_model_maps_relative import DIV

DX_KM = 2.0


def contributing_radars():
    """Radars in >= radar_min_hours_frac of hours, with position and band."""
    counts, nhours = {}, 0
    for fin in sorted(glob.glob(f"{cfg.path_rad_grid}/RAD_01H_RAIN_*.nc")):
        with xr.open_dataset(fin) as ds:
            nhours += ds.sizes["time"] - int(ds.attrs.get("hours_missing", 0))
            for item in ds.attrs.get("nodes", "").split(","):
                if ":" in item:
                    k, v = item.split(":")
                    counts[k] = counts.get(k, 0) + int(v)
    with open(cfg.opera_db) as fh:
        db = {r["odimcode"]: r for r in json.load(fh) if r.get("odimcode")}
    db.update(cfg.extra_radars)

    radars, unknown = [], []
    for node, n in sorted(counts.items()):
        if n < cfg.radar_min_hours_frac * nhours:
            continue
        rec = db.get(cfg.node_alias.get(node, node))
        if rec is None:
            unknown.append(node)
            continue
        radars.append({"node": node, "name": rec["location"],
                       "lat": float(rec["latitude"]), "lon": float(rec["longitude"]),
                       "band": rec.get("band", "C") or "C", "frac": n / nhours})
    if unknown:
        # most are far away (the node list is Europe-wide), but an unplaced radar
        # inside the region would leave a hole in the range mask
        logging.warning("contributing radars not in the OPERA database (alias them "
                        "in node_alias, or place them in extra_radars, if any is in the region): %s", ", ".join(unknown))
    return radars, nhours


def range_mask(lat, lon, radars):
    """Distance to the nearest radar (km) and whether it is within that radar's range."""
    dist = np.full(lat.shape, np.inf)
    ok = np.zeros(lat.shape, bool)
    rlat, rlon = np.radians(lat), np.radians(lon)
    for r in radars:
        a = (np.sin((rlat - np.radians(r["lat"])) / 2) ** 2
             + np.cos(rlat) * np.cos(np.radians(r["lat"]))
             * np.sin((rlon - np.radians(r["lon"])) / 2) ** 2)
        d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        dist = np.minimum(dist, d)
        ok |= d <= cfg.max_range_km.get(r["band"], min(cfg.max_range_km.values()))
    return dist, ok


def smooth_ratio(field, where, floor):
    """field / its Gaussian-smoothed self, the smoothing using only `where` cells.

    NaN where the smoothed reference is below `floor`, so a dry neighbourhood
    passes both tests rather than producing a meaningless ratio.
    """
    sigma = cfg.artefact_sigma_km / DX_KM
    f = np.where(where & np.isfinite(field), field, 0.0)
    w = (where & np.isfinite(field)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = ndimage.gaussian_filter(f, sigma) / ndimage.gaussian_filter(w, sigma)
        return np.where(where & (ref >= floor), field / ref, np.nan)


def running_nanmedian_circular(p, half):
    """Median over +-half bins along the last (azimuth) axis, circular."""
    stack = np.stack([np.roll(p, k, axis=-1) for k in range(-half, half + 1)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(stack, axis=0)


def radial_test(fields, where, lat, lon, radars, floors):
    """Flag cells in radial streaks around the radar that serves them.

    fields: {name: 2D climatology}; each is tested and the flags combined.
    Returns (flag, ratio of the first field) on the grid.
    """
    rlat = np.array([r["lat"] for r in radars])
    rlon = np.array([r["lon"] for r in radars])
    coslat = np.cos(np.radians(lat))
    # nearest radar per cell (km, local equirectangular: fine within ~250 km)
    best = np.full(lat.shape, -1)
    bestd = np.full(lat.shape, np.inf)
    for k in range(len(radars)):
        dx = (lon - rlon[k]) * 111.32 * coslat
        dy = (lat - rlat[k]) * 110.57
        d = np.hypot(dx, dy)
        closer = d < bestd
        best[closer], bestd[closer] = k, d[closer]
    flag = np.zeros(lat.shape, bool)
    ratio_map = np.full(lat.shape, np.nan)
    naz = int(round(360 / cfg.radial_az_bin_deg))
    half = int(round(cfg.radial_ref_halfwidth_deg / cfg.radial_az_bin_deg))
    for k in range(len(radars)):
        cells = where & (best == k)
        if cells.sum() < 100:
            continue
        dx = (lon[cells] - rlon[k]) * 111.32 * coslat[cells]
        dy = (lat[cells] - rlat[k]) * 110.57
        ia = (np.floor((np.degrees(np.arctan2(dx, dy)) % 360) / cfg.radial_az_bin_deg)
              .astype(int) % naz)
        ir = np.floor(np.hypot(dx, dy) / cfg.radial_range_bin_km).astype(int)
        nr = ir.max() + 1
        cell_flag = np.zeros(cells.sum(), bool)
        for j, (name, field) in enumerate(fields.items()):
            v = field[cells]
            df = pd.DataFrame({"ia": ia, "ir": ir, "v": v}).dropna()
            g = df.groupby(["ir", "ia"]).v
            med = g.median()[g.size() >= cfg.radial_min_cells]
            p = np.full((nr, naz), np.nan)
            p[med.index.get_level_values(0), med.index.get_level_values(1)] = med.values
            ref = running_nanmedian_circular(p, half)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(ref >= floors[name], p / ref, np.nan)
            bad = {}
            for sign, test in (("low", r < cfg.radial_low), ("high", r > cfg.radial_high)):
                # a run of >= radial_min_run consecutive range bins along one azimuth
                run = np.zeros_like(test, dtype=int)
                for i in range(nr):
                    run[i] = np.where(test[i], (run[i - 1] if i else 0) + 1, 0)
                keep = np.zeros_like(test)
                for i in range(nr - 1, -1, -1):
                    # back-fill: every bin of a long enough run is flagged
                    long_run = run[i] >= cfg.radial_min_run
                    keep[i] |= long_run
                    if i + 1 < nr:
                        keep[i] |= test[i] & keep[i + 1]
                bad[sign] = keep
            polar_bad = bad["low"] | bad["high"]
            cell_flag |= polar_bad[ir, ia]
            if j == 0:
                vals = ratio_map[cells]
                vals = r[ir, ia]
                ratio_map[cells] = vals
        tmp = flag[cells]
        tmp |= cell_flag
        flag[cells] = tmp
    return flag, ratio_map


def main():
    util.start_logger("make_radar_mask")
    grid = ru.model_grid()
    lat, lon = grid["lat"], grid["lon"]
    years = range(cfg.syear, cfg.eyear + 1)
    rad = ru.load_stats("EURADCLIM", 1, years, cfg.months)
    if rad is None:
        raise RuntimeError("no EURADCLIM statistics; run radar_model_stats.py first")

    radars, nhours = contributing_radars()
    logging.info("%d contributing radars, %d radar hours", len(radars), nhours)

    crit = {}
    crit["box"] = grid["in_box"]
    dist, crit["range"] = range_mask(lat, lon, radars)
    coast_km = ndimage.distance_transform_edt(~grid["land"]) * DX_KM
    crit["land"] = coast_km <= cfg.coast_buffer_km
    nvalid = rad.nvalid.values.astype(float)
    avail = nvalid / max(nhours, 1)
    crit["avail"] = avail >= cfg.min_availability

    base = crit["box"] & crit["range"] & crit["land"] & crit["avail"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = rad.total.values / nvalid * 24.0          # mm/day
        freq = rad.nwet.values / nvalid
    r_mean = smooth_ratio(mean, base, cfg.artefact_min_mean)
    r_freq = smooth_ratio(freq, base, cfg.artefact_min_freq)
    grow = max(int(round(cfg.artefact_dilate_km / DX_KM)), 0)
    shadow = base & (r_mean < cfg.shadow_ratio)
    clutter = base & (r_freq > cfg.clutter_ratio)
    if grow:
        shadow = ndimage.binary_dilation(shadow, iterations=grow)
        clutter = ndimage.binary_dilation(clutter, iterations=grow)
    crit["shadow"], crit["clutter"] = ~shadow, ~clutter
    near = [r for r in radars
            if (cfg.lat_min - 3 <= r["lat"] <= cfg.lat_max + 3)
            and (cfg.lon_min - 3 <= r["lon"] <= cfg.lon_max + 3)]
    radial, r_radial = radial_test({"mean": mean, "freq": freq}, base, lat, lon, near,
                                   {"mean": cfg.artefact_min_mean,
                                    "freq": cfg.artefact_min_freq})
    if grow:
        radial = ndimage.binary_dilation(radial, iterations=grow) & base
    crit["radial"] = ~radial
    manual = np.ones(lat.shape, bool)
    for la0, lo0, la1, lo1 in cfg.manual_exclude:
        manual &= ~((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    crit["manual"] = manual

    mask = np.logical_and.reduce(list(crit.values()))

    # what each criterion removes, from the cells in the box
    nbox = crit["box"].sum()
    lines = [f"cells in box: {nbox}"]
    for k, v in crit.items():
        if k != "box":
            lines.append(f"  fails {k:8s}: {(crit['box'] & ~v).sum() / nbox:6.1%}")
    lines.append(f"kept: {mask.sum()} ({mask.sum() / nbox:.1%}, "
                 f"{mask.sum() * DX_KM ** 2 / 1e3:.0f} x 10^3 km2)")
    for name, sub in regions.masks(lat, lon, box=cfg.subregions["ALL"]).items():
        lines.append(f"  {name}: {int((mask & sub).sum())} cells kept of {int(sub.sum())}")
    for line in lines:
        logging.info(line)

    out = {"mask_s1": (("y", "x"), mask.astype("i1"))}
    for k, v in crit.items():
        out[f"ok_{k}"] = (("y", "x"), v.astype("i1"))
    out.update(dist_radar_km=(("y", "x"), dist.astype("float32")),
               availability=(("y", "x"), avail.astype("float32")),
               ratio_mean=(("y", "x"), r_mean.astype("float32")),
               ratio_wetfreq=(("y", "x"), r_freq.astype("float32")),
               ratio_radial_mean=(("y", "x"), r_radial.astype("float32")))
    ds = xr.Dataset(out, coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
    for k in cfg.scales:
        if k == 1:
            continue
        frac = ru.aggregate_static(mask.astype(float), k)
        ds[f"mask_s{k}"] = (("y%d" % k, "x%d" % k), (frac >= cfg.agg_min_valid).astype("i1"))
    ds.attrs.update(
        radars=", ".join(f"{r['node']}({r['name']},{r['band']},{r['frac']:.0%})"
                         for r in radars),
        settings=json.dumps({k: getattr(cfg, k) for k in (
            "max_range_km", "radar_min_hours_frac", "coast_buffer_km",
            "min_availability", "artefact_sigma_km", "shadow_ratio",
            "clutter_ratio", "artefact_min_mean", "artefact_min_freq",
            "artefact_dilate_km", "radial_az_bin_deg", "radial_range_bin_km",
            "radial_ref_halfwidth_deg", "radial_low", "radial_high", "radial_min_run",
            "radial_min_cells", "manual_exclude")}),
        summary="\n".join(lines),
        climatology=f"EURADCLIM {cfg.syear}-{cfg.eyear}, months {cfg.months}")
    ds.to_netcdf(f"{cfg.mask_file}.tmp")
    os.replace(f"{cfg.mask_file}.tmp", cfg.mask_file)
    logging.info("wrote %s", cfg.mask_file)

    figure(grid, radars, mean, dist, avail, r_mean, r_freq, r_radial, crit, mask)


def figure(grid, radars, mean, dist, avail, r_mean, r_freq, r_radial, crit, mask):
    lat, lon = grid["lat"], grid["lon"]
    fig = plt.figure(figsize=(20, 7.6))
    fig.patch.set_facecolor("#fcfcfb")
    box = crit["box"]
    inbox = lambda f: np.ma.masked_where(~box | ~np.isfinite(f), f)
    panels = [
        ("a  EURADCLIM mean (mm/day)", inbox(mean), dict(cmap="viridis",
                                                          norm=LogNorm(0.3, 8))),
        ("b  distance to nearest radar (km)", inbox(dist), dict(cmap="viridis_r",
                                                                vmin=0, vmax=250)),
        ("c  availability (fraction of hours)", inbox(avail), dict(cmap="viridis",
                                                                    vmin=0.5, vmax=1)),
        (f"d  mean / smoothed mean  (out < {cfg.shadow_ratio})", inbox(r_mean),
         dict(cmap=DIV.reversed(), norm=TwoSlopeNorm(1, 0, 2))),
        (f"e  wet freq / smoothed  (out > {cfg.clutter_ratio})", inbox(r_freq),
         dict(cmap=DIV.reversed(), norm=TwoSlopeNorm(1, 0, 3))),
        (f"f  mean / neighbouring azimuths (out <{cfg.radial_low} or >{cfg.radial_high}, "
         f">={cfg.radial_min_run * cfg.radial_range_bin_km:.0f} km)", inbox(r_radial),
         dict(cmap=DIV.reversed(), norm=TwoSlopeNorm(1, 0.5, 1.5))),
    ]
    for i, (title, field, kw) in enumerate(panels):
        # a b c | g   /   d e f | h
        ax = fig.add_subplot(2, 4, (1, 2, 3, 5, 6, 7)[i], projection=ccrs.PlateCarree())
        m = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                          shading="nearest", rasterized=True, **kw)
        decorate(ax, title)
        cb = fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
        if i in (0, 1):
            for r in radars:
                ax.plot(r["lon"], r["lat"], marker="^", ms=5, color="#fcfcfb",
                        mec=INK, mew=0.8, transform=ccrs.PlateCarree())

    # radial flags alone, to see exactly what the new test removes
    ax = fig.add_subplot(2, 4, 4, projection=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, np.ma.masked_where(~box, mean), cmap="Greys", norm=LogNorm(0.3, 8),
                  transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    ax.pcolormesh(lon, lat, np.ma.masked_where(crit["radial"] | ~box, np.ones(lat.shape)),
                  cmap=ListedColormap(["#eb6834"]), transform=ccrs.PlateCarree(),
                  shading="nearest", rasterized=True)
    decorate(ax, "g  removed by the radial test (orange), over the mean")

    # final: kept cells over orography, and the reason for the rest
    ax = fig.add_subplot(2, 4, 8, projection=ccrs.PlateCarree())
    reason = np.full(lat.shape, np.nan)
    order = ["range", "land", "avail", "shadow", "clutter", "radial", "manual"]
    for j, k in enumerate(order):
        reason = np.where(np.isnan(reason) & box & ~crit[k], j, reason)
    hgt = np.ma.masked_where(~mask, grid["hgt"])
    ax.pcolormesh(lon, lat, hgt, cmap="Greys", vmin=-500, vmax=2500,
                  transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    cols = ["#e6e6e2", "#c9c8c3", "#8fb8e8", "#6d2a0f", "#1baf7a", "#eb6834", "#52514e"]
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(reason), cmap=ListedColormap(cols),
                  vmin=-0.5, vmax=len(order) - 0.5, transform=ccrs.PlateCarree(),
                  shading="nearest", rasterized=True)
    decorate(ax, "h  kept (grey = terrain) and why the rest is out")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, [f"out: {k}" for k in order], fontsize=6.5, loc="lower right",
              frameon=True, framealpha=0.9, ncol=2)

    fig.suptitle(f"EURADCLIM quality mask — {cfg.syear}-{cfg.eyear} "
                 f"({mask.sum() * DX_KM ** 2 / 1e3:.0f} ×10³ km² kept)",
                 fontsize=12, color=INK, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{cfg.path_rad_figs}/radar_mask.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)


def decorate(ax, title):
    ax.set_extent([cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID, alpha=0.8)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK_MUTED}
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")


if __name__ == "__main__":
    main()

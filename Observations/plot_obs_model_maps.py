#!/usr/bin/env python
"""
Where the storms are: track density maps for the observations and for the
coarsened model under both brightness-temperature conversions, plus the
differences and the diurnal cycle of storm initiation.

The distribution comparisons answer "are the storms the right size / length /
intensity"; this answers "are they in the right place, and do they start at the
right time of day", which the aggregate statistics cannot see. A model can match
every distribution and still put its storms over the wrong sea.

    python plot_obs_model_maps.py
    python plot_obs_model_maps.py --datasets obs mod0.1_YS_pres

Outputs, beside the distribution figure:
    obs_model_maps_<exp>.png
    obs_model_maps_<exp>_numbers.txt    the numbers, for quoting
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import obs_config as cfg
from plot_obs_model_comparison import LABELS, SHORT, COLORS, INK, INK_MUTED, GRID

BINSIZE = 0.5          # degrees; ~30 track points per cell over the record
EARTH_R = 6371.0       # km

# Sequential ramp: one hue, light to dark, from the categorical blue. Diverging
# ramp: the categorical orange and blue with a neutral (not white, not a hue)
# midpoint, so zero difference reads as absence rather than as a colour.
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fc", "#a8c6ea", "#2a78d6", "#12386b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8a3413", "#eb6834", "#efeeea", "#2a78d6", "#123c66"])


def load_tracks(dataset, exp):
    """Every (storm id, lat, lon, hour, month, year) point in the record."""
    root = f"{cfg.path_tracking}/{dataset}/{exp}"
    sid, lats, lons, ids, months, starts = 0, [], [], [], [], []
    for year in range(cfg.syear, cfg.eyear + 1):
        for month in cfg.months:
            fin = f"{root}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue
            for storm in pd.read_pickle(fin).values():
                track = storm["track"]
                inside = ((track[:, 0] >= cfg.lat_min) & (track[:, 0] <= cfg.lat_max)
                          & (track[:, 1] >= cfg.lon_min) & (track[:, 1] <= cfg.lon_max))
                if not inside.any():
                    continue
                sid += 1
                lats.append(track[:, 0])
                lons.append(track[:, 1])
                ids.append(np.full(track.shape[0], sid))
                months.append(np.full(track.shape[0], month))
                starts.append(pd.Timestamp(storm["times"][0]).hour)
    if sid == 0:
        return None
    return (np.concatenate(lats), np.concatenate(lons), np.concatenate(ids),
            np.array(starts), sid, np.concatenate(months))


def density(lat, lon, storm_id):
    """Storms per year per 10^4 km^2 on a regular lat/lon grid.

    Each storm is counted once per cell it passes through, so a slow storm does
    not outweigh a fast one -- this is a track density, not a residence time.
    """
    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + BINSIZE, BINSIZE)
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + BINSIZE, BINSIZE)
    jj = np.digitize(lat, lat_edges) - 1
    ii = np.digitize(lon, lon_edges) - 1
    ok = ((jj >= 0) & (jj < lat_edges.size - 1) & (ii >= 0) & (ii < lon_edges.size - 1))
    # unique (storm, cell) pairs
    pairs = np.unique(np.stack([storm_id[ok], jj[ok], ii[ok]], axis=1), axis=0)
    counts = np.zeros((lat_edges.size - 1, lon_edges.size - 1))
    np.add.at(counts, (pairs[:, 1].astype(int), pairs[:, 2].astype(int)), 1)

    # cell area in 10^4 km^2, shrinking with latitude
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    dy = BINSIZE * np.pi / 180 * EARTH_R
    dx = BINSIZE * np.pi / 180 * EARTH_R * np.cos(np.radians(latc))
    area = (dy * dx)[:, None] / 1e4
    nyears = cfg.eyear - cfg.syear + 1
    return counts / nyears / area, lat_edges, lon_edges


def add_map(ax, lat_edges, lon_edges, field, cmap, norm=None, vmax=None):
    ax.set_extent([cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max],
                  crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(lon_edges, lat_edges, field, cmap=cmap, norm=norm,
                         vmin=None if norm else 0, vmax=None if norm else vmax,
                         transform=ccrs.PlateCarree(), shading="flat")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID, alpha=0.8)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK_MUTED}
    return mesh


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--datasets", nargs="+",
                     default=["obs", "mod0.1_YS_pres", "mod0.1_SB_pres"])
    par.add_argument("--exp", default="exp1")
    par.add_argument("--out", default=None)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    tracks, dens, edges = {}, {}, None
    for dataset in args.datasets:
        got = load_tracks(dataset, args.exp)
        if got is None:
            logging.warning("no storms for %s", dataset)
            continue
        lat, lon, sid, starts, n, mon = got
        tracks[dataset] = (lat, lon, sid, starts, n, mon)
        field, lat_e, lon_e = density(lat, lon, sid)
        dens[dataset] = field
        edges = (lat_e, lon_e)
        logging.info("%-18s %5d storms, %6d track points", dataset, n, lat.size)

    lat_e, lon_e = edges
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14.5, 7.6))
    fig.patch.set_facecolor("#fcfcfb")
    vmax = max(np.percentile(f[f > 0], 99) for f in dens.values())

    # row 1: the three density fields, same scale
    for i, dataset in enumerate(args.datasets):
        if dataset not in dens:
            continue
        ax = fig.add_subplot(2, 3, i + 1, projection=proj)
        mesh = add_map(ax, lat_e, lon_e, dens[dataset], SEQ, vmax=vmax)
        ax.set_title(f"{'abc'[i]}  {LABELS.get(dataset, dataset)}", loc="left",
                     fontsize=10, color=INK, fontweight="bold")
        if i == 2:
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("storms yr$^{-1}$ per 10$^4$ km$^2$", fontsize=8, color=INK)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    # row 2: differences against the observations
    models = [d for d in args.datasets if d != "obs" and d in dens]
    dmax = max(np.abs(dens[d] - dens["obs"]).max() for d in models) if models else 1
    for i, dataset in enumerate(models):
        ax = fig.add_subplot(2, 3, 4 + i, projection=proj)
        diff = dens[dataset] - dens["obs"]
        mesh = add_map(ax, lat_e, lon_e, diff, DIV,
                       norm=TwoSlopeNorm(vcenter=0, vmin=-dmax, vmax=dmax))
        ax.set_title(f"{'de'[i]}  {SHORT.get(dataset, dataset)} − observations",
                     loc="left", fontsize=10, color=INK, fontweight="bold")
        if i == len(models) - 1:
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("difference", fontsize=8, color=INK)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    # panel f: diurnal cycle of storm initiation
    ax = fig.add_subplot(2, 3, 6)
    ax.set_facecolor("#fcfcfb")
    for dataset in args.datasets:
        if dataset not in tracks:
            continue
        starts = tracks[dataset][3]
        counts = np.bincount(starts, minlength=24) / starts.size * 100
        colour = COLORS["obs"] if dataset == "obs" else \
            COLORS["YS" if "_YS_" in dataset else "SB"]
        ax.plot(np.arange(24), counts, color=colour, lw=2,
                solid_capstyle="round", label=LABELS.get(dataset, dataset))
        ax.annotate(SHORT.get(dataset, dataset), (23, counts[23]),
                    xytext=(4, 0), textcoords="offset points",
                    color=colour, fontsize=7.5, va="center")
    ax.set_xlabel("hour of storm initiation (UTC)", fontsize=9, color=INK)
    ax.set_ylabel("% of storms", fontsize=9, color=INK)
    ax.set_xlim(0, 23)
    ax.set_title("f  Diurnal cycle of initiation", loc="left", fontsize=10,
                 color=INK, fontweight="bold")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)

    fig.suptitle("Where and when storms occur: observations vs coarsened model "
                 f"({cfg.reg}, {cfg.syear}-{cfg.eyear}, {args.exp}, 0.1° grid)",
                 fontsize=12, color=INK, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = args.out or f"{cfg.path_obs}/obs_model_maps_{args.exp}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    logging.info("wrote %s", out)

    write_numbers(out.replace(".png", "_numbers.txt"), args, tracks, dens, lat_e, lon_e)


def write_numbers(path, args, tracks, dens, lat_e, lon_e):
    """The numbers behind the maps, for quoting in the discussion."""
    latc = 0.5 * (lat_e[:-1] + lat_e[1:])
    lonc = 0.5 * (lon_e[:-1] + lon_e[1:])
    lines = [f"Spatial comparison, {cfg.reg} {cfg.syear}-{cfg.eyear}, {args.exp}, "
             f"{BINSIZE} deg bins", ""]

    lines.append(f"{'dataset':34s}{'storms/yr':>10}{'peak dens':>11}"
                 f"{'mean lat':>10}{'mean lon':>10}{'r vs obs':>10}")
    obs_field = dens.get("obs")
    for dataset in args.datasets:
        if dataset not in dens:
            continue
        f = dens[dataset]
        w = f / f.sum()
        mlat = float((w.sum(axis=1) * latc).sum())
        mlon = float((w.sum(axis=0) * lonc).sum())
        r = "" if dataset == "obs" else \
            f"{np.corrcoef(f.ravel(), obs_field.ravel())[0, 1]:10.3f}"
        lines.append(f"{LABELS.get(dataset, dataset):34s}"
                     f"{tracks[dataset][4] / (cfg.eyear - cfg.syear + 1):10.1f}"
                     f"{f.max():11.2f}{mlat:10.2f}{mlon:10.2f}{r:>10s}")

    # northern vs southern sub-region, the split the manuscript already uses
    lines += ["", "Storms per year by sub-region (centroid inside at any time):",
              f"  {'dataset':34s}{'south <40.3N':>14}{'north >40.5N':>14}{'S/N ratio':>11}"]
    for dataset in args.datasets:
        if dataset not in tracks:
            continue
        lat, lon, sid, _, n, _mon = tracks[dataset]
        nyr = cfg.eyear - cfg.syear + 1
        south = np.unique(sid[lat <= 40.3]).size / nyr
        north = np.unique(sid[lat >= 40.5]).size / nyr
        lines.append(f"  {LABELS.get(dataset, dataset):34s}{south:14.1f}{north:14.1f}"
                     f"{south / max(north, 1e-9):11.2f}")

    # The afternoon peak alone hides the clearest model-observation difference:
    # the observations have a second, nocturnal maximum around 21 UTC that the
    # model does not produce, so the evening share is reported separately.
    lines += ["", "Diurnal cycle of initiation:",
              f"  {'dataset':34s}{'peak hr':>9}{'min hr':>8}{'peak/min':>10}"
              f"{'% 09-15':>9}{'% 19-01':>9}{'21 UTC %':>10}"]
    for dataset in args.datasets:
        if dataset not in tracks:
            continue
        starts = tracks[dataset][3]
        c = np.bincount(starts, minlength=24) / starts.size * 100
        night = c[19:24].sum() + c[0:2].sum()
        lines.append(f"  {LABELS.get(dataset, dataset):34s}{int(c.argmax()):9d}"
                     f"{int(c.argmin()):8d}{c.max() / c.min():10.2f}"
                     f"{c[9:16].sum():9.1f}{night:9.1f}{c[21]:10.1f}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    logging.info("wrote %s", path)


if __name__ == "__main__":
    main()

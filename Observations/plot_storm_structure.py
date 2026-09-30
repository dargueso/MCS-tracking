#!/usr/bin/env python
"""
Storm-relative rain-rate structure: composite the precipitation field around
every tracked storm centre, so model and observations can be compared on how
rain is *arranged* within a storm rather than only on bulk totals.

The distribution comparisons say the model makes smaller storms with higher peak
rain rates. This says whether that is a narrower, more intense core, a weaker
surrounding shield, or both -- which the aggregate numbers cannot distinguish.

Composites are built on the storm-relative grid in model/observation grid cells
and converted to kilometres afterwards. Cells are ~11.1 km meridionally and
~8.4 km zonally at 41N, so the two axes are scaled separately rather than
pretending the box is square.

    python plot_storm_structure.py
    python plot_storm_structure.py --max-storms 300      # quick test

Outputs:
    storm_structure_<exp>.png
    storm_structure_<exp>_numbers.txt
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import obs_config as cfg
from plot_obs_model_comparison import LABELS, SHORT, COLORS, INK, INK_MUTED, GRID

HALF = 36              # composite half-width, in grid cells (~400 x 300 km)
EARTH_R = 6371.0

# Same perceptually uniform multi-hue ramp as the relative maps: rain rate spans
# a wide range and a single-hue ramp loses the outer shield entirely.
SEQ = plt.get_cmap("magma_r")


def rain_file(dataset, tag):
    """The RAIN file a tracked dataset was built from."""
    return cfg.datasets[dataset][0].format(tag=tag)


def composite(dataset, exp, max_storms=None):
    """Mean rain rate on the storm-relative grid, plus the storm count."""
    root = f"{cfg.path_tracking}/{dataset}/{exp}"
    total = np.zeros((2 * HALF + 1, 2 * HALF + 1))
    count = np.zeros_like(total)
    nstorm = nstep = 0

    for year in range(cfg.syear, cfg.eyear + 1):
        for month in cfg.months:
            fin = f"{root}/MCS_{year}{month:02d}"
            tag = f"{year}-{month:02d}"
            frain = rain_file(dataset, tag)
            if not (os.path.exists(fin) and os.path.exists(frain)):
                continue
            storms = pd.read_pickle(fin)
            if not storms:
                continue
            with xr.open_dataset(frain) as dset:
                rain = dset.RAIN.values
                lat = dset.lat.values[:, 0]
                lon = dset.lon.values[0, :]
                times = pd.to_datetime(dset.time.values)
            tindex = {t: i for i, t in enumerate(times)}

            for storm in storms.values():
                track = storm["track"]
                inside = ((track[:, 0] >= cfg.lat_min) & (track[:, 0] <= cfg.lat_max)
                          & (track[:, 1] >= cfg.lon_min) & (track[:, 1] <= cfg.lon_max))
                if not inside.any():
                    continue
                nstorm += 1
                if max_storms and nstorm > max_storms:
                    break
                for k, when in enumerate(storm["times"]):
                    it = tindex.get(pd.Timestamp(when))
                    if it is None:
                        continue
                    # nearest grid cell to the storm centre
                    jy = int(np.abs(lat - track[k, 0]).argmin())
                    jx = int(np.abs(lon - track[k, 1]).argmin())
                    y0, y1 = jy - HALF, jy + HALF + 1
                    x0, x1 = jx - HALF, jx + HALF + 1
                    # clip at the domain edge and place into the composite box
                    cy0, cx0 = max(0, -y0), max(0, -x0)
                    y0c, x0c = max(0, y0), max(0, x0)
                    y1c, x1c = min(rain.shape[1], y1), min(rain.shape[2], x1)
                    if y1c <= y0c or x1c <= x0c:
                        continue
                    patch = rain[it, y0c:y1c, x0c:x1c]
                    sl = (slice(cy0, cy0 + patch.shape[0]),
                          slice(cx0, cx0 + patch.shape[1]))
                    good = np.isfinite(patch)
                    total[sl] += np.where(good, patch, 0.0)
                    count[sl] += good
                    nstep += 1
                if max_storms and nstorm > max_storms:
                    break
            if max_storms and nstorm > max_storms:
                break
        if max_storms and nstorm > max_storms:
            break

    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    logging.info("%-18s composite from %d storms, %d time steps",
                 dataset, nstorm, nstep)
    return mean, nstorm, nstep


def axes_km():
    """Storm-relative axes in km, using the true cell size at mid-domain."""
    midlat = 0.5 * (cfg.lat_min + cfg.lat_max)
    dy = cfg.target_res * np.pi / 180 * EARTH_R
    dx = dy * np.cos(np.radians(midlat))
    off = np.arange(-HALF, HALF + 1)
    return off * dx, off * dy


def radial_profile(field, x_km, y_km, step=25.0):
    """Azimuthal mean rain rate against distance from the storm centre."""
    xx, yy = np.meshgrid(x_km, y_km)
    r = np.sqrt(xx ** 2 + yy ** 2)
    edges = np.arange(0, r.max(), step)
    mids, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi) & np.isfinite(field)
        if sel.sum():
            mids.append(0.5 * (lo + hi))
            vals.append(field[sel].mean())
    return np.array(mids), np.array(vals)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--datasets", nargs="+",
                     default=["obs", "mod0.1_YS_pres", "mod0.1_SB_pres"])
    par.add_argument("--exp", default="exp1")
    par.add_argument("--max-storms", type=int, default=None)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    comps = {}
    for dataset in args.datasets:
        field, nstorm, nstep = composite(dataset, args.exp, args.max_storms)
        if nstorm:
            comps[dataset] = (field, nstorm, nstep)

    x_km, y_km = axes_km()
    fig = plt.figure(figsize=(14.5, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    vmax = max(np.nanpercentile(f, 99.5) for f, _, _ in comps.values())

    for i, dataset in enumerate(args.datasets):
        if dataset not in comps:
            continue
        field, nstorm, _ = comps[dataset]
        ax = fig.add_subplot(1, 4, i + 1)
        ax.set_facecolor("#fcfcfb")
        mesh = ax.pcolormesh(x_km, y_km, field, cmap=SEQ, vmin=0, vmax=vmax,
                             shading="auto")
        ax.contour(x_km, y_km, field, levels=[1, 2, 5], colors="#ffffff",
                   linewidths=0.6, alpha=0.8)
        ax.set_aspect("equal")
        ax.axhline(0, color="#ffffff", lw=0.5, alpha=0.6)
        ax.axvline(0, color="#ffffff", lw=0.5, alpha=0.6)
        ax.set_title(f"{'abc'[i]}  {SHORT.get(dataset, dataset)}  (n={nstorm})",
                     loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("east–west distance (km)", fontsize=8, color=INK)
        if i == 0:
            ax.set_ylabel("north–south distance (km)", fontsize=8, color=INK)
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        if i == 2:
            cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.03, extend="max")
            cb.set_label("mean rain rate (mm h$^{-1}$)", fontsize=8, color=INK)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    # radial profiles, the quantitative version of the three maps
    ax = fig.add_subplot(1, 4, 4)
    ax.set_facecolor("#fcfcfb")
    rows = []
    for dataset in args.datasets:
        if dataset not in comps:
            continue
        field = comps[dataset][0]
        r, v = radial_profile(field, x_km, y_km)
        colour = COLORS["obs"] if dataset == "obs" else \
            COLORS["YS" if "_YS_" in dataset else "SB"]
        ax.plot(r, v, color=colour, lw=2, solid_capstyle="round",
                label=LABELS.get(dataset, dataset))
        rows.append((dataset, field, r, v))
    # The profiles converge beyond ~300 km, so end-of-line labels collide. A
    # legend is the honest relief here; direct labels would have to sit where
    # the curves are still separated, which is the least interesting part.
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for text, (dataset, *_ ) in zip(leg.get_texts(), rows):
        text.set_color(COLORS["obs"] if dataset == "obs"
                       else COLORS["YS" if "_YS_" in dataset else "SB"])
    ax.set_xlabel("distance from storm centre (km)", fontsize=8, color=INK)
    ax.set_ylabel("mean rain rate (mm h$^{-1}$)", fontsize=8, color=INK)
    ax.set_title("d  Radial profile", loc="left", fontsize=10, color=INK,
                 fontweight="bold")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle("Storm-relative rain-rate structure — "
                 f"{cfg.reg}, {cfg.syear}-{cfg.eyear}, {args.exp}, 0.1° grid",
                 fontsize=12, color=INK, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{cfg.path_figs_sat}/storm_structure_{args.exp}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    logging.info("wrote %s", out)

    # the numbers: how peaked, how wide, how much rain sits far from the centre
    lines = ["Storm-relative rain-rate structure", "",
             f"  {'dataset':34s}{'centre':>9}{'r=50km':>9}{'r=100km':>9}"
             f"{'r=200km':>9}{'e-fold':>9}{'% >100km':>10}"]
    for dataset, field, r, v in rows:
        centre = np.nanmax(field)
        def at(target):
            return v[np.abs(r - target).argmin()]
        # radius at which the profile falls to 1/e of the innermost ring
        thresh = v[0] / np.e
        below = np.where(v < thresh)[0]
        efold = r[below[0]] if below.size else np.nan
        xx, yy = np.meshgrid(x_km, y_km)
        rad = np.sqrt(xx ** 2 + yy ** 2)
        far = np.nansum(np.where((rad > 100) & np.isfinite(field), field, 0))
        allr = np.nansum(np.where(np.isfinite(field), field, 0))
        lines.append(f"  {LABELS.get(dataset, dataset):34s}{centre:9.2f}"
                     f"{at(50):9.2f}{at(100):9.2f}{at(200):9.2f}"
                     f"{efold:9.0f}{100 * far / allr:10.1f}")
    lines += ["", "  centre = peak of the composite; r=X = azimuthal mean at that "
                  "radius (mm/h);",
              "  e-fold = radius where the profile drops to 1/e of the innermost "
              "ring;",
              "  % >100km = share of composite rain falling outside 100 km."]
    text = "\n".join(lines)
    print("\n" + text)
    with open(out.replace(".png", "_numbers.txt"), "w") as fh:
        fh.write(text + "\n")
    logging.info("wrote %s", out.replace(".png", "_numbers.txt"))


if __name__ == "__main__":
    main()

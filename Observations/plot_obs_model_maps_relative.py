#!/usr/bin/env python
"""
Standardised storm-density maps: the *relative* spatial pattern, with the
overall storm count divided out.

The absolute maps in plot_obs_model_maps.py mix two things -- how many storms a
dataset produces and where it puts them. Stefan-Boltzmann produces twice as many
storms as observed, so its map is darker everywhere and the pattern is hard to
judge. Dividing each field by its own domain mean removes the count entirely:
1.0 is an average cell for that dataset, so the maps answer only "where do this
dataset's storms sit, relative to its own total".

Two figures:
    obs_model_maps_<exp>_relative.png   all months, 3 fields + 2 differences
    obs_model_maps_<exp>_seasonal.png   the same, split DJF/MAM/JJA/SON

    python plot_obs_model_maps_relative.py
"""

import os
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs

import obs_config as cfg
from plot_obs_model_comparison import LABELS, SHORT, INK, INK_MUTED
from plot_obs_model_maps import load_tracks, density, add_map

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
           "JJA": (6, 7, 8), "SON": (9, 10, 11)}

# A perceptually uniform, multi-hue sequential ramp. This is deliberately not a
# rainbow: hue varies but lightness still increases monotonically, so the map
# reads correctly in greyscale and under colour-vision deficiency while giving
# far more discriminable steps than a single-hue blue ramp. Values here span
# 0 to ~5x the domain mean, and the single-hue version made everything below 2x
# look identical.
SEQ = plt.get_cmap("viridis")

# Differences stay on a true diverging ramp: two hues, neutral (not white, not a
# hue) at zero, so "same as observed" reads as absence of colour.
DIV = LinearSegmentedColormap.from_list(
    "div", ["#6d2a0f", "#eb6834", "#f6b48f", "#efeeea",
            "#8fb8e8", "#2a78d6", "#123c66"])


def relative(field):
    """Density as a multiple of that dataset's own domain mean."""
    good = field[field > 0]
    return field / good.mean() if good.size else field


def panel_grid(fig, nrow, ncol, i, proj):
    return fig.add_subplot(nrow, ncol, i, projection=proj)


def figure_relative(tracks, args, proj):
    """All months: three standardised fields plus the two differences."""
    fields, edges = {}, None
    for dataset in args.datasets:
        if dataset not in tracks:
            continue
        lat, lon, sid, _, _, _ = tracks[dataset]
        f, lat_e, lon_e = density(lat, lon, sid)
        fields[dataset] = relative(f)
        edges = (lat_e, lon_e)
    lat_e, lon_e = edges

    fig = plt.figure(figsize=(14.5, 7.6))
    fig.patch.set_facecolor("#fcfcfb")
    vmax = np.percentile(np.concatenate([f[f > 0] for f in fields.values()]), 99)

    for i, dataset in enumerate(args.datasets):
        if dataset not in fields:
            continue
        ax = panel_grid(fig, 2, 3, i + 1, proj)
        mesh = add_map(ax, lat_e, lon_e, fields[dataset], SEQ, vmax=vmax)
        ax.set_title(f"{'abc'[i]}  {LABELS.get(dataset, dataset)}", loc="left",
                     fontsize=10, color=INK, fontweight="bold")
        if i == 2:
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, extend="max")
            cb.set_label("density / own domain mean", fontsize=8, color=INK)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    models = [d for d in args.datasets if d != "obs" and d in fields]
    dmax = max(np.abs(fields[d] - fields["obs"]).max() for d in models)
    dmax = min(dmax, np.percentile(
        np.concatenate([np.abs(fields[d] - fields["obs"]).ravel() for d in models]), 99))
    for i, dataset in enumerate(models):
        ax = panel_grid(fig, 2, 3, 4 + i, proj)
        mesh = add_map(ax, lat_e, lon_e, fields[dataset] - fields["obs"], DIV,
                       norm=TwoSlopeNorm(vcenter=0, vmin=-dmax, vmax=dmax))
        ax.set_title(f"{'de'[i]}  {SHORT.get(dataset, dataset)} − observations",
                     loc="left", fontsize=10, color=INK, fontweight="bold")
        if i == len(models) - 1:
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, extend="both")
            cb.set_label("difference in relative density", fontsize=8, color=INK)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle("Relative storm density (each field divided by its own domain "
                 f"mean) — {cfg.reg}, {cfg.syear}-{cfg.eyear}, {args.exp}",
                 fontsize=12, color=INK, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = f"{cfg.path_obs}/obs_model_maps_{args.exp}_relative.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)
    return fields


def figure_seasonal(tracks, args, proj):
    """Standardised within each season, so each panel is its own pattern."""
    nrow, ncol = len(SEASONS), len(args.datasets)
    fig = plt.figure(figsize=(4.6 * ncol, 2.9 * nrow))
    fig.patch.set_facecolor("#fcfcfb")

    fields = {}
    for season, months in SEASONS.items():
        for dataset in args.datasets:
            if dataset not in tracks:
                continue
            lat, lon, sid, _, _, mon = tracks[dataset]
            pick = np.isin(mon, months)
            if pick.sum() == 0:
                continue
            f, lat_e, lon_e = density(lat[pick], lon[pick], sid[pick])
            fields[(season, dataset)] = (relative(f), lat_e, lon_e)

    allv = np.concatenate([f[f > 0] for f, _, _ in fields.values()])
    vmax = np.percentile(allv, 99)

    mesh = None
    for r, season in enumerate(SEASONS):
        for c, dataset in enumerate(args.datasets):
            if (season, dataset) not in fields:
                continue
            f, lat_e, lon_e = fields[(season, dataset)]
            ax = panel_grid(fig, nrow, ncol, r * ncol + c + 1, proj)
            mesh = add_map(ax, lat_e, lon_e, f, SEQ, vmax=vmax)
            n = int(np.unique(tracks[dataset][2][
                np.isin(tracks[dataset][5], SEASONS[season])]).size)
            title = (f"{season} · {SHORT.get(dataset, dataset)}"
                     f"  (n={n})")
            ax.set_title(title, loc="left", fontsize=9, color=INK,
                         fontweight="bold")

    cax = fig.add_axes([0.25, 0.055, 0.5, 0.013])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("density / own domain mean (standardised within each panel)",
                 fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle("Relative storm density by season — "
                 f"{cfg.reg}, {cfg.syear}-{cfg.eyear}, {args.exp}",
                 fontsize=12, color=INK, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0.085, 1, 0.965])
    out = f"{cfg.path_obs}/obs_model_maps_{args.exp}_seasonal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)
    return fields


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--datasets", nargs="+",
                     default=["obs", "mod0.1_YS_pres", "mod0.1_SB_pres"])
    par.add_argument("--exp", default="exp1")
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    tracks = {}
    for dataset in args.datasets:
        got = load_tracks(dataset, args.exp)
        if got is None:
            logging.warning("no storms for %s", dataset)
            continue
        tracks[dataset] = got
        logging.info("%-18s %5d storms", dataset, got[4])

    fields = figure_relative(tracks, args, ccrs.PlateCarree())
    seasonal = figure_seasonal(tracks, args, ccrs.PlateCarree())

    # numbers: how well does the relative pattern match, and where is it worst
    lines = ["", "Relative spatial pattern vs observations "
                 "(count divided out; 1.0 = that dataset's average cell)", ""]
    lines.append(f"  {'dataset':34s}{'pattern r':>11}{'mean |diff|':>13}"
                 f"{'max excess':>12}{'max deficit':>13}")
    for dataset in args.datasets:
        if dataset == "obs" or dataset not in fields:
            continue
        d = fields[dataset] - fields["obs"]
        r = np.corrcoef(fields[dataset].ravel(), fields["obs"].ravel())[0, 1]
        lines.append(f"  {LABELS.get(dataset, dataset):34s}{r:11.3f}"
                     f"{np.abs(d).mean():13.3f}{d.max():12.2f}{d.min():13.2f}")

    lines += ["", "Seasonal share of storms (% of that dataset's total):", ""]
    lines.append(f"  {'dataset':34s}" + "".join(f"{s:>9}" for s in SEASONS))
    for dataset in args.datasets:
        if dataset not in tracks:
            continue
        ids, mon = tracks[dataset][2], tracks[dataset][5]
        tot = np.unique(ids).size
        share = [100 * np.unique(ids[np.isin(mon, m)]).size / tot
                 for m in SEASONS.values()]
        lines.append(f"  {LABELS.get(dataset, dataset):34s}" +
                     "".join(f"{v:9.1f}" for v in share))

    text = "\n".join(lines)
    print(text)
    out = f"{cfg.path_obs}/obs_model_maps_{args.exp}_relative_numbers.txt"
    with open(out, "w") as fh:
        fh.write(text + "\n")
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()

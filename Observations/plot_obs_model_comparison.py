#!/usr/bin/env python
"""
Compare tracked storm statistics between the observations and the coarsened
model, on the common 0.1 deg grid with identical tracker settings.

Distributions are shown as cumulative curves rather than histograms: there is no
bin width to choose, the whole distribution is visible at once, and differences
in the tail -- which is where the storms that matter live -- stay legible.

    python plot_obs_model_comparison.py
    python plot_obs_model_comparison.py --datasets obs mod0.1_YS_pres
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
from scipy import stats

import obs_config as cfg

# Categorical slots 1-3 of the validated default palette. Checked with the
# data-viz palette validator (light surface #fcfcfb): lightness band, chroma
# floor, CVD separation (worst adjacent dE 9.2 deutan) and normal-vision floor
# all pass. The aqua carries a contrast warning against the surface, which is
# why every series is also directly labelled rather than relying on the legend.
COLORS = {"obs": "#2a78d6", "YS": "#eb6834", "SB": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

LABELS = {"obs": "IMERG + MERGIR",
          "mod0.1_YS_pres": "EPICC 0.1° (Yang & Slingo)",
          "mod0.1_SB_pres": "EPICC 0.1° (Stefan-Boltzmann)",
          "mod0.1_YS_fut": "EPICC 0.1° PGW (Yang & Slingo)",
          "mod0.1_SB_fut": "EPICC 0.1° PGW (Stefan-Boltzmann)"}

# Short forms for the direct labels in panel (a). The full names are too long to
# sit beside a line, but they must still tell the two model series apart -- the
# BT method IS the difference between them, so it cannot be the part dropped.
SHORT = {"obs": "IMERG+MERGIR",
         "mod0.1_YS_pres": "EPICC YS",
         "mod0.1_SB_pres": "EPICC SB",
         "mod0.1_YS_fut": "EPICC YS (PGW)",
         "mod0.1_SB_fut": "EPICC SB (PGW)"}


def colour(dataset):
    if dataset == "obs":
        return COLORS["obs"]
    return COLORS["YS" if "_YS_" in dataset else "SB"]


def load(dataset, exp="exp1"):
    """Per-storm characteristics for one dataset, as a DataFrame."""
    root = f"{cfg.path_tracking}/{dataset}/{exp}"
    rows = []
    for year in range(cfg.syear, cfg.eyear + 1):
        for month in cfg.months:
            fin = f"{root}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue
            storms = pd.read_pickle(fin)
            for storm in storms.values():
                track = storm["track"]          # (nstep, 2) lat, lon
                # Same criterion as the model analysis: a storm counts if its
                # centre is inside the region at any point, but only the
                # timesteps inside contribute to its statistics - duration
                # included. Project convention, 2026-09-30; keeping the whole
                # track instead inflates area, duration and volume by amounts
                # that differ between datasets, because tracks leave the box
                # at different rates.
                inside = ((track[:, 0] >= cfg.lat_min) & (track[:, 0] <= cfg.lat_max)
                          & (track[:, 1] >= cfg.lon_min) & (track[:, 1] <= cfg.lon_max))
                if not inside.any():
                    continue
                rows.append({
                    "year": year, "month": month,
                    "area": np.nanmax(np.asarray(storm["size"])[inside]) / 1e6,     # km2
                    "duration": int(inside.sum()),                                  # h
                    "peak": np.nanmax(np.asarray(storm["max"])[inside]),            # mm/h
                    "volume": np.nansum(np.asarray(storm["volume"])[inside]) / 1e6, # 10^6 m3
                })
    if not rows:
        logging.warning("no storms found for %s", dataset)
    return pd.DataFrame(rows)


def cdf_panel(ax, data, key, xlabel, logx=False):
    """Cumulative distribution of one characteristic, one line per dataset."""
    for dataset, frame in data.items():
        if frame.empty:
            continue
        vals = np.sort(frame[key].values)
        ax.plot(vals, np.arange(1, vals.size + 1) / vals.size,
                color=colour(dataset), lw=2, solid_capstyle="round",
                label=LABELS.get(dataset, dataset))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cumulative fraction", fontsize=9, color=INK)
    style(ax)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


TAIL_Q = (90, 95, 99)
NBOOT = 10000


def year_blocks(frame, key):
    """Per-year arrays of one characteristic, for block resampling."""
    return {y: g[key].values for y, g in frame.groupby("year")}


def tail_ratio_ci(obs, mod, key, rng):
    """model/observed quantile ratios in the tail, with a year-block CI.

    The tail is where the science sits, and it is exactly where the KS statistic
    is weakest, so the two summaries answer different questions.

    Years are resampled in blocks, the same draw applied to observations and
    model, matching the paired year-block bootstrap described in the Methods:
    the two runs share a synoptic sequence, so a year that happens to be stormy
    is stormy in both and must be drawn for both together. Resampling storms
    individually would also understate the uncertainty, since storms within a
    season are not independent.
    """
    ob, mb = year_blocks(obs, key), year_blocks(mod, key)
    years = sorted(set(ob) & set(mb))
    point = {q: np.percentile(mod[key], q) / np.percentile(obs[key], q)
             for q in TAIL_Q}

    draws = {q: [] for q in TAIL_Q}
    for _ in range(NBOOT):
        pick = rng.choice(years, size=len(years), replace=True)
        o = np.concatenate([ob[y] for y in pick])
        m = np.concatenate([mb[y] for y in pick])
        for q in TAIL_Q:
            draws[q].append(np.percentile(m, q) / np.percentile(o, q))
    return {q: (point[q], *np.percentile(draws[q], [2.5, 97.5])) for q in TAIL_Q}


def tail_summary(data, args):
    """Print, and save, the tail-focused comparison."""
    rng = np.random.default_rng(20260928)
    print(f"\nTail summary: model/observed quantile ratio, "
          f"{NBOOT} paired year-block resamples")
    print("  1.00 = model matches observations at that quantile; a 95% interval "
          "excluding 1\n  means the difference survives interannual sampling "
          "variability.")
    rows = []
    for dataset, frame in data.items():
        if dataset == "obs" or frame.empty:
            continue
        print(f"\n  {LABELS.get(dataset, dataset)}")
        print(f"    {'variable':10s}" +
              "".join(f"{'p' + str(q):>22s}" for q in TAIL_Q))
        for key, unit in (("area", "km2"), ("duration", "h"),
                          ("peak", "mm/h"), ("volume", "1e6 m3")):
            res = tail_ratio_ci(data["obs"], frame, key, rng)
            cells = []
            for q in TAIL_Q:
                point, lo, hi = res[q]
                star = " " if lo <= 1.0 <= hi else "*"
                cells.append(f"{point:6.2f} [{lo:.2f},{hi:.2f}]{star}")
                rows.append({"dataset": dataset, "variable": key, "unit": unit,
                             "quantile": q, "ratio": point,
                             "ci_lo": lo, "ci_hi": hi,
                             "obs_value": np.percentile(data["obs"][key], q),
                             "model_value": np.percentile(frame[key], q)})
            print(f"    {key:10s}" + "".join(f"{c:>22s}" for c in cells))
    print("\n  * = 95% interval excludes 1")
    out = (args.out or f"{cfg.path_figs_sat}/obs_model_comparison_{args.exp}.png")
    out = out.replace(".png", "_tail.csv")
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    logging.info("wrote %s", out)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--datasets", nargs="+",
                     default=["obs", "mod0.1_YS_pres", "mod0.1_SB_pres"])
    par.add_argument("--exp", default="exp1")
    par.add_argument("--out", default=None)
    args = par.parse_args()

    logging.basicConfig(format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    data = {}
    for dataset in args.datasets:
        frame = load(dataset, args.exp)
        data[dataset] = frame
        logging.info("%-18s %5d storms", dataset, len(frame))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    # (a) seasonal cycle -- storms per month, averaged over the record
    ax = axes[0, 0]
    nyears = cfg.eyear - cfg.syear + 1
    for dataset, frame in data.items():
        if frame.empty:
            continue
        counts = frame.groupby("month").size().reindex(cfg.months, fill_value=0) / nyears
        ax.plot(counts.index, counts.values, color=colour(dataset), lw=2,
                marker="o", ms=4, solid_capstyle="round",
                label=LABELS.get(dataset, dataset))
        ax.annotate(SHORT.get(dataset, dataset),
                    (counts.index[-1], counts.values[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=colour(dataset), fontsize=7.5, va="center")
    ax.set_xlabel("month", fontsize=9, color=INK)
    ax.set_ylabel("storms per month per year", fontsize=9, color=INK)
    ax.set_title("a  Seasonal cycle", loc="left", fontsize=10,
                 color=INK, fontweight="bold")
    style(ax)

    # (b) interannual -- storms per year
    ax = axes[0, 1]
    for dataset, frame in data.items():
        if frame.empty:
            continue
        counts = (frame.groupby("year").size()
                  .reindex(range(cfg.syear, cfg.eyear + 1), fill_value=0))
        ax.plot(counts.index, counts.values, color=colour(dataset), lw=2,
                marker="o", ms=4, solid_capstyle="round",
                label=LABELS.get(dataset, dataset))
    ax.set_xlabel("year", fontsize=9, color=INK)
    ax.set_ylabel("storms per year", fontsize=9, color=INK)
    ax.set_title("b  Interannual variability", loc="left", fontsize=10,
                 color=INK, fontweight="bold")
    style(ax)

    # (c)-(f) the distributions
    panels = [(axes[0, 2], "area", "max instantaneous area (km$^2$)", True,
               "c  Storm area"),
              (axes[1, 0], "duration", "duration (h)", False, "d  Duration"),
              (axes[1, 1], "peak", "peak rain rate (mm h$^{-1}$)", False,
               "e  Peak rain rate"),
              (axes[1, 2], "volume", "total rain volume (10$^6$ m$^3$)", True,
               "f  Rain volume")]
    for ax, key, xlabel, logx, title in panels:
        cdf_panel(ax, data, key, xlabel, logx)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Tracked storm statistics: observations vs coarsened model "
                 f"({cfg.reg}, {cfg.syear}-{cfg.eyear}, {args.exp}, 0.1° grid)",
                 fontsize=12, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])

    out = args.out or f"{cfg.path_figs_sat}/obs_model_comparison_{args.exp}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    logging.info("wrote %s", out)

    # the numbers behind the figure, so the comparison can be quoted
    print(f"\n{'':22s}{'storms':>8}{'med area':>10}{'med dur':>9}"
          f"{'med peak':>10}{'med vol':>10}")
    print(f"{'':22s}{'':>8}{'(km2)':>10}{'(h)':>9}{'(mm/h)':>10}{'(10^6m3)':>10}")
    for dataset, frame in data.items():
        if frame.empty:
            continue
        print(f"{LABELS.get(dataset, dataset):22s}{len(frame):8d}"
              f"{frame.area.median():10.0f}{frame.duration.median():9.0f}"
              f"{frame.peak.median():10.1f}{frame.volume.median():10.0f}")
    if "obs" in data and not data["obs"].empty:
        print("\nKolmogorov-Smirnov D vs observations (whole distribution):")
        print("  D is the largest gap between the cumulative curves. Its power sits "
              "near the\n  median, so it says little about the largest storms -- see "
              "the tail summary below.\n  p-values are omitted deliberately: with "
              f"n~{len(data['obs'])} every difference is 'significant'.")
        for dataset, frame in data.items():
            if dataset == "obs" or frame.empty:
                continue
            bits = [f"{key} D={stats.ks_2samp(data['obs'][key], frame[key]).statistic:.3f}"
                    for key in ("area", "duration", "peak", "volume")]
            print(f"  {LABELS.get(dataset, dataset):34s} " + " | ".join(bits))

        tail_summary(data, args)


if __name__ == "__main__":
    main()

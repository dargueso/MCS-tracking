#!/usr/bin/env python
"""
Scatter-histogram of storm characteristics: IMERG + MERGIR against the
present-day model, both tracked on the same 0.1 deg grid with the same
settings (Observations/track_storms.py). Same layout as
plot_scatter_hist_storm_characteristics.py (present vs future), but the
observations have no 10-m wind, so the axes are

    x  peak rain rate (mm/h)           y  duration (h)
    marker size  maximum instantaneous area (10^3 km2), same scale as the
                 present/future figure          colour  total rain volume (10^6 m3)

Top left: model present; top right: IMERG + MERGIR; bottom left: both, the
model in the blue of the present/future figure and the observations in a
neutral grey.

    python plot_scatter_hist_obs_model.py            # exp1, ASON, WME

A storm counts if its centre lies inside the region at any point of its life,
as in the present/future figure and in plot_obs_model_comparison.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as cfg

exp = sys.argv[1] if len(sys.argv) > 1 else "exp1"
reg = "WME"
syear, eyear = 2011, 2020
allmonths = [8, 9, 10, 11]
TRACK = "/scratch3/dargueso/obs-mcs-tracking/tracking"
DATASETS = {"model": "mod0.1_YS_pres", "obs": "obs"}
LABEL = {"model": "EPICC present (0.1°)", "obs": "IMERG + MERGIR"}
COLOR = {"model": "deepskyblue", "obs": "dimgray"}
la0, lo0, la1, lo1 = cfg.reg_coords[reg]


def summary(dataset):
    rows = []
    for year in range(syear, eyear + 1):
        for month in allmonths:
            fin = f"{TRACK}/{dataset}/{exp}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue
            for storm in pd.read_pickle(fin).values():
                track = storm["track"]
                inside = ((track[:, 0] >= la0) & (track[:, 0] <= la1) & (track[:, 1] >= lo0) & (track[:, 1] <= lo1))
                if not inside.any():
                    continue
                rows.append(dict(prmax=np.nanmax(storm["max"]), max_size=np.nanmax(storm["size"]) / 1e6,
                                 tot_vol=np.nansum(storm["volume"]), duration=len(storm["times"])))
    return pd.DataFrame(rows)


data = {k: summary(d) for k, d in DATASETS.items()}
for k, f in data.items():
    print(f"{LABEL[k]}: {len(f)} storms; median peak {f.prmax.median():.1f} mm/h, area {f.max_size.median():.0f} km2, "
          f"duration {f.duration.median():.0f} h, volume {f.tot_vol.median() / 1e6:.0f} x10^6 m3")
pr_max = float(np.ceil(max(f.prmax.max() for f in data.values()) / 10) * 10)
du_max = float(np.ceil(max(f.duration.max() for f in data.values()) / 10) * 10)   # h
vmin, vmax = 10, 1e4
size = lambda f: (f.max_size / 100) * 2          # as in plot_scatter_hist_storm_characteristics.py

fig = plt.figure(figsize=(30, 20))
gs1 = fig.add_gridspec(2, 2, width_ratios=(9, 9), height_ratios=(9, 9), left=0.05, right=0.95, bottom=0.05,
                       top=0.95, wspace=0.1, hspace=0.1)
sub = [gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7), subplot_spec=gs1[i, j],
                                        wspace=0, hspace=0) for i, j in ((0, 0), (0, 1), (1, 0))]
gs03 = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=(7, 2), height_ratios=(0.5, 8.5), subplot_spec=gs1[1, 1],
                                        wspace=0, hspace=0)
xbins, ybins = np.linspace(0, pr_max, 20), np.linspace(0, du_max, 20)


def panel(gs, share=None):
    ax = fig.add_subplot(gs[1, 0], sharex=share, sharey=share)
    hx = fig.add_subplot(gs[0, 0], sharex=ax)
    hy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.set_xlim(0, pr_max); ax.set_ylim(0, du_max)
    hx.tick_params(axis="x", labelbottom=False); hy.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Max. precip. rate (mm/hr)", fontsize="xx-large")
    ax.set_ylabel("Duration (h)", fontsize="xx-large")
    ax.tick_params(labelsize="x-large"); hx.tick_params(labelsize="x-large"); hy.tick_params(labelsize="x-large")
    # the corner labels collide with the histogram axes, as in the original figure
    plt.setp(ax.get_xticklabels()[-1], visible=False)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    return ax, hx, hy


def hists(hx, hy, f, color, edge):
    w = np.ones(len(f)) / len(f)
    hx.hist(f.prmax, bins=xbins, color=color, alpha=0.8, edgecolor=edge, rwidth=0.8, weights=w)
    hy.hist(f.duration, bins=ybins, color=color, alpha=0.8, edgecolor=edge, orientation="horizontal",
            rwidth=0.8, weights=w)


def size_legend(ax, sct):
    # the same size marks in every panel, whatever each dataset's range
    areas = [5, 20, 50, 100]                      # 10^3 km2
    handles = [plt.scatter([], [], s=(a * 1e3 / 100) * 2, color="gray", alpha=0.6) for a in areas]
    leg = ax.legend(handles, [f"{a}" for a in areas], loc="upper left", bbox_to_anchor=(0.0, 0.8),
                    title="Storm size ($10^3$ $km^2$)",
                    frameon=False, labelspacing=2, borderaxespad=1, ncol=7, fontsize="xx-large",
                    title_fontsize="xx-large", handletextpad=1.5, columnspacing=2.5)
    plt.setp(leg.get_title(), multialignment="center")


sct = None
ax0 = None
for i, key in enumerate(("model", "obs")):
    f = data[key]
    ax, hx, hy = panel(sub[i], share=ax0)
    ax0 = ax0 or ax
    sct = ax.scatter(f.prmax, f.duration, s=size(f), c=f.tot_vol / 1e6, cmap="Spectral",
                     norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
    ax.text(0.95, 0.95, LABEL[key], fontsize="xx-large", fontweight="bold", color=COLOR[key],
            horizontalalignment="right", transform=ax.transAxes)
    hists(hx, hy, f, "lightgray", "gray")

# both together
ax3, hx3, hy3 = panel(sub[2], share=ax0)
for key in ("obs", "model"):
    f = data[key]
    ax3.scatter(f.prmax, f.duration, s=size(f), c=COLOR[key], alpha=0.5)
    hists(hx3, hy3, f, COLOR[key], COLOR[key])
ax3.text(0.88, 0.90, f"{len(data['model'])}", fontsize="xx-large", fontweight="bold", color=COLOR["model"],
         transform=ax3.transAxes)
ax3.text(0.88, 0.85, f"{len(data['obs'])}", fontsize="xx-large", fontweight="bold", color=COLOR["obs"],
         transform=ax3.transAxes)

# one size legend for all panels, in the free quadrant under the colour bar
leg_ax = fig.add_subplot(gs03[1, 0])
leg_ax.axis("off")
size_legend(leg_ax, sct)
cbar_ax = fig.add_subplot(gs03[0, 0])
cbar = plt.colorbar(sct, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Total rain volume ($10^6$ $m^3$)", fontsize="xx-large")
cbar.ax.tick_params(labelsize="x-large")

fig.suptitle(f"Storm characteristics in the Western Mediterranean (Aug-Nov): model vs IMERG + MERGIR, "
             f"same tracker on the 0.1° grid ({exp})", fontsize=30, fontweight="bold")
out = f"{cfg.path_figures}/MCS-tracking/Scatter_storm_characteristics_obs_model_{reg}_{exp}.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)

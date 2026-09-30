#!/usr/bin/env python
"""
Seasonal maxima at the AEMET (Arnau) stations: the events that matter.

Quantiles up to p99.9 of all days do not reach the events the study is about.
Here every station-season gives one maximum of the daily total (P24) and of
the 60-min amount (PMAX60), from the validated AEMET days; the model gives
its own maximum over the same station-days (cell and 3x3 max). The maxima are
pooled per region and compared rank by rank (a Q-Q of block maxima, on a
Gumbel axis), plus the median ratio of the paired station-season maxima and
the interannual series of the regional mean.

Outputs, in {cfg.path_arnau_figs}:
    arnau_model_maxima_<tag>.png
    arnau_model_maxima_<tag>_numbers.txt
    arnau_model_maxima_<tag>.csv        every station-season maximum, both sides

    python plot_annual_maxima.py             # ASON
    python plot_annual_maxima.py --all-months
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import station_config as cfg
import seasons
from plot_obs_model_comparison import INK, INK_MUTED, GRID
from plot_station_model import C_OBS, C_MOD, region_of
from plot_arnau_model import load, LABEL

MIN_COVER = 0.8
NBOOT = 1000


def maxima(d, months):
    years = d["days"].year.values
    rows = []
    for v in ("p24", "pmax60"):
        o, val = d["obs"][v], d["valid"][v]
        for y in np.unique(years):
            s = years == y
            # coverage: valid days over the days of the season in that year
            nday = s.sum()
            cover = val[:, s].sum(1) / nday
            ob = np.where(val[:, s], o[:, s], -np.inf).max(1)
            mc = np.where(val[:, s], d["mod"]["cell"][v][:, s], -np.inf).max(1)
            mn = np.where(val[:, s], d["mod"]["nmax"][v][:, s], -np.inf).max(1)
            ok = (cover >= MIN_COVER) & (ob > 0)
            for st in np.where(ok)[0]:
                rows.append(dict(station=d["codes"][st], year=int(y), var=v, arnau=ob[st],
                                 cell=mc[st], nmax=mn[st], lat=d["lat"][st], lon=d["lon"][st]))
    return pd.DataFrame(rows)


def gumbel_axis(n):
    """Reduced variate of the Gringorten plotting positions for n ranked values."""
    p = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    return -np.log(-np.log(p))


def ranked_ratio(o, m, top):
    """Ratio of the mean of the top `top` fraction, ranked values."""
    k = max(int(round(top * o.size)), 1)
    return np.sort(m)[-k:].mean() / np.sort(o)[-k:].mean()


def bootstrap_top(o, m, years, top, rng):
    ys = np.unique(years)
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(ys, size=ys.size, replace=True)
        sel = np.concatenate([np.where(years == y)[0] for y in pick])
        out.append(ranked_ratio(o[sel], m[sel], top))
    return np.percentile(out, [2.5, 97.5])


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(cfg.path_arnau_figs, exist_ok=True)
    months, tag = seasons.resolve(args)
    d = load(months)
    regs = region_of(d["lat"], d["lon"])
    mx = maxima(d, months)
    mx.to_csv(f"{cfg.path_arnau_figs}/arnau_model_maxima_{tag}.csv", index=False, float_format="%.4g")
    rng = np.random.default_rng(0)
    stmask = {r: dict(zip(d["codes"], m)) for r, m in regs.items()}

    lines = [f"Station-season maxima, EPICC vs AEMET (Arnau), {tag}, 2011-2019, station-seasons with >= "
             f"{MIN_COVER:.0%} validated days",
             "Ranked (pooled) comparison: model/Arnau ratio of the mean of the top 10% and top 1% of maxima "
             "[95% year-block bootstrap]; paired: median ratio per station-season"]
    fig, axes = plt.subplots(2, len(regs), figsize=(3.2 * len(regs), 6.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for c, reg in enumerate(regs):
        for r, v in enumerate(("p24", "pmax60")):
            g = mx[(mx["var"] == v) & mx.station.map(stmask[reg])]
            ax = axes[r, c]
            if len(g) < 20:
                ax.set_title(f"{reg} · {LABEL[v]} (n={len(g)})", loc="left", fontsize=9, color=INK, fontweight="bold")
                continue
            o = g.arnau.values; yrs = g.year.values
            xg = gumbel_axis(o.size)
            ax.plot(xg, np.sort(o), color=C_OBS, lw=2, label="AEMET (Arnau)")
            parts = [f"n={len(g)}"]
            for k, ls, lab in (("cell", "-", "EPICC cell"), ("nmax", "--", "EPICC 3x3 max")):
                m = g[k].values
                ax.plot(xg, np.sort(m), color=C_MOD, lw=1.8, ls=ls, label=lab, alpha=1 if k == "cell" else 0.6)
                r10 = ranked_ratio(o, m, 0.10); r1 = ranked_ratio(o, m, 0.01)
                lo10, hi10 = bootstrap_top(o, m, yrs, 0.10, rng); lo1, hi1 = bootstrap_top(o, m, yrs, 0.01, rng)
                parts.append(f"{k} top10% {r10:.2f} [{lo10:.2f}-{hi10:.2f}] top1% {r1:.2f} [{lo1:.2f}-{hi1:.2f}] "
                             f"paired median {np.median(m / o):.2f}")
            lines.append(f"  {reg:4s} {LABEL[v]:7s} " + "   ".join(parts))
            ax.set_title(f"{reg} · {LABEL[v]} season maxima", loc="left", fontsize=9, color=INK, fontweight="bold")
            ax.set_xlabel("Gumbel reduced variate", fontsize=7.5, color=INK)
            ax.set_ylabel("mm", fontsize=7.5, color=INK)
            for T in (2, 5, 10, 50):
                ax.axvline(-np.log(-np.log(1 - 1 / T)), color=GRID, lw=0.6)
            ax.grid(True, color=GRID, lw=0.5); ax.tick_params(labelsize=7, colors=INK_MUTED)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Station-season maxima, EPICC vs AEMET (Arnau), {tag} 2011-2019 (ranked, Gumbel axis; "
                 "vertical lines: 2, 5, 10, 50 station-seasons)", fontsize=10.5, color=INK, fontweight="bold",
                 x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{cfg.path_arnau_figs}/arnau_model_maxima_{tag}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())

    # interannual: regional mean daily rain per year, Arnau vs model (validated days)
    years = d["days"].year.values
    lines.append("\nInterannual: regional mean of P24 (mm/day) per year, Arnau | EPICC cell, and correlation")
    for reg, rm in regs.items():
        o, m, val = d["obs"]["p24"][rm], d["mod"]["cell"]["p24"][rm], d["valid"]["p24"][rm]
        yo = [np.nanmean(o[:, years == y][val[:, years == y]]) for y in np.unique(years)]
        ym = [np.nanmean(m[:, years == y][val[:, years == y]]) for y in np.unique(years)]
        lines.append(f"  {reg:4s} " + " ".join(f"{a:.2f}|{b:.2f}" for a, b in zip(yo, ym))
                     + f"   r = {np.corrcoef(yo, ym)[0, 1]:.2f}")
    text = "\n".join(lines)
    print(text)
    with open(f"{cfg.path_arnau_figs}/arnau_model_maxima_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()

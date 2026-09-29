#!/usr/bin/env python
"""
Hourly-intensity Q-Q comparison of EPICC 2 km against EURADCLIM, per subregion
and spatial scale, plus the diurnal cycle.

Distributions, not matched pairs: the ERA5-driven run follows the large-scale
weather but its convective cells do not line up hour by hour with the observed
ones, so a point-to-point comparison mostly measures timing noise. Pooling every
masked cell-hour of a subregion and comparing quantiles does not care when or
exactly where a cell rained.

Quantiles are of ALL hours, dry ones included (p99 = the intensity exceeded
1% of the time), so the frequency of rain and its intensity are judged together;
the wet-hour fraction is printed separately so the two can be told apart.

Uncertainty: year-block bootstrap, paired (the same resampled years for both
datasets), as in plot_obs_model_comparison.py. Hourly rain is correlated in
space and time, so resampling cell-hours would give intervals far too narrow;
years are close to independent.

Outputs, in {cfg.path_rad_figs}:
    radar_model_qq_<months>.png        Q-Q, subregions x scales
    radar_model_qq_<months>.csv        every quantile, ratio and interval
    radar_model_diurnal_<months>.png   mean rain and wet-hour frequency by hour

    python plot_radar_model_qq.py                    # ASON, the study season
    python plot_radar_model_qq.py --all-months       # whole year
    python plot_radar_model_qq.py --season JJA      # or --months 6 7 8
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

import radar_config as cfg
import radar_utils as ru
import seasons
from seasons import SEASONS
from plot_obs_model_comparison import INK, INK_MUTED, GRID, COLORS

RAD, MOD = "EURADCLIM", cfg.model_run
C_RAD, C_MOD = COLORS["SB"], COLORS["YS"]   # EURADCLIM aqua, EPICC orange, as in every figure (gauges blue)
PROBS = np.array([0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
                  0.9995, 0.9999, 0.99995, 0.99999])
LABELLED = {0.99: "p99", 0.999: "p99.9", 0.9999: "p99.99"}
# Resampling fewer years than this gives intervals that are not intervals
# (with one year, every resample is that year): report none rather than a
# spuriously "significant" zero-width one.
MIN_BOOT_YEARS = 3
MONTH_TAG = {sel: name for name, sel in SEASONS.items()}   # kept for older imports


def region_masks(scale):
    """{region: boolean (y, x)} at this scale: quality mask AND region box."""
    with xr.open_dataset(cfg.mask_file) as ds:
        mask = ds[f"mask_s{scale}"].values.astype(bool)
    grid = ru.model_grid()
    lat = ru.aggregate_static(grid["lat"], scale)
    lon = ru.aggregate_static(grid["lon"], scale)
    return {name: mask & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
            for name, (la0, lo0, la1, lo1) in cfg.subregions.items()}


def regional(dataset, scale, months, masks):
    """Per-year regional sums: {year: {region: dict of arrays}}.

    Reduces each monthly file to the regions straight away, so a full year of
    2 km histograms is never in memory at once.
    """
    out = {}
    for year in range(cfg.syear, cfg.eyear + 1):
        for month in months:
            fin = ru.stats_file(dataset, scale, year, month)
            if not os.path.exists(fin):
                continue
            with xr.open_dataset(fin) as ds:
                hist = ds.hist.values
                nvalid, total, nwet = ds.nvalid.values, ds.total.values, ds.nwet.values
                dsum, dvalid, dwet = ds.dsum.values, ds.dvalid.values, ds.dwet.values
            yr = out.setdefault(year, {})
            for name, m in masks.items():
                acc = yr.setdefault(name, {"hist": 0, "nvalid": 0, "total": 0, "nwet": 0,
                                           "dsum": 0, "dvalid": 0, "dwet": 0})
                acc["hist"] = acc["hist"] + hist[:, m].sum(1)
                acc["nvalid"] += nvalid[m].sum()
                acc["total"] += total[m].sum()
                acc["nwet"] += nwet[m].sum()
                acc["dsum"] = acc["dsum"] + dsum[:, m].sum(1)
                acc["dvalid"] = acc["dvalid"] + dvalid[:, m].sum(1)
                acc["dwet"] = acc["dwet"] + dwet[:, m].sum(1)
    return out


def quantiles(per_year, years, region, edges):
    hist = sum(per_year[y][region]["hist"] for y in years)
    nvalid = sum(per_year[y][region]["nvalid"] for y in years)
    if nvalid == 0:
        return np.full(PROBS.size, np.nan)
    return ru.quantile_from_hist(hist, nvalid - hist.sum(), PROBS, edges)


def bootstrap(rad, mod, region, edges, nboot, seed=0):
    """Paired year-block bootstrap of both quantile sets and their ratio."""
    years = sorted(set(rad) & set(mod))
    if len(years) < MIN_BOOT_YEARS:
        nan = np.full((2, PROBS.size), np.nan)
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    qr = np.empty((nboot, PROBS.size))
    qm = np.empty((nboot, PROBS.size))
    for b in range(nboot):
        pick = rng.choice(years, size=len(years), replace=True)
        qr[b] = quantiles(rad, pick, region, edges)
        qm[b] = quantiles(mod, pick, region, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(qr > 0, qm / qr, np.nan)
    ci = lambda a: np.nanpercentile(a, [2.5, 97.5], axis=0)
    return ci(qr), ci(qm), ci(ratio)


def summary(per_year, region):
    tot = {k: sum(per_year[y][region][k] for y in per_year)
           for k in ("nvalid", "total", "nwet", "dsum", "dvalid", "dwet")}
    return tot


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--nboot", type=int, default=1000)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        level=logging.INFO)
    months, tag = seasons.resolve(args)
    edges = ru.hist_edges()
    regions = list(cfg.subregions)

    data, rows = {}, []
    for k in cfg.scales:
        masks = region_masks(k)
        rad = regional(RAD, k, months, masks)
        mod = regional(MOD, k, months, masks)
        if not rad or not mod:
            raise RuntimeError(f"no statistics at scale {k}; run radar_model_stats.py")
        years = sorted(set(rad) & set(mod))
        for reg in regions:
            q_r = quantiles(rad, years, reg, edges)
            q_m = quantiles(mod, years, reg, edges)
            ci_r, ci_m, ci_ratio = bootstrap(rad, mod, reg, edges, args.nboot)
            sr, sm = summary(rad, reg), summary(mod, reg)
            data[(k, reg)] = dict(q_r=q_r, q_m=q_m, ci_r=ci_r, ci_m=ci_m,
                                  sr=sr, sm=sm, ncell=int(masks[reg].sum()), years=years)
            for i, p in enumerate(PROBS):
                rows.append({"scale_km": 2 * k, "region": reg, "prob": p,
                             "euradclim": q_r[i], "model": q_m[i],
                             "ratio": q_m[i] / q_r[i] if q_r[i] > 0 else np.nan,
                             "ratio_lo": ci_ratio[0, i], "ratio_hi": ci_ratio[1, i],
                             "euradclim_lo": ci_r[0, i], "euradclim_hi": ci_r[1, i],
                             "model_lo": ci_m[0, i], "model_hi": ci_m[1, i]})

    table = pd.DataFrame(rows)
    fcsv = f"{cfg.path_rad_figs}/radar_model_qq_{tag}.csv"
    table.to_csv(fcsv, index=False, float_format="%.4g")
    logging.info("wrote %s", fcsv)
    report(data, table, tag)
    figure_qq(data, regions, tag)
    figure_diurnal(data, regions, tag)
    figure_seasonal(regions)


def figure_seasonal(regions):
    """Gridded seasonal cycle, EPICC vs EURADCLIM over the masked cells, 2 km.

    Always all twelve months, whatever --season says. Mean rain, wet-hour
    frequency and all-hour P99 by month, and the EPICC/EURADCLIM ratio of
    each. The station version (plot_station_model.py) adds the gauges.
    """
    k = cfg.scales[0]
    masks = region_masks(k)
    edges = ru.hist_edges()
    rows = [("mean rain (mm/day)", "mean"), ("wet hours (%)", "wet"), ("all-hour P99 (mm/h)", "p99")]
    vals = {}
    for m in range(1, 13):
        for ds in (RAD, MOD):
            per = regional(ds, k, [m], masks)
            yrs = sorted(per)
            for reg in regions:
                if not yrs:
                    continue
                nv = sum(per[y][reg]["nvalid"] for y in yrs)
                if nv == 0:
                    continue
                vals[(ds, reg, m, "mean")] = sum(per[y][reg]["total"] for y in yrs) / nv * 24
                vals[(ds, reg, m, "wet")] = 100 * sum(per[y][reg]["nwet"] for y in yrs) / nv
                vals[(ds, reg, m, "p99")] = quantiles(per, yrs, reg, edges)[
                    int(np.argmin(np.abs(PROBS - 0.99)))]
    fig, axes = plt.subplots(len(rows) + 1, len(regions), figsize=(3.1 * len(regions), 8.6),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    mm = np.arange(1, 13)
    for c, reg in enumerate(regions):
        for r, (label, key) in enumerate(rows):
            ax = axes[r, c]
            for ds, col, lab in ((RAD, C_RAD, "EURADCLIM"), (MOD, C_MOD, "EPICC")):
                ax.plot(mm, [vals.get((ds, reg, m, key), np.nan) for m in mm], color=col, lw=2,
                        marker="o", ms=3.5, label=lab)
            ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, color=INK)
        ax = axes[-1, c]
        for (label, key), ls in zip(rows, ("-", "--", ":")):
            ratio = [vals.get((MOD, reg, m, key), np.nan) / vals.get((RAD, reg, m, key), np.nan)
                     if vals.get((RAD, reg, m, key), 0) else np.nan for m in mm]
            ax.plot(mm, ratio, color=INK_MUTED, lw=1.6, ls=ls, label=label.split(" (")[0])
        ax.axhline(1, color=INK_MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["¼", "½", "1", "2", "4"])
        if c == 0:
            ax.set_ylabel("EPICC / EURADCLIM", fontsize=8, color=INK)
        axes[0, c].set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    for ax in axes.ravel():
        ax.set_xticks(mm)
        ax.set_xticklabels("JFMAMJJASOND")
        ax.axvspan(7.5, 11.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[-1, 0].get_legend_handles_labels()
    fig.legend(h + h2, l + ["ratio: " + x for x in l2], loc="upper right", ncol=5, fontsize=7.5,
               frameon=False)
    fig.suptitle(f"Seasonal cycle, EPICC vs EURADCLIM, masked, {2 * k} km, "
                 f"{cfg.syear}-{cfg.eyear}; shaded: ASON", fontsize=11, color=INK,
                 fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{cfg.path_rad_figs}/radar_model_seasonal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)


def report(data, table, tag):
    lines = [f"EPICC vs EURADCLIM, {tag}, all-hour quantiles (mm/h), "
             "ratio model/radar with 95% year-block interval; * = excludes 1"]
    for (k, reg), d in data.items():
        sr, sm = d["sr"], d["sm"]
        if sr["nvalid"] == 0:
            continue
        lines.append(
            f"\n{reg} at {2 * k} km: {d['ncell']} cells, years {d['years'][0]}-{d['years'][-1]}\n"
            f"  mean     {sr['total'] / sr['nvalid'] * 24:6.2f} vs {sm['total'] / sm['nvalid'] * 24:6.2f} mm/day\n"
            f"  wet hrs  {100 * sr['nwet'] / sr['nvalid']:6.2f} vs {100 * sm['nwet'] / sm['nvalid']:6.2f} %")
        sub = table[(table.scale_km == 2 * k) & (table.region == reg)]
        for _, r in sub[sub.prob.isin(list(LABELLED))].iterrows():
            star = "*" if (r.ratio_lo > 1 or r.ratio_hi < 1) else " "
            lines.append(f"  {LABELLED[r.prob]:7s}  {r.euradclim:6.2f} vs {r.model:6.2f}"
                         f"   ratio {r.ratio:5.2f} [{r.ratio_lo:4.2f}-{r.ratio_hi:4.2f}]{star}")
    text = "\n".join(lines)
    print(text)
    with open(f"{cfg.path_rad_figs}/radar_model_qq_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


def figure_qq(data, regions, tag):
    nrow, ncol = len(cfg.scales), len(regions)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.25 * nrow), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    good = np.concatenate([np.r_[d["q_r"], d["q_m"]] for d in data.values()])
    good = good[np.isfinite(good) & (good > 0)]
    lo, hi = good.min() / 1.5, good.max() * 1.5

    for r, k in enumerate(cfg.scales):
        for c, reg in enumerate(regions):
            ax = axes[r, c]
            d = data[(k, reg)]
            ok = (d["q_r"] > 0) & (d["q_m"] > 0)
            ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=0.8, ls="--", zorder=1)
            xerr = np.abs(d["ci_r"][:, ok] - d["q_r"][ok])
            yerr = np.abs(d["ci_m"][:, ok] - d["q_m"][ok])
            ax.errorbar(d["q_r"][ok], d["q_m"][ok], xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=C_MOD, elinewidth=0.9, alpha=0.55, zorder=2)
            ax.plot(d["q_r"][ok], d["q_m"][ok], "o", ms=5.5, color=C_MOD,
                    mec="#fcfcfb", mew=1.2, zorder=3)
            for p, lab in LABELLED.items():
                i = int(np.argmin(np.abs(PROBS - p)))
                if ok[i]:
                    ax.annotate(lab, (d["q_r"][i], d["q_m"][i]), xytext=(-6, 5),
                                textcoords="offset points", ha="right",
                                fontsize=7, color=INK_MUTED)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, which="major", color=GRID, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7, colors=INK_MUTED)
            sr, sm = d["sr"], d["sm"]
            if sr["nvalid"]:
                ax.text(0.97, 0.04,
                        f"wet hours {100 * sr['nwet'] / sr['nvalid']:.1f}% → "
                        f"{100 * sm['nwet'] / sm['nvalid']:.1f}%\n"
                        f"mean {sr['total'] / sr['nvalid'] * 24:.2f} → "
                        f"{sm['total'] / sm['nvalid'] * 24:.2f} mm/d\n"
                        f"{d['ncell']} cells",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.5, color=INK_MUTED)
            ax.set_title(f"{reg} · {2 * k} km", loc="left", fontsize=9.5,
                         color=INK, fontweight="bold")
            if r == nrow - 1:
                ax.set_xlabel("EURADCLIM (mm/h)", fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel("EPICC 2 km (mm/h)", fontsize=8, color=INK)

    fig.suptitle(f"Hourly rain quantiles, all hours — EPICC vs EURADCLIM, masked, {tag} "
                 f"{cfg.syear}-{cfg.eyear}  (p90 … p99.999; bars: 95% year-block bootstrap)",
                 fontsize=11, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = f"{cfg.path_rad_figs}/radar_model_qq_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)


def figure_diurnal(data, regions, tag):
    """Mean rain and wet-hour frequency by UTC hour, native scale."""
    k = cfg.scales[0]
    fig, axes = plt.subplots(2, len(regions), figsize=(3.1 * len(regions), 5.2),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    hours = np.arange(24) + 0.5                  # hour START stamps: plot mid-hour
    for c, reg in enumerate(regions):
        d = data[(k, reg)]
        for r, (key, scale, ylab) in enumerate((("dsum", 1.0, "mean rain (mm/h)"),
                                                ("dwet", 100.0, "wet hours (%)"))):
            ax = axes[r, c]
            for s, col, lab in ((d["sr"], C_RAD, "EURADCLIM"), (d["sm"], C_MOD, "EPICC")):
                with np.errstate(invalid="ignore", divide="ignore"):
                    y = scale * np.asarray(s[key], float) / np.asarray(s["dvalid"], float)
                ax.plot(hours, y, color=col, lw=2, label=lab)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(True, color=GRID, lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7, colors=INK_MUTED)
            ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8, color=INK)
            if r == 0:
                ax.set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
            else:
                ax.set_xlabel("hour (UTC)", fontsize=8, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Diurnal cycle — EPICC vs EURADCLIM, masked, {2 * k} km, {tag} "
                 f"{cfg.syear}-{cfg.eyear}", fontsize=11, color=INK,
                 fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{cfg.path_rad_figs}/radar_model_diurnal_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()

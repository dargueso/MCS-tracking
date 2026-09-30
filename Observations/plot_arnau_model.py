#!/usr/bin/env python
"""
EPICC against AEMET's validated daily record (Arnau): daily totals and the
maxima over 10 min to 12 h, like for like.

Arnau is AEMET's quality-controlled climatological record: per station and
UTC day, the total P24 and the maximum rain in sliding windows of 10, 20 and
30 min and 1, 2, 6 and 12 h inside the day. The model values are the same
quantities built from its 10-min rain at the station's cell
(extract_model_arnau.py). No diurnal information, but the whole distribution,
from the daily total down to the 10-min burst, and 403 stations (the 10-min
gauge set has 291).

Sample: days whose two AEMET quality flags are both validated (0 manual,
1 automatic), for each variable where Arnau has a value. The model is taken on
exactly those station-days. As a check the p99 ratios are repeated on the days
Arnau derives from the 10-min record (ID_FLAG_P = 1).

Uncertainty: paired year-block bootstrap (the same resampled years for both).

Outputs, in {cfg.path_arnau_figs}:
    arnau_model_qq_<tag>.png          Q-Q per region: P24, PMAX60, PMAX10
    arnau_model_durations_<tag>.png   model/Arnau ratio against window length
    arnau_model_biasmaps_<tag>.png    per-station mean and P99 ratios
    arnau_model_seasonal.png          monthly mean P24 and P99 (always all months)
    arnau_model_<tag>_numbers.txt     the numbers, for quoting
    arnau_model_<tag>.csv             every quantile, ratio and interval

    python plot_arnau_model.py                # ASON
    python plot_arnau_model.py --all-months
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
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

import station_config as cfg
import radar_utils as ru
import seasons
from plot_obs_model_comparison import INK, INK_MUTED, GRID
from plot_obs_model_maps_relative import DIV
from plot_station_model import C_OBS, C_MOD, region_of, station_map

MOD_FILE = f"{cfg.path_eval}/AT_ARNAU_{cfg.model_run}_DAILY_2011-2019.nc"
VARS = {"p24": 24 * 60, "pmax10": 10, "pmax20": 20, "pmax30": 30, "pmax60": 60,
        "pmax2h": 120, "pmax6h": 360, "pmax12h": 720}          # window, minutes
LABEL = {"p24": "P24", "pmax10": "PMAX10", "pmax20": "PMAX20", "pmax30": "PMAX30",
         "pmax60": "PMAX60", "pmax2h": "PMAX2h", "pmax6h": "PMAX6h", "pmax12h": "PMAX12h"}
PROBS = np.array([0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
LABELLED = {0.9: "p90", 0.99: "p99", 0.999: "p99.9"}
EDGES = np.geomspace(0.1, 2000.0, 2001)     # mm; below 0.1 mm counts as dry
WET = 1.0                                   # mm, wet day
MIN_BOOT_YEARS = 3
VARIANTS = {"cell": "EPICC cell", "nmax": "EPICC 3x3 max"}
TICK = {10: "10m", 20: "20m", 30: "30m", 60: "1h", 120: "2h", 360: "6h", 720: "12h", 1440: "24h"}


###########################################################
# Data
###########################################################

def load(months, derived_only=False):
    """Arnau and model per station-day, restricted to the months and to validated days."""
    with xr.open_dataset(cfg.file_arnau) as a:
        codes = [str(c) for c in a.code.values]
        days = pd.to_datetime(a.day.values)
        lat, lon = a.lat.values, a.lon.values
        q1, q2, fp = a.id_flag_q1.values, a.id_flag_q2.values, a.id_flag_p.values
        obs = {v: a[v].values.astype("float64") for v in VARS}
    with xr.open_dataset(MOD_FILE) as m:
        assert [str(c) for c in m.code.values] == codes
        assert np.array_equal(pd.to_datetime(m.day.values), days)
        mod = {k: {v: m[f"{v}_{k}"].values.astype("float64") for v in VARS} for k in VARIANTS}
    good = np.isin(q1, cfg.arnau_good_flags) & np.isin(q2, cfg.arnau_good_flags)
    if derived_only:
        good &= fp == 1
    sel = np.isin(days.month, months)
    valid = {v: good[:, sel] & np.isfinite(obs[v][:, sel]) for v in VARS}
    return dict(obs={v: x[:, sel] for v, x in obs.items()},
                mod={k: {v: x[:, sel] for v, x in d.items()} for k, d in mod.items()},
                valid=valid, days=days[sel], lat=lat, lon=lon, codes=codes)


###########################################################
# Quantiles with a year-block bootstrap
###########################################################

def per_year(x, valid, years, rmask):
    """{year: (hist, nvalid)} of x over the valid station-days of a region."""
    out = {}
    for y in np.unique(years):
        v = x[rmask][:, years == y][valid[rmask][:, years == y]]
        out[y] = (np.histogram(v, EDGES)[0], v.size)
    return out


def quantiles(py, years):
    hist = sum(py[y][0] for y in years)
    n = sum(py[y][1] for y in years)
    if n == 0:
        return np.full(PROBS.size, np.nan)
    return ru.quantile_from_hist(hist, n - hist.sum(), PROBS, EDGES)


def bootstrap(po, pm, nboot, seed=0):
    years = sorted(set(po) & set(pm))
    if nboot <= 0 or len(years) < MIN_BOOT_YEARS:
        return (np.full((2, PROBS.size), np.nan),) * 2
    rng = np.random.default_rng(seed)
    qo, qm = np.empty((nboot, PROBS.size)), np.empty((nboot, PROBS.size))
    for b in range(nboot):
        pick = rng.choice(years, size=len(years), replace=True)
        qo[b], qm[b] = quantiles(po, pick), quantiles(pm, pick)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(qo > 0, qm / qo, np.nan)
    ci = lambda z: np.nanpercentile(z, [2.5, 97.5], axis=0)
    return ci(qm), ci(ratio)


def quantile_table(d, regions, nboot):
    years = d["days"].year.values
    rows = []
    for reg, rm in regions.items():
        for v in VARS:
            po = per_year(d["obs"][v], d["valid"][v], years, rm)
            q_obs = quantiles(po, list(po))
            for k in VARIANTS:
                pm = per_year(d["mod"][k][v], d["valid"][v], years, rm)
                q_mod = quantiles(pm, list(pm))
                ci_m, ci_r = bootstrap(po, pm, nboot)
                for p, a, b, mlo, mhi, rlo, rhi in zip(PROBS, q_obs, q_mod, *ci_m, *ci_r):
                    rows.append(dict(region=reg, var=v, variant=k, prob=p, arnau=a, model=b,
                                     model_lo=mlo, model_hi=mhi,
                                     ratio=b / a if a > 0 else np.nan, ratio_lo=rlo, ratio_hi=rhi))
    return pd.DataFrame(rows)


def season_maxima(d, regions):
    """Median model/Arnau ratio of station-season maxima (station-seasons >= 80% valid)."""
    years = d["days"].year.values
    rows = []
    for v in VARS:
        o, val = d["obs"][v], d["valid"][v]
        for y in np.unique(years):
            s = years == y
            cover = val[:, s].mean(1)
            ob = np.where(val[:, s], o[:, s], -np.inf).max(1)
            for k in VARIANTS:
                md = np.where(val[:, s], d["mod"][k][v][:, s], -np.inf).max(1)
                ok = (cover >= 0.8) & (ob > 0)
                for st in np.where(ok)[0]:
                    rows.append(dict(station=d["codes"][st], year=y, var=v, variant=k,
                                     arnau=ob[st], model=md[st], ratio=md[st] / ob[st],
                                     **{f"in_{r}": bool(m[st]) for r, m in regions.items()}))
    return pd.DataFrame(rows)


###########################################################
# Figures
###########################################################

def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def figure_qq(q, regions, tag):
    rows = ["p24", "pmax60", "pmax10"]
    fig, axes = plt.subplots(len(rows), len(regions), figsize=(3.2 * len(regions), 3.1 * len(rows)),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for r, v in enumerate(rows):
        for c, reg in enumerate(regions):
            ax = axes[r, c]
            for k, ls, mk in (("cell", "-", "o"), ("nmax", "--", "s")):
                g = q[(q.region == reg) & (q["var"] == v) & (q.variant == k) & (q.prob >= 0.9)]
                ax.errorbar(g.arnau, g.model, yerr=[g.model - g.model_lo, g.model_hi - g.model],
                            color=C_MOD, ls=ls, marker=mk, ms=3.5, lw=1.3, elinewidth=0.8,
                            alpha=1 if k == "cell" else 0.55, label=VARIANTS[k])
                for _, row in g[g.prob.isin(list(LABELLED))].iterrows():
                    if k == "cell":
                        ax.annotate(LABELLED[row.prob], (row.arnau, row.model), fontsize=6.5,
                                    color=INK_MUTED, xytext=(4, -9), textcoords="offset points")
            g = q[(q.region == reg) & (q["var"] == v) & (q.prob >= 0.9)]
            lo = np.nanmin(g[["arnau", "model"]].values) * 0.8
            hi = np.nanmax(g[["arnau", "model_hi"]].values) * 1.2
            ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=1, label="1:1 (Arnau)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{reg} · {LABEL[v]}", loc="left", fontsize=9, color=INK, fontweight="bold")
            ax.set_xlabel("Arnau (mm)", fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel("EPICC (mm)", fontsize=7.5, color=INK)
            style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"EPICC vs Arnau (AEMET validated daily record), {tag} 2011-2019 — "
                 f"all validated days, p90 … p99.9; bars: 95% year-block bootstrap",
                 fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{cfg.path_arnau_figs}/arnau_model_qq_{tag}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_durations(q, smax, regions, tag):
    order = sorted(VARS, key=VARS.get)                      # by window length, 10 min .. 24 h
    dur = np.array([VARS[v] for v in order])
    fig, axes = plt.subplots(3, len(regions), figsize=(3.2 * len(regions), 8.4), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for c, reg in enumerate(regions):
        for r, p in enumerate((0.99, 0.999)):
            ax = axes[r, c]
            for k, ls in (("cell", "-"), ("nmax", "--")):
                g = q[(q.region == reg) & (q.variant == k) & np.isclose(q.prob, p)].set_index("var").loc[order]
                ax.fill_between(dur, g.ratio_lo, g.ratio_hi, color=C_MOD, alpha=0.18 if k == "cell" else 0.08, lw=0)
                ax.plot(dur, g.ratio, color=C_MOD, ls=ls, marker="o", ms=3, lw=1.5,
                        alpha=1 if k == "cell" else 0.6, label=VARIANTS[k])
            ax.set_title(f"{reg} · {LABELLED[p]} ratio", loc="left", fontsize=9, color=INK, fontweight="bold")
        ax = axes[2, c]
        for k, ls in (("cell", "-"), ("nmax", "--")):
            g = smax[(smax.variant == k) & smax[f"in_{reg}"]].groupby("var").ratio
            med = g.median().loc[order]
            lo, hi = g.quantile(0.25).loc[order], g.quantile(0.75).loc[order]
            ax.fill_between(dur, lo, hi, color=C_MOD, alpha=0.12 if k == "cell" else 0.06, lw=0)
            ax.plot(dur, med, color=C_MOD, ls=ls, marker="o", ms=3, lw=1.5, alpha=1 if k == "cell" else 0.6)
        ax.set_title(f"{reg} · seasonal max ratio (median, IQR)", loc="left", fontsize=9, color=INK,
                     fontweight="bold")
        ax.set_xlabel("window", fontsize=7.5, color=INK)
        for r in range(3):
            a = axes[r, c]
            a.axhline(1, color=C_OBS, lw=1)
            a.set_xscale("log")
            a.set_xticks(dur)
            a.set_xticklabels([TICK[x] for x in dur], fontsize=6.5)
            a.minorticks_off()
            style(a)
        axes[0, c].set_ylim(0.4, 2.5); axes[1, c].set_ylim(0.4, 2.5); axes[2, c].set_ylim(0.2, 3)
    axes[0, 0].set_ylabel("EPICC / Arnau", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("EPICC / Arnau", fontsize=7.5, color=INK)
    axes[2, 0].set_ylabel("EPICC / Arnau", fontsize=7.5, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(1, 0.975))
    fig.suptitle(f"EPICC / Arnau against accumulation window, {tag} 2011-2019\n"
                 f"shading: 95% year-block bootstrap (rows 1-2), IQR over station-seasons (row 3)",
                 fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{cfg.path_arnau_figs}/arnau_model_durations_{tag}.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def station_quantity(x, valid, what, min_days):
    x = np.where(valid, x, np.nan)
    n = valid.sum(1)
    with np.errstate(invalid="ignore"):
        if what == "mean":
            v = np.nanmean(x, 1)
        else:
            v = np.nanquantile(x, what, axis=1)
    return np.where(n >= min_days, v, np.nan)


def figure_biasmaps(d, regions, tag):
    rows = [("mean P24 (mm/day)", "p24", "mean", 300), ("P99 of P24 (mm)", "p24", 0.99, 1000),
            ("P99 of PMAX60 (mm)", "pmax60", 0.99, 1000)]
    fig = plt.figure(figsize=(9.6, 3.2 * len(rows)))
    fig.patch.set_facecolor("#fcfcfb")
    lines = [f"\nStation biases, EPICC cell / Arnau, {tag}: median over stations [share > 1]"]
    for r, (label, v, what, nmin) in enumerate(rows):
        o = station_quantity(d["obs"][v], d["valid"][v], what, nmin)
        m = station_quantity(d["mod"]["cell"][v], d["valid"][v], what, nmin)
        ratio = m / o
        ok = np.isfinite(o) & (o > 0) & regions["ALL"]
        ax = fig.add_subplot(len(rows), 2, 2 * r + 1, projection=ccrs.PlateCarree())
        sc = ax.scatter(d["lon"][ok], d["lat"][ok], c=o[ok], cmap="viridis", s=14,
                        edgecolor="#fcfcfb", linewidth=0.3, transform=ccrs.PlateCarree())
        station_map(ax, f"Arnau · {label}")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
        ok &= np.isfinite(ratio) & (ratio > 0)
        ax = fig.add_subplot(len(rows), 2, 2 * r + 2, projection=ccrs.PlateCarree())
        sc = ax.scatter(d["lon"][ok], d["lat"][ok], c=np.log2(ratio[ok]), cmap=DIV.reversed(),
                        norm=TwoSlopeNorm(0, -1.5, 1.5), s=14, edgecolor="#fcfcfb", linewidth=0.3,
                        transform=ccrs.PlateCarree())
        station_map(ax, f"EPICC cell / Arnau · {label.split(' (')[0]}")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, extend="both")
        cb.set_ticks(np.log2([1 / 2.5, 1 / 1.5, 1, 1.5, 2.5]))
        cb.set_ticklabels(["0.4", "0.67", "1", "1.5", "2.5"])
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
        parts = [f"{reg} {np.median(ratio[ok & rm]):.2f} [{np.mean(ratio[ok & rm] > 1):.0%}]"
                 for reg, rm in regions.items() if (ok & rm).sum() >= 3]
        lines.append(f"  {label.split(' (')[0]:14s} " + "  ".join(parts) + f"   ({ok.sum()} stations)")
    fig.suptitle(f"Station biases against Arnau, {tag} 2011-2019 (ratios on a log scale)",
                 fontsize=11, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{cfg.path_arnau_figs}/arnau_model_biasmaps_{tag}.png", dpi=180,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return lines


def figure_seasonal(regions):
    d = load(list(range(1, 13)))
    mon = d["days"].month.values
    rows = [("mean P24 (mm/day)", "p24", "mean"), ("P99 of P24 (mm)", "p24", 0.99),
            ("P99 of PMAX60 (mm)", "pmax60", 0.99)]
    fig, axes = plt.subplots(len(rows), len(regions), figsize=(3.2 * len(regions), 7.6), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    table = []
    for c, (reg, rm) in enumerate(regions.items()):
        for r, (label, v, what) in enumerate(rows):
            vals = {}
            for name, x in (("Arnau", d["obs"][v]), ("EPICC cell", d["mod"]["cell"][v])):
                prof = []
                for mth in range(1, 13):
                    s = mon == mth
                    y = x[rm][:, s][d["valid"][v][rm][:, s]]
                    prof.append(y.mean() if what == "mean" else np.quantile(y, what))
                vals[name] = prof
                axes[r, c].plot(range(1, 13), prof, color=C_OBS if name == "Arnau" else C_MOD,
                                lw=2, marker="o", ms=3, label=name)
            if reg == "ALL":
                table.append((label, vals))
            axes[r, c].set_title(f"{reg} · {label.split(' (')[0]}", loc="left", fontsize=9, color=INK,
                                 fontweight="bold")
            axes[r, c].set_ylim(bottom=0)
            style(axes[r, c])
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=7.5, color=INK)
        axes[-1, c].set_xticks(range(1, 13))
        axes[-1, c].set_xticklabels("JFMAMJJASOND", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Seasonal cycle, EPICC vs Arnau, 2011-2019 (validated days)", fontsize=11, color=INK,
                 fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{cfg.path_arnau_figs}/arnau_model_seasonal.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    lines = ["\nSeasonal cycle, ALL, by month (J..D):"]
    for label, vals in table:
        for name, prof in vals.items():
            lines.append(f"  {label.split(' (')[0]:14s} {name:10s} " + " ".join(f"{p:5.2f}" for p in prof))
    return lines


###########################################################

def fmt_ratio(g, p):
    r = g[np.isclose(g.prob, p)].iloc[0]
    return f"{LABELLED[p]} {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]"


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--nboot", type=int, default=1000)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(cfg.path_arnau_figs, exist_ok=True)
    months, tag = seasons.resolve(args)
    d = load(months)
    regions = region_of(d["lat"], d["lon"])
    nst = {reg: int((rm & d["valid"]["p24"].any(1)).sum()) for reg, rm in regions.items()}
    lines = [f"EPICC vs Arnau (AEMET validated daily record), {tag}, 2011-2019",
             f"validated station-days (P24): {int(d['valid']['p24'].sum()):,}; stations "
             + ", ".join(f"{r} {n}" for r, n in nst.items())]

    logging.info("quantiles")
    q = quantile_table(d, regions, args.nboot)
    q.to_csv(f"{cfg.path_arnau_figs}/arnau_model_{tag}.csv", index=False, float_format="%.4g")
    lines.append("\nQuantiles over all validated days, EPICC/Arnau [95% year-block interval]:")
    for reg in regions:
        for v in VARS:
            for k in VARIANTS:
                g = q[(q.region == reg) & (q["var"] == v) & (q.variant == k)]
                lines.append(f"  {reg:4s} {LABEL[v]:8s} {k:5s}  " +
                             "  ".join(fmt_ratio(g, p) for p in LABELLED))

    lines.append(f"\nMean and wet-day (>= {WET:g} mm) statistics of P24, EPICC cell / Arnau:")
    for reg, rm in regions.items():
        val = d["valid"]["p24"][rm]
        o, m = d["obs"]["p24"][rm][val], d["mod"]["cell"]["p24"][rm][val]
        lines.append(f"  {reg:4s} mean {m.mean() / o.mean():.2f}   wet-day frequency "
                     f"{np.mean(m >= WET) / np.mean(o >= WET):.2f} ({100 * np.mean(o >= WET):.1f}% -> "
                     f"{100 * np.mean(m >= WET):.1f}%)   wet-day mean {m[m >= WET].mean() / o[o >= WET].mean():.2f}")

    logging.info("seasonal maxima")
    smax = season_maxima(d, regions)
    lines.append("\nStation-season maxima, EPICC/Arnau (median over station-seasons):")
    for reg in regions:
        for k in VARIANTS:
            g = smax[(smax.variant == k) & smax[f"in_{reg}"]].groupby("var").ratio.median()
            lines.append(f"  {reg:4s} {k:5s}  " + "  ".join(f"{LABEL[v]} {g[v]:.2f}" for v in VARS))

    dp = load(months, derived_only=True)
    qp = quantile_table(dp, regions, nboot=0)
    lines.append("\nCheck: only days Arnau derives from the 10-min record (ID_FLAG_P = 1), p99, cell:")
    for reg in regions:
        g = qp[(qp.region == reg) & (qp.variant == "cell") & np.isclose(qp.prob, 0.99)].set_index("var")
        lines.append(f"  {reg:4s} " + "  ".join(f"{LABEL[v]} {g.loc[v, 'ratio']:.2f}" for v in VARS))

    figure_qq(q, regions, tag)
    figure_durations(q, smax, regions, tag)
    lines += figure_biasmaps(d, regions, tag)
    lines += figure_seasonal(regions)
    text = "\n".join(lines)
    print(text)
    with open(f"{cfg.path_arnau_figs}/arnau_model_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()

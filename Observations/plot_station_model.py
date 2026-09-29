#!/usr/bin/env python
"""
Evaluate EPICC 2 km hourly rain against the combined AEMET gauge dataset, and
EURADCLIM against the same gauges when it is available.

Four comparisons, each using the part of the combined dataset suited to it:

  1  hourly Q-Q        clock-hour gauge totals (10-min dataset) against the
                       model cell and the neighbourhood maximum, pooled per
                       subregion; year-block bootstrap as in the radar branch
  2  diurnal cycle     mean rain and wet-hour frequency by UTC hour
  3  duration maxima   seasonal maximum d-hour totals, d = 1..24 h, model/gauge
                       ratio per station-season. Both on clock-aligned hourly
                       steps (like for like). The 10-min data also give the
                       factor by which a sliding-window gauge maximum (AEMET's
                       PMAX) exceeds the clock-hour one, reported alongside.
  4  daily paired      the one timescale where an ERA5-driven run should match
                       observed events day by day: correlation and categorical
                       scores per station. Uses the 10-min stations AND the
                       Arnau-only ones (validated daily totals), 456 in all.

Only verified or automatically checked days are used (cfg.eval_tiers), and the
model is only counted in hours where the gauge is valid.

    python plot_station_model.py                   # ASON
    python plot_station_model.py --all-months       # whole year
    python plot_station_model.py --season JJA      # or --months 6 7 8
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
import cartopy.feature as cfeature

import station_config as cfg
import radar_utils as ru
import radar_config as rcfg
from plot_obs_model_comparison import INK, INK_MUTED, GRID, COLORS
from plot_obs_model_maps_relative import DIV
from matplotlib.colors import TwoSlopeNorm, LogNorm
import seasons
from plot_radar_model_qq import (PROBS, LABELLED, quantiles, bootstrap,
                                 MIN_BOOT_YEARS)

C_OBS, C_MOD, C_RAD = COLORS["obs"], COLORS["YS"], COLORS["SB"]   # validated trio
MOD_FILE = f"{cfg.path_eval}/AT_STATIONS_{cfg.model_run}_01H_{cfg.syear}-{cfg.eyear}.nc"
RAD_FILE = f"{cfg.path_eval}/AT_STATIONS_EURADCLIM_01H_{cfg.syear}-{cfg.eyear}.nc"


###########################################################
# Data
###########################################################

def load(months):
    """Hourly gauge, model and (if present) radar at the 10-min stations, season only."""
    with xr.open_dataset(cfg.file_01h) as h:
        obs = h.prec.values
        tier = h.day_tier.values
        codes = [str(c) for c in h.code.values]
        lat, lon = h.lat.values, h.lon.values
        times = pd.to_datetime(h.time.values)
    # day tier -> hour
    ok_day = np.isin(tier, cfg.eval_tiers)
    ok = np.repeat(ok_day, 24, axis=1)
    obs = np.where(ok, obs, np.nan)

    with xr.open_dataset(MOD_FILE) as m:
        idx = [list(m.code.values).index(c) for c in codes]
        mod = {k: m[k].values[idx] for k in ("cell", "nmax", "nmean")}
    rad = None
    if os.path.exists(RAD_FILE):
        with xr.open_dataset(RAD_FILE) as r:
            idx = [list(r.code.values).index(c) for c in codes]
            rad = {k: r[k].values[idx] for k in ("cell", "nmax")}
            jj, ii = r.model_j.values[idx], r.model_i.values[idx]
        # EURADCLIM counts at a station only where its cell passes the radar
        # quality mask (range, availability, shadows, clutter): the same mask
        # as the gridded radar comparison
        if os.path.exists(rcfg.mask_file):
            grid = ru.model_grid()
            with xr.open_dataset(rcfg.mask_file) as mk:
                mask = mk.mask_s1.values.astype(bool)
            jr, ir = jj - grid["ys"].start, ii - grid["xs"].start
            inside = (jr >= 0) & (jr < mask.shape[0]) & (ir >= 0) & (ir < mask.shape[1])
            good = np.zeros(len(codes), bool)
            good[inside] = mask[jr[inside], ir[inside]]
            for k in rad:
                rad[k][~good] = np.nan
            logging.info("EURADCLIM usable at %d of %d stations (radar quality mask)",
                         good.sum(), len(codes))
    sel = np.isin(times.month, months)
    obs = obs[:, sel]
    mod = {k: v[:, sel] for k, v in mod.items()}
    if rad is not None:
        rad = {k: v[:, sel] for k, v in rad.items()}
    return dict(obs=obs, mod=mod, rad=rad, codes=codes, lat=lat, lon=lon,
                times=times[sel])


def region_of(lat, lon):
    return {name: (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
            for name, (la0, lo0, la1, lo1) in cfg.subregions.items()}


def per_year_hist(values, valid, years, regions, edges):
    """{year: {region: {hist, nvalid, ...}}} in the form plot_radar_model_qq expects."""
    out = {}
    for y in np.unique(years):
        sel = years == y
        out[y] = {}
        for name, rmask in regions.items():
            v = values[rmask][:, sel]
            good = valid[rmask][:, sel]
            x = v[good]
            wet = x[x >= edges[0]]
            h, _ = np.histogram(np.clip(wet, edges[0], edges[-1] * 0.999), bins=edges)
            out[y][name] = {"hist": h, "nvalid": int(good.sum()), "total": float(x.sum()),
                            "nwet": int((x >= cfg.wet_thres).sum())}
    return out


###########################################################
# 1 + 2  hourly distribution and diurnal cycle
###########################################################

def hourly(d, regions, tag, nboot, joint_radar):
    edges = ru.hist_edges()
    years = d["times"].year.values
    valid = np.isfinite(d["obs"]) & np.isfinite(d["mod"]["cell"])
    series = {"gauges": d["obs"], "EPICC cell": d["mod"]["cell"],
              "EPICC 3x3 max": d["mod"]["nmax"]}
    if joint_radar:
        valid &= np.isfinite(d["rad"]["cell"])
        series["EURADCLIM cell"] = d["rad"]["cell"]
    per = {k: per_year_hist(v, valid, years, regions, edges) for k, v in series.items()}

    rows = []
    fig, axes = plt.subplots(1, len(regions), figsize=(3.3 * len(regions), 3.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    style = {"EPICC cell": dict(color=C_MOD, mfc=C_MOD),
             "EPICC 3x3 max": dict(color=C_MOD, mfc="#fcfcfb"),
             "EURADCLIM cell": dict(color=C_RAD, mfc=C_RAD)}
    allq = []
    for c, reg in enumerate(regions):
        ax = axes[0, c]
        yrs = sorted(per["gauges"])
        qo = quantiles(per["gauges"], yrs, reg, edges)
        for name in series:
            if name == "gauges":
                continue
            qm = quantiles(per[name], yrs, reg, edges)
            ci_o, ci_m, ci_r = bootstrap(per["gauges"], per[name], reg, edges, nboot)
            ok = (qo > 0) & (qm > 0)
            allq += list(qo[ok]) + list(qm[ok])
            ax.errorbar(qo[ok], qm[ok], yerr=np.abs(ci_m[:, ok] - qm[ok]), fmt="o-", ms=5,
                        lw=1.2, elinewidth=0.8, capsize=0, label=name, mec=style[name]["color"],
                        mew=1.2, **style[name])
            for i, p in enumerate(PROBS):
                rows.append({"region": reg, "series": name, "prob": p, "gauges": qo[i],
                             "value": qm[i], "ratio": qm[i] / qo[i] if qo[i] > 0 else np.nan,
                             "ratio_lo": ci_r[0, i], "ratio_hi": ci_r[1, i]})
        for p, lab in LABELLED.items():
            i = int(np.argmin(np.abs(PROBS - p)))
            if qo[i] > 0:
                ax.axvline(qo[i], color=GRID, lw=0.8, zorder=0)
                ax.text(qo[i], 0.98, lab, transform=ax.get_xaxis_transform(), fontsize=6.5,
                        color=INK_MUTED, ha="center", va="top")
        n = int(regions[reg].sum())
        s = {k: sum(per[k][y][reg]["nwet"] for y in yrs) / max(sum(per[k][y][reg]["nvalid"]
                                                                  for y in yrs), 1)
             for k in series}
        ax.text(0.97, 0.04, "\n".join([f"{n} stations"] + [f"wet {k}: {100 * v:.1f}%"
                                                          for k, v in s.items()]),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK_MUTED)
        ax.set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlabel("gauges (mm/h)", fontsize=8, color=INK)
        if c == 0:
            ax.set_ylabel("gridded product (mm/h)", fontsize=8, color=INK)
    lo, hi = min(allq) / 1.5, max(allq) * 1.5
    for ax in axes[0]:
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, ls="--", lw=0.8, zorder=0)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), fontsize=8, frameon=False)
    sub = "joint with EURADCLIM" if joint_radar else f"{cfg.syear}-{cfg.eyear}"
    fig.suptitle(f"Hourly rain quantiles at AEMET gauges (all hours, p90 … p99.999), {tag}, {sub}",
                 fontsize=11, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    suffix = "_radar" if joint_radar else ""
    fig.savefig(f"{cfg.path_st_figs}/station_model_qq_{tag}{suffix}.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # diurnal cycle on the same sample
    fig, axes = plt.subplots(2, len(regions), figsize=(3.3 * len(regions), 5.0), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    hod = d["times"].hour.values
    colours = {"gauges": C_OBS, "EPICC cell": C_MOD, "EURADCLIM cell": C_RAD}
    for c, (reg, rmask) in enumerate(regions.items()):
        for name, arr in series.items():
            if name not in colours:
                continue
            v, g = arr[rmask], valid[rmask]
            mean = [np.nanmean(v[:, hod == hr][g[:, hod == hr]]) for hr in range(24)]
            freq = [100 * np.mean(v[:, hod == hr][g[:, hod == hr]] >= cfg.wet_thres)
                    for hr in range(24)]
            for r, y in enumerate((mean, freq)):
                axes[r, c].plot(np.arange(24) + 0.5, y, color=colours[name], lw=2, label=name)
        axes[0, c].set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        axes[1, c].set_xlabel("hour (UTC)", fontsize=8, color=INK)
    axes[0, 0].set_ylabel("mean rain (mm/h)", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("wet hours (%)", fontsize=8, color=INK)
    for ax in axes.ravel():
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), fontsize=8, frameon=False)
    fig.suptitle(f"Diurnal cycle at AEMET gauges, {tag}, {sub}", fontsize=11, color=INK,
                 fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{cfg.path_st_figs}/station_model_diurnal_{tag}{suffix}.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return pd.DataFrame(rows)


###########################################################
# 3  duration maxima
###########################################################

def running_max(x, valid, d):
    """Max of d-hour running sums over windows whose hours are all valid. x: (nst, nt)."""
    v = np.where(valid, x, 0.0)
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(v, 1)], 1)
    n = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(valid, 1)], 1)
    s = c[:, d:] - c[:, :-d]
    full = (n[:, d:] - n[:, :-d]) == d
    s = np.where(full, s, -np.inf)
    m = s.max(1)
    return np.where(np.isfinite(m), m, np.nan)


def sliding_factor(months):
    """Median ratio of the sliding 60-min maximum (10-min steps) to the clock-hour one,
    per station-season, from the 10-min dataset."""
    with xr.open_dataset(cfg.file_10min) as ds:
        t = pd.to_datetime(ds.time.values)
        sel = np.isin(t.month, months)
        x = ds.prec.values[:, sel]
        t = t[sel]
    ratios = []
    for y in np.unique(t.year):
        s = t.year == y
        xs = x[:, s]
        valid = np.isfinite(xs)
        slide = running_max(xs, valid, 6)
        n = xs.shape[1] // 6 * 6
        # clock hours: 6 consecutive steps starting on the hour (the axis starts at 00:00)
        hx = xs[:, :n].reshape(xs.shape[0], -1, 6)
        hv = np.isfinite(hx).all(2)
        clock = np.where(hv, np.nansum(hx, 2), -np.inf).max(1)
        good = (valid.mean(1) > 0.8) & (clock >= 5.0)        # a real rain hour, enough data
        ratios += list(slide[good] / clock[good])
    return float(np.median(ratios)), len(ratios)


def durations(d, regions, tag, months):
    years = d["times"].year.values
    valid = np.isfinite(d["obs"]) & np.isfinite(d["mod"]["cell"])
    rows = []
    for y in np.unique(years):
        s = years == y
        cover = valid[:, s].mean(1)
        for dur in cfg.durations_h:
            ob = running_max(d["obs"][:, s], valid[:, s], dur)
            for key in ("cell", "nmax"):
                md = running_max(d["mod"][key][:, s], valid[:, s], dur)
                for ist in np.where((cover >= 0.8) & (ob > 0))[0]:
                    rows.append({"station": d["codes"][ist], "year": y, "duration": dur,
                                 "variant": key, "obs": ob[ist], "model": md[ist],
                                 **{f"in_{r}": bool(m[ist]) for r, m in regions.items()}})
    df = pd.DataFrame(rows)
    df["ratio"] = df.model / df.obs
    fac, nfac = sliding_factor(months)

    fig, axes = plt.subplots(1, len(regions), figsize=(3.3 * len(regions), 3.3), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for c, reg in enumerate(regions):
        ax = axes[0, c]
        sub = df[df[f"in_{reg}"]]
        for key, ls, mfc in (("cell", "-", C_MOD), ("nmax", "--", "#fcfcfb")):
            g = sub[sub.variant == key].groupby("duration").ratio
            med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.fill_between(med.index, q25, q75, color=C_MOD, alpha=0.12 if key == "cell" else 0.06,
                            lw=0)
            ax.plot(med.index, med, ls=ls, marker="o", ms=5, color=C_MOD, mfc=mfc, mec=C_MOD,
                    lw=1.8, label=f"EPICC {'cell' if key == 'cell' else '3x3 max'}")
        ax.axhline(1, color=INK_MUTED, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(cfg.durations_h)
        ax.set_xticklabels([str(x) for x in cfg.durations_h])
        ax.set_title(f"{reg}  ({sub.station.nunique()} stations)", loc="left", fontsize=9.5,
                     color=INK, fontweight="bold")
        ax.set_xlabel("duration (h)", fontsize=8, color=INK)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel("model / gauge seasonal max\n(median, IQR band)", fontsize=8, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(1.0, 0.93))
    fig.suptitle(f"Seasonal maximum d-hour rain, clock-aligned, {tag} {cfg.syear}-{cfg.eyear}",
                 fontsize=11, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.905, f"At the gauges a sliding-window 1-h maximum (AEMET PMAX60) is "
             f"×{fac:.2f} the clock-hour one (median of {nfac} station-seasons)",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(f"{cfg.path_st_figs}/station_model_durations_{tag}.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return df, fac, nfac


###########################################################
# 4  daily paired
###########################################################

def daily(months, tag):
    """Daily totals: 10-min stations (accepted tiers) and Arnau-only stations (validated)."""
    with xr.open_dataset(MOD_FILE) as m:
        mcodes = [str(c) for c in m.code.values]
        mt = pd.to_datetime(m.time.values)
        mod = m.cell.values
        mlat, mlon = m.lat.values, m.lon.values
    nd = mod.shape[1] // 24
    days = mt[::24][:nd]
    mod_d = mod[:, :nd * 24].reshape(len(mcodes), nd, 24).sum(2)

    obs_d = np.full((len(mcodes), nd), np.nan)
    source = np.array(["none"] * len(mcodes), dtype=object)
    with xr.open_dataset(cfg.file_01h) as h:
        hcodes = [str(c) for c in h.code.values]
        x = h.prec.values.reshape(len(hcodes), -1, 24)
        ok = np.isin(h.day_tier.values, cfg.eval_tiers) & np.isfinite(x).all(2)
        hd = np.where(ok, np.nansum(x, 2), np.nan)
    for k, c in enumerate(hcodes):
        i = mcodes.index(c)
        obs_d[i] = hd[k, :nd]
        source[i] = "10-min"
    with xr.open_dataset(cfg.file_arnau) as a:
        acodes = [str(c) for c in a.code.values]
        adays = pd.to_datetime(a.day.values)
        good = np.isin(a.id_flag_q1.values, (0, 1))
        p24 = np.where(good, a.p24.values, np.nan)
    pos = days.get_indexer(adays)
    for k, c in enumerate(acodes):
        i = mcodes.index(c) if c in mcodes else None
        if i is None or source[i] == "10-min":
            continue
        obs_d[i, pos[pos >= 0]] = p24[k, pos >= 0]
        source[i] = "Arnau"

    sel = np.isin(days.month, months)
    rows = []
    for i, c in enumerate(mcodes):
        o, m = obs_d[i, sel], mod_d[i, sel]
        v = np.isfinite(o) & np.isfinite(m)
        if v.sum() < 200:
            continue
        o, m = o[v], m[v]
        r = np.corrcoef(o, m)[0, 1]
        rs = pd.Series(o).corr(pd.Series(m), method="spearman")
        row = {"station": c, "lat": mlat[i], "lon": mlon[i], "source": source[i], "ndays": int(v.sum()),
               "r": r, "r_spearman": rs, "bias_ratio": m.sum() / max(o.sum(), 1e-9)}
        for thr in (1.0, 20.0):
            hit = ((o >= thr) & (m >= thr)).sum()
            miss = ((o >= thr) & (m < thr)).sum()
            fa = ((o < thr) & (m >= thr)).sum()
            n = o.size
            rnd = (hit + miss) * (hit + fa) / n
            row[f"pod_{thr:g}"] = hit / max(hit + miss, 1)
            row[f"far_{thr:g}"] = fa / max(hit + fa, 1)
            row[f"ets_{thr:g}"] = (hit - rnd) / max(hit + miss + fa - rnd, 1e-9)
            row[f"fbias_{thr:g}"] = (hit + fa) / max(hit + miss, 1)
        rows.append(row)
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(13, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax = fig.add_subplot(1, 3, (1, 2), projection=ccrs.PlateCarree())
    ax.set_extent([-5, 5, 36.5, 43.5], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    for src, mk in (("10-min", "o"), ("Arnau", "s")):
        s = df[df.source == src]
        sc = ax.scatter(s.lon, s.lat, c=s.r, cmap="viridis", vmin=0, vmax=1, s=22, marker=mk,
                        edgecolor="#fcfcfb", linewidth=0.5, transform=ccrs.PlateCarree(),
                        label=f"{src} ({len(s)})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("correlation of daily totals", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK_MUTED}
    ax.set_title("a  daily correlation, EPICC cell vs gauge", loc="left", fontsize=9.5,
                 color=INK, fontweight="bold")
    ax2 = fig.add_subplot(1, 3, 3)
    scores = ["r", "pod_1", "far_1", "ets_1", "pod_20", "far_20", "ets_20"]
    ax2.boxplot([df[s].dropna() for s in scores], vert=False, widths=0.55, showfliers=False,
                medianprops=dict(color=C_MOD, lw=2), boxprops=dict(color=INK_MUTED),
                whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
    ax2.set_yticklabels(["r", "POD ≥1", "FAR ≥1", "ETS ≥1", "POD ≥20", "FAR ≥20", "ETS ≥20"],
                        fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", color=GRID, lw=0.5)
    ax2.tick_params(labelsize=7, colors=INK_MUTED)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_title(f"b  scores across {len(df)} stations", loc="left", fontsize=9.5, color=INK,
                  fontweight="bold")
    fig.suptitle(f"Paired daily rain (00-24 UTC), EPICC vs AEMET gauges, {tag}",
                 fontsize=11, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{cfg.path_st_figs}/station_model_daily_{tag}.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return df


###########################################################
# 5  bias maps, 6  seasonal cycle: gauges, EPICC and EURADCLIM, three ways
###########################################################

def joint_valid(d):
    """Hours valid in all datasets being compared: gauge and model, plus radar if present."""
    v = np.isfinite(d["obs"]) & np.isfinite(d["mod"]["cell"])
    if has_radar(d):
        v &= np.isfinite(d["rad"]["cell"])
    return v


def has_radar(d):
    return d["rad"] is not None and np.isfinite(d["rad"]["cell"]).any()


def station_stats(x, valid, q_list, min_hours):
    """Per station: mean (mm/day) and all-hour quantiles over the valid hours."""
    x = np.where(valid, x, np.nan)
    n = valid.sum(1)
    with np.errstate(invalid="ignore"):
        out = {"mean": np.where(n >= min_hours["mean"], np.nanmean(x, 1) * 24, np.nan)}
        for q in q_list:
            v = np.nanquantile(x, q, axis=1)
            out[q] = np.where(n >= 10 / (1 - q), v, np.nan)     # >= 10 exceedances
    return out


def pairs(d):
    """(label, numerator, denominator) of the comparisons that the data allow."""
    p = [("EPICC / gauges", "mod", "obs")]
    if has_radar(d):
        p += [("EURADCLIM / gauges", "rad", "obs"), ("EPICC / EURADCLIM", "mod", "rad")]
    return p


def bias_maps(d, regions, tag):
    valid = joint_valid(d)
    qs = [0.99, 0.999]
    series = {"obs": d["obs"], "mod": d["mod"]["cell"]}
    if has_radar(d):
        series["rad"] = d["rad"]["cell"]
    st = {k: station_stats(v, valid, qs, {"mean": 2000}) for k, v in series.items()}
    rows = [("mean", "mean rain (mm/day)", LogNorm(0.5, 5)),
            (0.99, "all-hour P99 (mm/h)", LogNorm(0.5, 5)),
            (0.999, "all-hour P99.9 (mm/h)", LogNorm(2, 30))]
    comps = pairs(d)
    ncol = 1 + len(comps)
    fig = plt.figure(figsize=(4.1 * ncol, 3.2 * len(rows)))
    fig.patch.set_facecolor("#fcfcfb")
    lines = [f"\nStation bias maps ({tag}, joint sample: "
             f"{'gauge + EPICC + EURADCLIM' if has_radar(d) else 'gauge + EPICC'}); "
             "median ratio over stations [fraction of stations > 1]:"]
    lat, lon = d["lat"], d["lon"]
    for r, (key, label, norm) in enumerate(rows):
        ax = fig.add_subplot(len(rows), ncol, r * ncol + 1, projection=ccrs.PlateCarree())
        v = st["obs"][key]
        ok = np.isfinite(v)
        sc = ax.scatter(lon[ok], lat[ok], c=np.maximum(v[ok], norm.vmin), cmap="viridis",
                        norm=norm, s=16, edgecolor="#fcfcfb", linewidth=0.4,
                        transform=ccrs.PlateCarree())
        station_map(ax, f"gauges · {label}")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, extend="both")
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
        for c, (name, num, den) in enumerate(comps):
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = st[num][key] / st[den][key]
            ok = np.isfinite(ratio) & (ratio > 0)
            ax = fig.add_subplot(len(rows), ncol, r * ncol + 2 + c, projection=ccrs.PlateCarree())
            sc = ax.scatter(lon[ok], lat[ok], c=np.log2(ratio[ok]), cmap=DIV.reversed(),
                            norm=TwoSlopeNorm(0, -1.5, 1.5), s=16, edgecolor="#fcfcfb",
                            linewidth=0.4, transform=ccrs.PlateCarree())
            station_map(ax, f"{name} · {label.split(' (')[0]}")
            cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, extend="both")
            cb.set_ticks(np.log2([1 / 2.5, 1 / 1.5, 1, 1.5, 2.5]))
            cb.set_ticklabels(["0.4", "0.67", "1", "1.5", "2.5"])
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
            parts = []
            for reg, rm in regions.items():
                sel = ok & rm
                if sel.sum() >= 3:
                    parts.append(f"{reg} {np.median(ratio[sel]):.2f} [{np.mean(ratio[sel] > 1):.0%}]")
            lines.append(f"  {label.split(' (')[0]:14s} {name:19s} " + "  ".join(parts))
    fig.suptitle(f"Station biases, {tag} — ratios on a log scale, joint sample of "
                 f"{'gauges, EPICC and EURADCLIM' if has_radar(d) else 'gauges and EPICC'}",
                 fontsize=11, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{cfg.path_st_figs}/station_model_biasmaps_{tag}.png", dpi=180,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return lines


def station_map(ax, title):
    ax.set_extent([-5, 5, 36.5, 43.6], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5, "color": INK_MUTED}
    ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold")


def seasonal_cycle(regions):
    """Monthly mean rain, wet-hour frequency and hourly P99, all months, per region.

    Always the whole year, whatever --season says: a seasonal cycle needs every
    month. Joint sample of all available datasets, so the three comparisons
    (EPICC-gauges, EURADCLIM-gauges, EPICC-EURADCLIM) are over the same hours.
    """
    d = load(list(range(1, 13)))
    valid = joint_valid(d)
    mon = d["times"].month.values
    series = [("gauges", d["obs"], C_OBS), ("EPICC cell", d["mod"]["cell"], C_MOD)]
    if has_radar(d):
        series.append(("EURADCLIM cell", d["rad"]["cell"], C_RAD))
    rows = [("mean rain (mm/day)", lambda x: np.mean(x) * 24),
            ("wet hours (%)", lambda x: 100 * np.mean(x >= cfg.wet_thres)),
            ("hourly P99 (mm/h)", lambda x: np.quantile(x, 0.99))]
    fig, axes = plt.subplots(len(rows), len(regions), figsize=(3.3 * len(regions), 7.2),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    table = []
    for c, (reg, rm) in enumerate(regions.items()):
        for name, x, col in series:
            for r, (label, fn) in enumerate(rows):
                y = []
                for m in range(1, 13):
                    sel = x[rm][:, mon == m][valid[rm][:, mon == m]]
                    y.append(fn(sel) if sel.size else np.nan)
                    if r == 0:
                        table.append({"region": reg, "series": name, "month": m,
                                      "mean_mm_day": y[-1]})
                axes[r, c].plot(range(1, 13), y, color=col, lw=2, marker="o", ms=3.5, label=name)
        axes[0, c].set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    for r, (label, _) in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=8, color=INK)
    for ax in axes.ravel():
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels("JFMAMJJASOND")
        ax.set_ylim(bottom=0)
        ax.axvspan(7.5, 11.5, color=GRID, alpha=0.35, lw=0, zorder=0)   # ASON
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), fontsize=8, frameon=False)
    period = "2013-2020 (joint with EURADCLIM)" if has_radar(d) else f"{cfg.syear}-{cfg.eyear}"
    fig.suptitle(f"Seasonal cycle at AEMET gauges, {period}; shaded: ASON", fontsize=11,
                 color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{cfg.path_st_figs}/station_model_seasonal.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    t = pd.DataFrame(table).pivot_table(index=["region", "month"], columns="series",
                                        values="mean_mm_day")
    t.to_csv(f"{cfg.path_eval}/station_model_seasonal.csv", float_format="%.4g")
    allr = t.loc["ALL"]
    lines = ["\nSeasonal cycle, ALL, mean rain (mm/day) by month:"]
    for name in allr.columns:
        lines.append(f"  {name:15s} " + " ".join(f"{v:5.2f}" for v in allr[name].values))
    return lines


###########################################################

def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--nboot", type=int, default=1000)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        level=logging.INFO)
    os.makedirs(cfg.path_eval, exist_ok=True)
    months, tag = seasons.resolve(args)
    d = load(months)
    regions = region_of(d["lat"], d["lon"])

    lines = [f"EPICC vs AEMET gauges, {tag}, {cfg.syear}-{cfg.eyear}, tiers {cfg.eval_tiers}"]
    q = hourly(d, regions, tag, args.nboot, joint_radar=False)
    q.to_csv(f"{cfg.path_eval}/station_model_qq_{tag}.csv", index=False, float_format="%.4g")
    lines.append("\nHourly quantiles, model/gauge ratio [95% year-block interval]:")
    for (reg, name), g in q.groupby(["region", "series"], sort=False):
        parts = [f"{LABELLED[p]} {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]"
                 for p in LABELLED for _, r in g[np.isclose(g.prob, p)].iterrows()]
        lines.append(f"  {reg:4s} {name:15s} " + "  ".join(parts))
    if d["rad"] is not None and np.isfinite(d["rad"]["cell"]).any():
        qr = hourly(d, regions, tag, args.nboot, joint_radar=True)
        qr.to_csv(f"{cfg.path_eval}/station_model_qq_{tag}_radar.csv", index=False,
                  float_format="%.4g")
        lines.append("\nJoint with EURADCLIM (radar-valid hours only):")
        for (reg, name), g in qr.groupby(["region", "series"], sort=False):
            parts = [f"{LABELLED[p]} {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]"
                     for p in LABELLED for _, r in g[np.isclose(g.prob, p)].iterrows()]
            lines.append(f"  {reg:4s} {name:15s} " + "  ".join(parts))

    dur, fac, nfac = durations(d, regions, tag, months)
    dur.to_csv(f"{cfg.path_eval}/station_model_durations_{tag}.csv", index=False,
               float_format="%.4g")
    lines.append(f"\nSeasonal maxima, model/gauge (median over station-seasons), ALL:")
    for key in ("cell", "nmax"):
        g = dur[(dur.variant == key) & dur.in_ALL].groupby("duration").ratio.median()
        lines.append(f"  {key:5s} " + "  ".join(f"{k}h {v:.2f}" for k, v in g.items()))
    lines.append(f"  sliding/clock-hour 1-h maximum at the gauges: x{fac:.2f} "
                 f"(median of {nfac} station-seasons with a >= 5 mm hour)")

    day = daily(months, tag)
    day.to_csv(f"{cfg.path_eval}/station_model_daily_{tag}.csv", index=False, float_format="%.4g")
    lines.append(f"\nPaired daily, {len(day)} stations "
                 f"({(day.source == '10-min').sum()} 10-min, {(day.source == 'Arnau').sum()} Arnau-only):")
    for s in ("r", "r_spearman", "bias_ratio", "pod_1", "far_1", "ets_1", "pod_20", "far_20",
              "ets_20", "fbias_20"):
        lines.append(f"  {s:11s} median {day[s].median():.2f}  IQR "
                     f"{day[s].quantile(.25):.2f}-{day[s].quantile(.75):.2f}")
    lines += bias_maps(d, regions, tag)
    lines += seasonal_cycle(regions)
    text = "\n".join(lines)
    print(text)
    with open(f"{cfg.path_eval}/station_model_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()

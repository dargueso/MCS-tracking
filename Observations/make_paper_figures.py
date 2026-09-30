#!/usr/bin/env python
"""
The evaluation figures for the paper, assembled from the evaluation outputs.

Four figures, each for one region (ALL for the main text, CAT/VAL/BAL/MUR/AND
for the supplement), all for one season (ASON by default):

    1  scales   model/observed ratio against accumulation window, 10 min to
                24 h (AEMET daily record), with the hourly gauges (AEMET/HyMEX)
                and EURADCLIM at 1 h; station-season maxima of P24 and PMAX60
    2  where    maps of the hourly P99.9 and mean-rain ratio to EURADCLIM at
                2 km, the gauge ratios on top; and the two references against
                each other
    3  when     diurnal cycle of rain at the gauges (gauges, EURADCLIM, EPICC),
                of storm hours (observed vs model storms, like for like), of
                cold cloud tops (MERGIR vs the model Tb), the seasonal cycle,
                and the afternoon excess by station altitude
    4  storms   tracked storms against IMERG + MERGIR at 0.1 deg: seasonal
                cycle, distributions, relative track density (maps for ALL)

Names in the figures: "AEMET" is AEMET's validated daily record (Arnau);
"AEMET/HyMEX" is the combined 10-min gauge dataset at clock hours.

Inputs are the CSV/netCDF outputs of the evaluation scripts (run them first:
plot_arnau_model, plot_annual_maxima, plot_station_model, plot_radar_model_qq,
radar_model_stats, extract_storms_at_stations, plot_tb_evaluation, and the
trackings). Outputs go to {ocfg.path_figs}/paper/.

    python make_paper_figures.py                         # ALL, ASON, all four
    python make_paper_figures.py --regions CAT VAL BAL MUR AND --figs 1 3
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

import station_config as cfg
import radar_config as rcfg
import obs_config as ocfg
import radar_utils as ru
import regions
import seasons
from plot_obs_model_comparison import INK, INK_MUTED, GRID, COLORS, tail_ratio_ci
from plot_obs_model_maps_relative import DIV
from plot_obs_model_maps import load_tracks, density, add_map
from plot_obs_model_maps_relative import relative, SEQ
from plot_radar_model_maps import fields as radar_fields
from plot_annual_maxima import gumbel_axis, ranked_ratio
import plot_station_model as psm
import plot_storm_rain_stations as psr

OUT = f"{ocfg.path_figs}/paper"
FIG = ocfg.path_figs
C_OBS, C_MOD, C_RAD, C_SB = COLORS["obs"], COLORS["YS"], COLORS["SB"], "#8a8a86"
BG = "#fcfcfb"
WIN = {"pmax10": 10, "pmax20": 20, "pmax30": 30, "pmax60": 60, "pmax2h": 120, "pmax6h": 360,
       "pmax12h": 720, "p24": 1440}
TICK = {10: "10 min", 20: "20", 30: "30", 60: "1 h", 120: "2", 360: "6", 720: "12", 1440: "24 h"}
BOX = cfg.subregions["ALL"]


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def title(ax, text, size=9.5):
    ax.set_title(text, loc="left", fontsize=size, color=INK, fontweight="bold")


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name, region, tag, lines):
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/{name}_{region}_{tag}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    with open(out.replace(".png", "_numbers.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logging.info("wrote %s", out)


def region_extent(region, pad=0.4):
    if region == "ALL":
        return [-4.6, 5.0, 36.2, 43.8]
    x0, y0, x1, y1 = regions.polygons()[region].bounds
    x0 = max(x0, BOX[1])
    return [x0 - pad, x1 + pad, y0 - pad, y1 + pad]


def base_map(ax, extent):
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    regions.outline(ax, lw=0.6, color="#6a6a66")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID, alpha=0.8)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5, "color": INK_MUTED}


def inset_cbar(fig, mesh, ax, label, extend="both", ticks=None, labels=None):
    cax = ax.inset_axes([1.02, 0.08, 0.03, 0.84])
    cb = fig.colorbar(mesh, cax=cax, extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks); cb.set_ticklabels(labels)
    cb.set_label(label, fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
    return cb


def ratio_colorbar(fig, mesh, ax, label):
    cb = inset_cbar(fig, mesh, ax, label)
    cb.set_ticks(np.log2([0.5, 0.67, 1, 1.5, 2]))
    cb.set_ticklabels(["0.5", "0.67", "1", "1.5", "2"])
    cb.set_label(label, fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)


RNORM = TwoSlopeNorm(0, -1.0, 1.0)      # log2 ratio, 0.5 .. 2


###########################################################
# Figure 1: intensity across scales
###########################################################

def fig1_scales(tag, region, months):
    arn = pd.read_csv(f"{FIG}/arnau/arnau_model_{tag}.csv")
    stn = pd.read_csv(f"{FIG}/stations/station_model_qq_{tag}.csv")
    rad = pd.read_csv(f"{FIG}/radar/radar_model_qq_{tag}.csv")
    mx = pd.read_csv(f"{FIG}/arnau/arnau_model_maxima_{tag}.csv")
    order = sorted(WIN, key=WIN.get)
    dur = np.array([WIN[v] for v in order])
    with __import__("xarray").open_dataset(cfg.file_01h) as h:
        n_hourly = int(regions.masks(h.lat.values, h.lon.values, box=BOX)[region].sum())
    lines = [f"Figure 1, {region}, {tag}: EPICC / observed ratio against accumulation window; "
             f"{n_hourly} hourly gauges in the region"]

    fig = new_fig(9.6, 6.6)
    axes = fig.subplots(2, 2)
    for ax, p, lab in ((axes[0, 0], 0.99, "p99"), (axes[0, 1], 0.999, "p99.9")):
        for k, ls, alpha in (("cell", "-", 1.0), ("nmax", "--", 0.55)):
            g = arn[(arn.region == region) & (arn.variant == k) & np.isclose(arn.prob, p)].set_index("var")
            if g.empty:
                continue
            g = g.reindex(order)
            if k == "cell":
                ax.fill_between(dur, g.ratio_lo, g.ratio_hi, color=C_MOD, alpha=0.15, lw=0)
            ax.plot(dur, g.ratio, color=C_MOD, ls=ls, marker="o", ms=3.5, lw=1.6, alpha=alpha,
                    label=f"vs AEMET daily record, {'model cell' if k == 'cell' else '3x3 max'}")
            lines.append(f"  {lab} {k:4s} AEMET: " + " ".join(f"{TICK[w].strip()}={r:.2f}" for w, r in zip(dur, g.ratio)))
        # hourly gauges and radar at 1 h
        gs = stn[(stn.region == region) & (stn.series == "EPICC cell") & np.isclose(stn.prob, p)]
        if n_hourly >= 5 and not gs.empty and np.isfinite(gs.ratio.iloc[0]):
            r = gs.iloc[0]
            ax.errorbar([60 * 1.12], [r.ratio], yerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]],
                        fmt="D", color=C_OBS, ms=5, capsize=2, lw=1, label="vs AEMET/HyMEX gauges, 1 h")
            lines.append(f"  {lab} cell AEMET/HyMEX 1 h: {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]")
        gr = rad[(rad.region == region) & (rad.scale_km == 2) & np.isclose(rad.prob, p)]
        if not gr.empty and np.isfinite(gr.ratio.iloc[0]) and gr.ratio.iloc[0] > 0:
            r = gr.iloc[0]
            ax.errorbar([60 / 1.12], [r.ratio], yerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]],
                        fmt="s", color=C_RAD, ms=5, capsize=2, lw=1, label="vs EURADCLIM, 1 h, 2 km")
            lines.append(f"  {lab} EURADCLIM 1 h: {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]")
        ax.axhline(1, color=INK_MUTED, lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(dur); ax.set_xticklabels([TICK[w] for w in dur]); ax.minorticks_off()
        ax.set_ylim(0.5, 2.0)
        ax.set_ylabel("EPICC / observed", fontsize=8, color=INK)
        ax.set_xlabel("accumulation window", fontsize=8, color=INK)
        style(ax)
    title(axes[0, 0], "a  p99 of all days (all hours at 1 h)")
    title(axes[0, 1], "b  p99.9")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(h, l, fontsize=6.5, frameon=False, loc="upper left")

    # station-season maxima, ranked
    lat, lon = mx.lat.values, mx.lon.values
    rm = regions.masks(lat, lon, box=BOX)[region]
    for ax, v, lab in ((axes[1, 0], "p24", "daily total"), (axes[1, 1], "pmax60", "60-min maximum")):
        g = mx[(mx["var"] == v) & rm]
        if len(g) < 20:
            title(ax, f"{'cd'[v == 'pmax60']}  season maxima, {lab} (n = {len(g)})")
            style(ax)
            continue
        o = g.arnau.values
        xg = gumbel_axis(o.size)
        ax.plot(xg, np.sort(o), color=C_OBS, lw=2, label="AEMET daily record")
        ax.plot(xg, np.sort(g.cell.values), color=C_MOD, lw=1.8, label="EPICC, model cell")
        ax.plot(xg, np.sort(g.nmax.values), color=C_MOD, lw=1.6, ls="--", alpha=0.55, label="EPICC, 3x3 max")
        for T in (2, 10, 100):
            x = -np.log(-np.log(1 - 1 / T))
            ax.axvline(x, color=GRID, lw=0.7)
            ax.text(x, 0.97, f" 1 in {T}", fontsize=6.5, color=INK_MUTED, va="top", ha="left",
                    transform=ax.get_xaxis_transform())
        r10c, r1c = ranked_ratio(o, g.cell.values, 0.1), ranked_ratio(o, g.cell.values, 0.01)
        r10n, r1n = ranked_ratio(o, g.nmax.values, 0.1), ranked_ratio(o, g.nmax.values, 0.01)
        lines.append(f"  season maxima {lab}: n={len(g)}; top 10% ratio cell {r10c:.2f} nmax {r10n:.2f}; "
                     f"top 1% cell {r1c:.2f} nmax {r1n:.2f}; paired median cell {np.median(g.cell / g.arnau):.2f}")
        ax.text(0.98, 0.04, f"top 10%: ×{r10c:.2f} (cell), ×{r10n:.2f} (3x3)\ntop 1%: ×{r1c:.2f}, ×{r1n:.2f}",
                transform=ax.transAxes, fontsize=7, color=INK, ha="right", va="bottom")
        title(ax, f"{'cd'[v == 'pmax60']}  season maxima, {lab} ({len(g)} station-seasons)")
        ax.set_xlabel("Gumbel reduced variate (ranked station-seasons)", fontsize=8, color=INK)
        ax.set_ylabel("mm", fontsize=8, color=INK)
        style(ax)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(f"Rain intensity across time scales, EPICC 2 km vs observations, {regions.LONG[region]}, "
                 f"{tag} 2011–2019/2020", fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig1_scales", region, tag, lines)


###########################################################
# Figure 2: where
###########################################################

def fig2_where(tag, region, months):
    years = range(rcfg.syear, rcfg.eyear + 1)
    # 10 km block statistics: the 2 km ratio field is too noisy to read as a map
    with __import__("xarray").open_dataset(rcfg.mask_file) as m:
        mask = m.mask_s5.values.astype(bool)
    rad = ru.load_stats("EURADCLIM", 5, years, months)
    mod = ru.load_stats(rcfg.model_run, 5, years, months)
    lat, lon = rad.lat.values, rad.lon.values
    fr, fm = radar_fields(rad, 0.999), radar_fields(mod, 0.999)
    rmask = regions.masks(lat, lon, box=BOX)[region] & mask
    # stations: joint sample of gauges, EPICC and EURADCLIM
    d = psm.load(months)
    valid = psm.joint_valid(d)
    st = {k: psm.station_stats(x, valid, [0.999], {"mean": 2000})
          for k, x in (("obs", d["obs"]), ("mod", d["mod"]["cell"]), ("rad", d["rad"]["cell"]))}
    srm = psm.region_of(d["lat"], d["lon"])[region]
    extent = region_extent(region)
    lines = [f"Figure 2, {region}, {tag}"]

    fig = new_fig(10.0, 7.6)
    panels = [("a", "hourly P99.9: EPICC / EURADCLIM (10 km), EPICC / gauges (dots)", "quant", 0.999, "mod", "rad", "mod", "obs"),
              ("b", "mean rain: EPICC / EURADCLIM (10 km), EPICC / gauges (dots)", "mean", "mean", "mod", "rad", "mod", "obs"),
              ("c", "hourly P99.9: EURADCLIM / gauges (AEMET/HyMEX)", None, 0.999, None, None, "rad", "obs"),
              ("d", "mean rain: EURADCLIM / gauges (AEMET/HyMEX)", None, "mean", None, None, "rad", "obs")]
    for i, (let, ttl, fkey, skey, fa, fb, sa, sb) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
        base_map(ax, extent)
        if fkey is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                field = np.where(mask & (fr[fkey] > 0) & (fm[fkey] > 0), np.log2(fm[fkey] / fr[fkey]), np.nan)
            mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), shading="nearest", cmap=DIV.reversed(),
                                 norm=RNORM, transform=ccrs.PlateCarree(), rasterized=True)
            ok = rmask & np.isfinite(field)
            lines.append(f"  {let} field: median ratio {2 ** np.nanmedian(field[ok]):.2f} over {ok.sum()} cells, "
                         f"regional mean ratio {np.nanmean(fm[fkey][ok]) / np.nanmean(fr[fkey][ok]):.2f}")
        else:
            mesh = None
        num, den = st[sa][skey], st[sb][skey]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where((num > 0) & (den > 0), np.log2(num / den), np.nan)
        ok = np.isfinite(r)
        sc = ax.scatter(d["lon"][ok], d["lat"][ok], c=r[ok], cmap=DIV.reversed(), norm=RNORM, s=13,
                        edgecolor=BG if mesh is None else INK, linewidth=0.35, transform=ccrs.PlateCarree(), zorder=4)
        sel = ok & srm
        if sel.any():
            lines.append(f"  {let} stations: median ratio {2 ** np.median(r[sel]):.2f} [{np.mean(r[sel] > 0):.0%} > 1], "
                         f"{sel.sum()} stations")
        ratio_colorbar(fig, mesh if mesh is not None else sc, ax, "ratio")
        title(ax, f"{let}  {ttl}", size=8.5)
    fig.suptitle(f"Where the model differs: EPICC 2 km against EURADCLIM and the AEMET/HyMEX gauges, "
                 f"{regions.LONG[region]}, {tag} 2013–2020 (radar) / 2011–2020 (gauges)",
                 fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(left=0.04, right=0.93, top=0.9, bottom=0.04, wspace=0.22, hspace=0.18)
    save(fig, "fig2_where", region, tag, lines)


###########################################################
# Figure 3: when
###########################################################

def diurnal(x, m, hod):
    return np.array([np.nanmean(x[:, hod == h][m[:, hod == h]]) if m[:, hod == h].any() else np.nan
                     for h in range(24)])


def fig3_when(tag, region, months):
    d = psm.load(months)
    regs = psm.region_of(d["lat"], d["lon"])
    rm = regs[region]
    hod = d["times"].hour.values
    has_gauges = rm.sum() >= 5
    valid = psm.joint_valid(d) & rm[:, None]
    AFT, NIG = (hod >= 12) & (hod <= 18), hod <= 8
    lines = [f"Figure 3, {region}, {tag}: {rm.sum()} hourly gauges"]
    fig = new_fig(10.8, 6.8)
    axes = fig.subplots(2, 3)

    # a, b: rain at the gauges by hour, joint sample
    series = [("AEMET/HyMEX gauges", d["obs"], C_OBS, "-"), ("EURADCLIM", d["rad"]["cell"], C_RAD, "-"),
              ("EPICC cell", d["mod"]["cell"], C_MOD, "-")]
    if has_gauges:
        for name, x, col, ls in series:
            mean = diurnal(x, valid, hod)
            freq = 100 * np.array([np.mean(x[:, hod == h][valid[:, hod == h]] >= cfg.wet_thres) for h in range(24)])
            axes[0, 0].plot(np.arange(24) + 0.5, mean, color=col, lw=1.8, ls=ls, label=name)
            axes[0, 1].plot(np.arange(24) + 0.5, freq, color=col, lw=1.8, ls=ls, label=name)
            a = np.nanmean(x[:, AFT][valid[:, AFT]]) / np.nanmean(x[:, NIG][valid[:, NIG]])
            lines.append(f"  {name:18s} afternoon(12-19)/night(00-09) mean rain {a:.2f}")
    title(axes[0, 0], "a  mean rain at the gauges, by hour")
    axes[0, 0].set_ylabel("mm h$^{-1}$", fontsize=8, color=INK)
    title(axes[0, 1], "b  wet hours (≥ 0.1 mm), by hour")
    axes[0, 1].set_ylabel("% of hours", fontsize=8, color=INK)
    axes[0, 0].legend(fontsize=6.5, frameon=False)

    # c: storm hours at the gauges, each side its own storms (like for like)
    s = psr.load_all(months)
    sv = s["valid"] & rm[:, None]
    for name, key, col, ls in ((("observed storms (IMERG + MERGIR)", "obs", C_OBS, "-"),
                                ("EPICC storms, tracked at 0.1°", "mod01", C_MOD, "-")) if has_gauges else ()):
        m = (s["storms"][key] > 0) & sv
        frac = 100 * np.array([m[:, hod == h].sum() / max(sv[:, hod == h].sum(), 1) for h in range(24)])
        axes[0, 2].plot(np.arange(24) + 0.5, frac, color=col, lw=1.8, ls=ls, label=name)
        lines.append(f"  storm hours {key:6s} aft/night {frac[12:19].mean() / frac[0:9].mean():.2f}, "
                     f"{m.sum() / max(sv.sum(), 1) * 8760:.1f} h per station-year")
    title(axes[0, 2], "c  storm overhead at the gauges, by hour")
    axes[0, 2].set_ylabel("% of hours", fontsize=8, color=INK)
    axes[0, 2].legend(fontsize=6.5, frameon=False)

    # d: cold cloud tops, MERGIR vs model Tb
    tb = pd.read_csv(f"{FIG}/tb/tb_eval_{tag}_diurnal.csv")
    for name, col, ls, lab in (("MERGIR", C_OBS, "-", "MERGIR"), ("EPICC YS", C_MOD, "-", "EPICC Tb (Yang & Slingo)"),
                               ("EPICC SB", C_SB, "-", "EPICC Tb (Stefan–Boltzmann)")):
        g = tb[(tb.region == region) & (tb.dataset == name)].sort_values("hour")
        axes[1, 0].plot(g.hour + 0.5, 100 * g.frac225, color=col, lw=1.8, ls=ls, label=lab)
        lines.append(f"  cold cores <= 225 K, {name:9s}: {100 * (g.frac225 * g.n).sum() / g.n.sum():.2f}% of cell-hours, "
                     f"aft/night {g.frac225.values[12:19].mean() / g.frac225.values[0:9].mean():.2f}")
    title(axes[1, 0], "d  cold cloud tops (Tb ≤ 225 K), by hour")
    axes[1, 0].set_ylabel("% of cell-hours", fontsize=8, color=INK)
    axes[1, 0].legend(fontsize=6.5, frameon=False)

    # e: seasonal cycle of mean rain at the gauges, joint sample, all months
    if has_gauges:
        da = psm.load(list(range(1, 13)))
        va = psm.joint_valid(da) & rm[:, None]
        mon = da["times"].month.values
        for name, x, col, ls in (("AEMET/HyMEX gauges", da["obs"], C_OBS, "-"), ("EURADCLIM", da["rad"]["cell"], C_RAD, "-"),
                                 ("EPICC cell", da["mod"]["cell"], C_MOD, "-")):
            prof = [24 * np.nanmean(x[:, mon == k][va[:, mon == k]]) for k in range(1, 13)]
            axes[1, 1].plot(range(1, 13), prof, color=col, lw=1.8, ls=ls, marker="o", ms=3, label=name)
            lines.append(f"  seasonal {name:18s} " + " ".join(f"{p:.2f}" for p in prof))
        axes[1, 1].set_xticks(range(1, 13)); axes[1, 1].set_xticklabels(list("JFMAMJJASOND"))
    title(axes[1, 1], "e  seasonal cycle at the gauges")
    axes[1, 1].set_ylabel("mm day$^{-1}$", fontsize=8, color=INK)

    # f: afternoon/night ratio by station altitude, gauges vs EPICC
    if has_gauges:
        alt = pd.read_csv(cfg.file_stations).set_index("code").reindex(d["codes"])["alt"].values
        bands = [(-1, 100), (100, 400), (400, 800), (800, 4000)]
        xs = np.arange(len(bands))
        for name, x, col, off in (("AEMET/HyMEX gauges", d["obs"], C_OBS, -0.15), ("EPICC cell", d["mod"]["cell"], C_MOD, 0.15)):
            vals, ns = [], []
            for lo, hi in bands:
                sb = rm & (alt > lo) & (alt <= hi)
                v = psm.joint_valid(d) & sb[:, None]
                if sb.sum() < 3:
                    vals.append(np.nan); ns.append(sb.sum()); continue
                vals.append(np.nanmean(x[:, AFT][v[:, AFT]]) / np.nanmean(x[:, NIG][v[:, NIG]]))
                ns.append(sb.sum())
            axes[1, 2].bar(xs + off, vals, width=0.3, color=col, label=name)
            lines.append(f"  aft/night by altitude {name:18s} " + " ".join(f"{v:.2f}(n={n})" for v, n in zip(vals, ns)))
        axes[1, 2].set_xticks(xs); axes[1, 2].set_xticklabels(["<100 m", "100–400", "400–800", ">800 m"], fontsize=7)
        axes[1, 2].axhline(1, color=INK_MUTED, lw=0.8)
        axes[1, 2].legend(fontsize=6.5, frameon=False)
    title(axes[1, 2], "f  afternoon / night rain, by altitude")
    axes[1, 2].set_ylabel("12–19 UTC / 00–09 UTC", fontsize=8, color=INK)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel("hour (UTC)", fontsize=8, color=INK)
    for ax in axes.ravel():
        ax.set_ylim(bottom=0)
        style(ax)
    if not has_gauges:
        for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
            ax.text(0.5, 0.5, f"no hourly gauge comparison:\n{rm.sum()} AEMET/HyMEX station{'s' if rm.sum() != 1 else ''} "
                    f"in {regions.LONG[region]}", transform=ax.transAxes, ha="center", va="center", fontsize=8,
                    color=INK_MUTED)
            ax.set_yticks([])
    fig.suptitle(f"When it rains: diurnal and seasonal cycles, EPICC 2 km vs observations, {regions.LONG[region]}, "
                 f"{tag} 2011–2020", fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig3_when", region, tag, lines)


###########################################################
# Figure 4: storms
###########################################################

def load_storms(dataset, region, months, exp="exp1"):
    """Per-storm characteristics; a storm counts if its centre enters the region (+50 km)."""
    root = f"{ocfg.path_tracking}/{dataset}/{exp}"
    rows = []
    for year in range(ocfg.syear, ocfg.eyear + 1):
        for month in months:
            fin = f"{root}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue
            for storm in pd.read_pickle(fin).values():
                track = storm["track"]
                m = regions.masks(track[:, 0], track[:, 1], box=[ocfg.lat_min, ocfg.lon_min, ocfg.lat_max, ocfg.lon_max],
                                  buffer_deg=0.5, names=[region] if region != "ALL" else [])[region]
                if not m.any():
                    continue
                rows.append({"year": year, "month": month, "start_hour": pd.Timestamp(storm["times"][0]).hour,
                             "area": np.nanmax(storm["size"]) / 1e6, "duration": len(storm["times"]),
                             "peak": np.nanmax(storm["max"]), "volume": np.nansum(storm["volume"]) / 1e6})
    return pd.DataFrame(rows)


def cdf(ax, frames, key, logx):
    for name, (frame, col, ls) in frames.items():
        if frame.empty:
            continue
        v = np.sort(frame[key].values)
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, lw=1.8, ls=ls, label=name)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_ylabel("cumulative fraction", fontsize=8, color=INK)


def fig4_storms(tag, region, months, radar_pair=False):
    if radar_pair:
        sets = {"observed storms (EURADCLIM + MERGIR)": ("rad", C_OBS, "-"),
                "EPICC storms, radar coverage": ("mod0.1_YS_pres_radcov", C_MOD, "-")}
        name, years = "fig4b_storms_radar", "2013–2020"
    else:
        sets = {"observed storms (IMERG + MERGIR)": ("obs", C_OBS, "-"),
                "EPICC storms (Yang & Slingo Tb)": ("mod0.1_YS_pres", C_MOD, "-"),
                "EPICC storms (Stefan–Boltzmann Tb)": ("mod0.1_SB_pres", C_SB, "-")}
        name, years = "fig4_storms", "2011–2020"
    frames = {}
    for lab, (ds, col, ls) in sets.items():
        f = load_storms(ds, region, months)
        if f.empty:
            logging.warning("%s: no storms for %s", ds, region)
            continue
        frames[lab] = (f, col, ls)
    obs_lab = list(sets)[0]
    if obs_lab not in frames:
        logging.warning("no observed storms; skipping figure 4 for %s", region)
        return
    nyears = len(set().union(*[set(f.year) for f, _, _ in frames.values()]))
    lines = [f"Figure 4{'b' if radar_pair else ''}, {region}, {tag}: storms per year " +
             ", ".join(f"{lab} {len(f) / nyears:.1f}" for lab, (f, _, _) in frames.items())]
    maps = region == "ALL" and not radar_pair
    fig = new_fig(11.0, 7.0 if maps else 6.6)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(2, 3, 1)
    for lab, (f, col, ls) in frames.items():
        c = f.groupby("month").size().reindex(months, fill_value=0) / nyears
        ax.plot(c.index, c.values, color=col, lw=1.8, ls=ls, marker="o", ms=3.5, label=lab)
    ax.set_xticks(months); ax.set_xticklabels([list("JFMAMJJASOND")[m - 1] for m in months])
    ax.set_ylabel("storms per month", fontsize=8, color=INK); title(ax, "a  seasonal cycle of storm counts"); style(ax)
    ax.legend(fontsize=6.5, frameon=False)
    ax = fig.add_subplot(2, 3, 2)
    for lab, (f, col, ls) in frames.items():
        c = f.groupby("start_hour").size().reindex(range(24), fill_value=0) / nyears
        ax.plot(np.arange(24) + 0.5, c.values, color=col, lw=1.8, ls=ls, label=lab)
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel("hour (UTC)", fontsize=8, color=INK)
    ax.set_ylabel("storms per year", fontsize=8, color=INK); title(ax, "b  hour of storm initiation"); style(ax)
    ax = fig.add_subplot(2, 3, 3)
    cdf(ax, frames, "volume", True); ax.set_xlabel("rain volume (10$^6$ m$^3$)", fontsize=8, color=INK)
    title(ax, "c  storm rain volume"); style(ax)
    rng = np.random.default_rng(0)
    obs = frames[obs_lab][0]
    for lab, (f, _, _) in frames.items():
        if lab == obs_lab:
            continue
        for key in ("area", "duration", "peak", "volume"):
            res = tail_ratio_ci(obs, f, key, rng)
            lines.append(f"  {lab:38s} {key:8s} " + "  ".join(f"p{q} {v[0]:.2f} [{v[1]:.2f}-{v[2]:.2f}]" for q, v in res.items()))
    if maps:
        tracks = {ds: load_tracks(ds, "exp1") for _, (ds, _, _) in sets.items()}
        fields = {}
        for ds, t in tracks.items():
            if t is None:
                continue
            f, lat_e, lon_e = density(t[0], t[1], t[2])
            fields[ds] = relative(f)
        vmax = np.percentile(np.concatenate([f[f > 0] for f in fields.values()]), 99)
        for i, (lab, (ds, col, ls)) in enumerate(list(sets.items())[:2]):
            ax = fig.add_subplot(2, 3, 4 + i, projection=proj)
            mesh = add_map(ax, lat_e, lon_e, fields[ds], SEQ, vmax=vmax)
            regions.outline(ax, lw=0.5, color="#6a6a66")
            title(ax, f"{'de'[i]}  track density, {lab.split(' (')[0].replace(' storms', '')}")
            if i == 1:
                inset_cbar(fig, mesh, ax, "density / own domain mean", extend="max")
        ax = fig.add_subplot(2, 3, 6, projection=proj)
        diff = fields["mod0.1_YS_pres"] - fields["obs"]
        dmax = np.percentile(np.abs(diff), 99)
        mesh = add_map(ax, lat_e, lon_e, diff, DIV, norm=TwoSlopeNorm(0, -dmax, dmax))
        regions.outline(ax, lw=0.5, color="#6a6a66")
        inset_cbar(fig, mesh, ax, "EPICC − observed")
        title(ax, "f  difference in relative density")
    else:
        for i, (key, lab, logx) in enumerate((("area", "maximum area (km$^2$)", True), ("duration", "duration (h)", False),
                                               ("peak", "peak rain rate (mm h$^{-1}$)", False))):
            ax = fig.add_subplot(2, 3, 4 + i)
            cdf(ax, frames, key, logx); ax.set_xlabel(lab, fontsize=8, color=INK)
            title(ax, f"{'def'[i]}  storm {key}"); style(ax)
    what = "EURADCLIM + MERGIR, radar coverage only" if radar_pair else "IMERG + MERGIR"
    fig.suptitle(f"Tracked storms, EPICC coarsened to 0.1° vs {what}, {regions.LONG[region]}, {tag} {years}",
                 fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, name, region, tag, lines)


###########################################################

def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--regions", nargs="+", default=["ALL"])
    par.add_argument("--figs", nargs="+", default=["1", "2", "3", "4", "4b"])
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    months, tag = seasons.resolve(args)
    for region in args.regions:
        for f in args.figs:
            logging.info("figure %s, %s", f, region)
            try:
                if f == "1":
                    fig1_scales(tag, region, months)
                elif f == "2":
                    fig2_where(tag, region, months)
                elif f == "3":
                    fig3_when(tag, region, months)
                elif f == "4":
                    fig4_storms(tag, region, months)
                elif f == "4b":
                    fig4_storms(tag, region, months, radar_pair=True)
            except Exception:
                logging.exception("figure %s failed for %s", f, region)


if __name__ == "__main__":
    main()
